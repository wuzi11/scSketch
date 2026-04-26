# -*- coding: utf-8 -*-
# @Author: Zijian Yuan
# @Last Modified by:   Zijian Yuan
# @Last Modified time: 2026-04-25

import io
import os
import socket

import torch as th
import torch.distributed as dist
import argparse
from datetime import datetime
from scSketch import logger
from scSketch.runtime import dev

from scSketch.scrna_datasets import prepared_data
from scSketch.resample import create_named_schedule_sampler
from scSketch.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    str2bool,
)
from scSketch.train_util import TrainLoop,plot_loss

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')



GPUS_PER_NODE = 1  # Set this to the actual number of GPUs per node

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across ranks.
    """
    chunk_size = 2 ** 30  # Size limit for data chunks
    if dist.get_rank() == 0:
        with open(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        dist.broadcast(th.tensor(num_chunks), 0)
        for i in range(0, len(data), chunk_size):
            dist.broadcast(th.tensor(data[i: i + chunk_size]), 0)
    else:
        num_chunks = dist.broadcast(th.tensor(0), 0).item()
        data = bytes()
        for _ in range(num_chunks):
            chunk = th.zeros(chunk_size, dtype=th.uint8)
            dist.broadcast(chunk, 0)
            data += bytes(chunk.numpy())

    return th.load(io.BytesIO(data), **kwargs)

def run_training(args):
        
    logger.configure(dir=args['logger_path'])
    logger.log("*********creating model and diffusion**********")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.to(dev())
    schedule_sampler = create_named_schedule_sampler(args['schedule_sampler'], diffusion)

    logger.log("creating data loader...")
    data, sketch_info = prepared_data(
        data_dir = args['data_path'],
        control_data_dir = args.get('control_data_path', None),
        batch_size = args['batch_size'],
        use_drug_structure= args['use_drug_structure'],
        comb_num = args['comb_num'],
        progeny_model_path=args.get('progeny_model_path', None),
        progeny_organism=args.get('progeny_organism', 'human'),
    )
    #logger.log(f'with gpu {dev()}')
    start_time = datetime.now()
    logger.log(f'**********training started at {start_time} **********')
    train_ = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args['batch_size'],
        microbatch=args['microbatch'],
        lr=args['lr'],
        ema_rate=args['ema_rate'],
        log_interval=args['log_interval'],
        save_interval=args['save_interval'],
        resume_checkpoint=args['resume_checkpoint'],
        use_fp16=args['use_fp16'],
        fp16_scale_growth=args['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=args['weight_decay'],
        lr_anneal_steps=args['lr_anneal_steps'],
        use_drug_structure= args['use_drug_structure'],
        comb_num=args['comb_num'],
        sketch_info=sketch_info,
        progeny_calculator=sketch_info.get('progeny_calculator', None) if sketch_info else None,
    )
    train_.run_loop()
    
    end_time = datetime.now()

    during_time = (end_time-start_time).seconds/60

    logger.log(f'start time: {start_time} end_time: {end_time} time:{during_time} min')
    
    return train_.loss_list


def parse_args():
    """Parse command-line arguments and update with default values."""
    # Define default arguments
    default_args = {}
    default_args.update(model_and_diffusion_defaults())
    fixed_args = {
        'use_dit1d': True,
        'use_drug_structure': True,
        'progeny_organism': 'human',
        'progeny_model_path': None,
        'use_dpm_solver': False,
    }
    updated_args = {
        'data_path': '',
        'schedule_sampler': 'uniform',
        'lr': 1e-4,
        'weight_decay': 0.0,
        'lr_anneal_steps': 1e5,
        'batch_size': 32,
        'microbatch': -1,
        'ema_rate': '0.9999',
        'log_interval': 1e4,
        'save_interval': 5e4,
        'resume_checkpoint': '',
        'use_fp16': False,
        'fp16_scale_growth': 1e-3,
        'gene_size': 2000,
        'output_dim': 100,
        'num_layers': 3,
        'class_cond': False,
        'use_encoder': True,
        'diffusion_steps': 1000,
        'logger_path': '',
        'use_drug_structure': True,
        'comb_num': 1,
        # DiT1D parameters
        'use_dit1d': True,
        'dit_hidden_size': 768,
        'dit_num_heads': 12,
        'dit_mlp_ratio': 4.0,
        'dit_class_dropout_prob': 0.1,
        'dit_latent_dim': 60,
        'dit1d_use_value_embedding': True,
        'control_data_path': '',
    }
    default_args.update(updated_args)
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Perturbation-conditioned generative diffusion model')
    
    # Add arguments to the parser (these should correspond to the keys in default_args)
    for key, value in default_args.items():
        if key in fixed_args:
            continue
        v_type = type(value)
        if value is None:
            v_type = str
        elif isinstance(value, bool):
            v_type = str2bool
        parser.add_argument(f'--{key}', default=value, type=v_type, help=f'{key} (default: {value})')

    # Parse command-line arguments
    args = parser.parse_args()

    # Convert the parsed arguments to a dictionary and update the defaults
    updated_args = vars(args)
    updated_args.update(fixed_args)
    updated_args['num_layers'] = 12
    
    # Check if 'logger_path' is None and raise an error if so
    if updated_args['logger_path']=='':
        logger.log('ERROR:Please specify the logger path --logger_path.')
        raise ValueError("Logger path is required. Please specify the logger path.")

            # Check if 'logger_path' is None and raise an error if so
    if updated_args['data_path']=='':
        logger.log("ERROR:Please specify the data path --data_path.")
        raise ValueError("Dataset path is required. Please specify the path where the training adata is.")

    if updated_args.get('control_data_path', '') == '':
        logger.log("ERROR:Please specify the control data path --control_data_path.")
        raise ValueError("Control dataset path is required when use_drug_structure is enabled.")


    # Return the updated arguments as a dictionary
    return updated_args



if __name__ == "__main__":
    args_train = parse_args()
    print('**************training args*************')
    print(args_train)
    losses = run_training(args_train)
    
    plot_loss(losses,args_train)
    
    

