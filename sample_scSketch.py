# -*- coding: utf-8 -*-
# @Author: Zijian Yuan
# @Last Modified by:   Zijian Yuan
# @Last Modified time: 2026-04-25


import argparse
import os
import numpy as np
import torch.distributed as dist
import torch
from scSketch import logger
from scSketch.runtime import dev, load_state_dict
from scSketch.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from scSketch.resample import create_named_schedule_sampler
from sklearn.metrics import r2_score
import scipy
class sampler:
    def __init__(self,model_path,gene_size,output_dim,hidden_size=768,dit_hidden_size=768,dit_num_heads=12,dit_mlp_ratio=4.0,dit1d_use_value_embedding=True,dit1d_patch_size=1,num_layers=None,timestep_respacing=''):
        args = self.parse_args(model_path,gene_size,output_dim,hidden_size,dit_hidden_size,dit_num_heads,dit_mlp_ratio,dit1d_use_value_embedding,dit1d_patch_size,num_layers,timestep_respacing)
        print("load model and diffusion...")

        model, diffusion = create_model_and_diffusion(
                **args_to_dict(args, model_and_diffusion_defaults().keys())
            )

        state_dict = load_state_dict(args['model_path'])
        # Use strict=False to allow loading models without sketch parameters
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        model.to(dev())
        if args['use_fp16']:
            model.convert_to_fp16()
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        self.model = model
        self.arg = args
        self.diffusion = diffusion
        # Select sampling function based on method
        if args['use_ddim']:
            self.sample_fn = diffusion.ddim_sample_loop
        else:
            self.sample_fn = diffusion.p_sample_loop
    
    def stochastic_encode(
        self, model, x, t, model_kwargs):
        """
        ddim reverse sample
        """
        sample = x
        sample_t = []
        xstart_t = []
        T = []
        indices = list(range(t))

        for i in indices:
            timestep = torch.full((x.shape[0],), i, device='cuda').long()
            with torch.no_grad():
                out = self.diffusion.ddim_reverse_sample(model, 
                                                    sample, 
                                                    timestep, 
                                                    model_kwargs=model_kwargs)
                sample = out['sample']
                sample_t.append(sample)
                xstart_t.append(out['pred_xstart'])
                T.append(timestep)

        return {
        'sample': sample,
        'sample_t': sample_t,
        'xstart_t': xstart_t,
        'T': T,
    }

    def parse_args(self,model_path,gene_size,output_dim,hidden_size=768,dit_hidden_size=768,dit_num_heads=12,dit_mlp_ratio=4.0,dit1d_use_value_embedding=True,dit1d_patch_size=1,num_layers=None,timestep_respacing=''):
        """Parse command-line arguments and update with default values."""
        # Define default arguments
        default_args = {}
        default_args.update(model_and_diffusion_defaults())
        
        if num_layers is None:
            num_layers = 12
        
        updated_args = {
            'data_path': '',
            'schedule_sampler': 'uniform',
            'lr': 1e-4,
            'weight_decay': 0.0,
            'lr_anneal_steps': 1e5,
            'batch_size': 128,
            'microbatch': -1,
            'ema_rate': '0.9999',
            'log_interval': 1e4,
            'save_interval': 1e4,
            'resume_checkpoint': '',
            'use_fp16': False,
            'fp16_scale_growth': 1e-3,
            'gene_size': gene_size,
            'output_dim': output_dim,
            'hidden_sizes': hidden_size,  # MLP model hidden size
            'num_layers': num_layers,
            'class_cond': False,
            'use_encoder': True,
            'use_ddim': True,
            'timestep_respacing': timestep_respacing,
            'diffusion_steps': 1000,
            'logger_path': '',
            'model_path': model_path,
            'use_drug_structure': True,
            'comb_num':1,
            'drug_dimension':1024,
            'use_dit1d': True,
            'dit_hidden_size': dit_hidden_size,
            'dit_num_heads': dit_num_heads,
            'dit_mlp_ratio': dit_mlp_ratio,
            'dit1d_use_value_embedding': dit1d_use_value_embedding,
            'dit1d_patch_size': dit1d_patch_size,
        }
        default_args.update(updated_args)

        # Return the updated arguments as a dictionary
        return default_args

    def load_scSketch_model(self):
        print("load model and diffusion...")
        return self.model

    def load_sample_fn(self):
        
        return self.sample_fn

    def get_diffused_data(self,model, x, t, model_kwargs):
        sample = x
        sample_t = [x]  # Store initial data for plotting
        xstart_t = []
        T = []

        indices = list(range(t))

        for i in indices:
            timestep = torch.full((x.shape[0],), i, device='cuda').long()
            with torch.no_grad():
                # Replacing ddim_reverse_sample with a simpler forward diffusion process
                noise = torch.randn_like(sample)  # Add noise at each step
                out = sample + noise * (i / t)    # Simulating diffusion based on time step
                sample = out
                sample_t.append(sample.cpu())  # Store the samples for visualization
                xstart_t.append(sample.cpu())
                T.append(timestep)

        return {
            'sample': sample,
            'sample_t': sample_t,
            'xstart_t': xstart_t,
            'T': T
        }

    def sample_around_point(self, point, num_samples=None, scale=0.7):
        return point + scale * np.random.randn(num_samples, point.shape[0])

    def pred(self, z_sem, gene_size, batch_size=None, drug_dose=None, control_feature=None, delta_s_override=None):
        """
        delta_s_override: optional (N, 14) tensor for sketch ablation (real / noised / no sketch).
        """
        total_samples = z_sem.shape[0]
        
        # translated batch_size，translated
        if batch_size is None:
            # translated DiT translated，translated batch size
            if self.arg.get('use_dit1d', False):
                batch_size = 1024  # DiT translated
            else:
                batch_size = 64  # MLP translated batch
        
        # translated batch_size，translated
        if total_samples <= batch_size:
            with torch.no_grad():
                model_kwargs = {'z_mod': z_sem}
                # translated
                if drug_dose is not None:
                    model_kwargs['drug_dose'] = drug_dose
                if control_feature is not None:
                    model_kwargs['control_feature'] = control_feature
                if delta_s_override is not None:
                    model_kwargs['delta_s_override'] = delta_s_override.to(z_sem.device)
                
                # translated
                pred_result = self.sample_fn(
                    self.model,
                    shape=(total_samples, gene_size),
                    model_kwargs=model_kwargs,
                    noise=None
                )
            return pred_result
        
        # translated
        all_results = []
        print(f"Processing {total_samples} samples in batches of {batch_size}...")
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_z_sem = z_sem[i:end_idx]
            batch_size_actual = end_idx - i
            
            with torch.no_grad():
                if i > 0:
                    torch.cuda.empty_cache()
                
                model_kwargs = {'z_mod': batch_z_sem}
                # translated（translated）
                if drug_dose is not None:
                    model_kwargs['drug_dose'] = drug_dose[i:end_idx]
                if control_feature is not None:
                    model_kwargs['control_feature'] = control_feature[i:end_idx]
                if delta_s_override is not None:
                    model_kwargs['delta_s_override'] = delta_s_override[i:end_idx].to(z_sem.device)
                
                # translated
                batch_result = self.sample_fn(
                    self.model,
                    shape=(batch_size_actual, gene_size),
                    model_kwargs=model_kwargs,
                    noise=None
                )
                all_results.append(batch_result.cpu())
                print(f"  Processed batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
        
        # translated
        pred_result = torch.cat(all_results, dim=0).to(z_sem.device)
        return pred_result
    
    def interp_with_direction(self, z_sem_origin = None, gene_size = None, direction = None, scale = 1, add_noise_term = True):

        z_sem_origin = z_sem_origin.detach().cpu().numpy()
        z_sem_interp_ = z_sem_origin.mean(axis=0) + direction.detach().cpu().numpy() * scale
        if add_noise_term:
            z_sem_interp_ = self.sample_around_point(z_sem_interp_, num_samples=z_sem_origin.shape[0])

        z_sem_interp_ = torch.tensor(z_sem_interp_,dtype=torch.float32).to('cuda')
        # translated
        sample_interp = self.sample_fn(
                            self.model,
                            shape = (z_sem_origin.shape[0],gene_size),
                            model_kwargs={
                                'z_mod': z_sem_interp_
                            },
                            noise =  None
        )
        return sample_interp
        
    def cal_metric(self,x1,x2):
        r2 = r2_score(x1.detach().cpu().numpy().mean(axis=0),
                      x2.X.mean(axis=0))
        pearsonr,_ = scipy.stats.pearsonr(x1.detach().cpu().numpy().mean(axis=0),
                      x2.X.mean(axis=0))
        return r2, pearsonr

        

