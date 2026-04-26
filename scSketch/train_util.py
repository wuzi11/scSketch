"""
This code is adapted from openai's guided-diffusion models and Konpat's diffae model:
https://github.com/openai/guided-diffusion
https://github.com/phizaz/diffae
"""
import copy
import functools
import os

import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import logger
from .precision import MixedPrecisionTrainer
from .runtime import dev, load_state_dict, sync_params
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import matplotlib.pyplot as plt
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

INITIAL_LOG_LOSS_SCALE = 20.0

def plot_loss(losses,args_train):
    # Convert losses to a numpy array
    losses_np = np.array([i.detach().cpu().numpy() for i in losses])

    # Define the window size for the moving average
    window_size = int(args_train['lr_anneal_steps']/1000)+1  # You can adjust this value based on your preference

    # Calculate the moving average (mean of the windowed losses)
    windowed_mean_loss = np.convolve(losses_np, np.ones(window_size) / window_size, mode='valid')

    # Adjust the x-axis values for the windowed mean loss
    x_vals = np.linspace(0, args_train['lr_anneal_steps']-1, len(losses_np))
    windowed_x_vals = x_vals[window_size - 1:]  # Adjust to match the length of the windowed_mean_loss

    # Plotting
    fig,ax = plt.subplots(figsize=(4.5, 3.4), dpi=800)
    plt.plot(x_vals, losses_np, label='Training Loss', alpha=0.2)
    plt.plot(windowed_x_vals, windowed_mean_loss, label='Windowed Mean Loss', color='r', linewidth=1)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    fig.savefig(args_train['resume_checkpoint']+'/loss_plot.pdf', transparent=True)

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        use_drug_structure=False,
        comb_num=1,
        lambda_sketch=1.0,  # sketch loss translated (legacy)
        lambda_pathway=1.0,  # pathway loss translated (two-stage)
        lambda_consistency=1.0,  # consistency loss translated (post-hoc)
        sketch_info=None,  # Sketch information dict (type: 'progeny' or 'marker', data: sketches or indices)
        progeny_calculator=None,  # PROGENy calculator for consistency loss
    ):
        
        self.model = model
        self.diffusion = diffusion
        self.use_drug_structure = use_drug_structure
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.lambda_sketch = lambda_sketch
        self.lambda_pathway = lambda_pathway
        self.lambda_consistency = lambda_consistency
        self.use_two_stage = False
        # Store sketch information
        self.sketch_info = sketch_info
        self.progeny_calculator = progeny_calculator
        
        # Check if consistency loss is enabled
        self.use_consistency_loss = hasattr(model, 'use_consistency_loss') and model.use_consistency_loss
        if self.use_consistency_loss:
            if progeny_calculator is None:
                raise ValueError("use_consistency_loss=True requires progeny_calculator to be provided")
            print(f"\n=== Consistency Loss Mode ===")
            print(f"  Using post-hoc PROGENy consistency loss")
            print(f"  Lambda consistency: {self.lambda_consistency}")
            print(f"  PROGENy pathways: {len(progeny_calculator.pathway_names)}")
        
        # PROGENy sketch is required and always enabled.
        if sketch_info is None or sketch_info.get('type') != 'progeny':
            raise ValueError("PROGENy sketches are required when use_two_stage is disabled.")
        self.marker_indices = None
        self.use_progeny_sketch = True
        self.use_sketch = True
        print(f"Using PROGENy sketch loss with {sketch_info['n_pathways']} pathways")
        print(f"  Lambda sketch: {self.lambda_sketch}")

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size #* dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        self.loss_list = []
        #self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            #self.ddp_model = DDP(
            #    self.model,
            #    device_ids=[dist_util.dev()],
            #    output_device=dist_util.dev(),
            #    broadcast_buffers=False,
            #    bucket_cap_mb=128,
            #    find_unused_parameters=False,
            #)
            self.ddp_model = self.model
        else:
            # translated，translated get_world_size()
            if dist.is_initialized() and dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        
        # Freeze/unfreeze parameters based on training stage
        if self.use_two_stage:
            self._setup_stage_training()
        

    def _setup_stage_training(self):
        """Setup parameter freezing and checkpoint loading for separate stage training."""
        import os
        
        # Load Stage 1 checkpoint if training Stage 2
        if self.training_stage == 'stage2' and self.stage1_checkpoint:
            if os.path.exists(self.stage1_checkpoint):
                logger.log(f"Loading Stage 1 checkpoint from: {self.stage1_checkpoint}")
                checkpoint_path = self.stage1_checkpoint
                if os.path.isdir(self.stage1_checkpoint):
                    checkpoint_path = os.path.join(self.stage1_checkpoint, "model.pt")
                
                if os.path.exists(checkpoint_path):
                    state_dict = load_state_dict(checkpoint_path, map_location=dev())
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.log("Stage 1 checkpoint loaded successfully")
                else:
                    logger.log(f"Warning: Stage 1 checkpoint file not found: {checkpoint_path}")
            else:
                logger.log(f"Warning: Stage 1 checkpoint path does not exist: {self.stage1_checkpoint}")
        
        # Freeze/unfreeze parameters based on training stage
        if self.training_stage == 'stage1':
            # Stage 1: Only train pathway predictor, freeze everything else
            logger.log("\n=== Freezing diffusion model parameters (Stage 1 training) ===")
            frozen_count = 0
            trainable_count = 0
            
            for name, param in self.model.named_parameters():
                if 'pathway_predictor' in name:
                    param.requires_grad = True
                    trainable_count += 1
                else:
                    param.requires_grad = False
                    frozen_count += 1
            
            logger.log(f"  Frozen parameters: {frozen_count}")
            logger.log(f"  Trainable parameters (pathway predictor): {trainable_count}")
            
        elif self.training_stage == 'stage2':
            # Stage 2: Only train diffusion model, freeze pathway predictor
            logger.log("\n=== Freezing pathway predictor parameters (Stage 2 training) ===")
            frozen_count = 0
            trainable_count = 0
            
            for name, param in self.model.named_parameters():
                if 'pathway_predictor' in name:
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    param.requires_grad = True
                    trainable_count += 1
            
            logger.log(f"  Frozen parameters (pathway predictor): {frozen_count}")
            logger.log(f"  Trainable parameters (diffusion model): {trainable_count}")
            
        else:
            # Joint training: Train everything
            logger.log("\n=== Joint training: All parameters trainable ===")
            trainable_count = 0
            for param in self.model.parameters():
                param.requires_grad = True
                trainable_count += 1
            logger.log(f"  Trainable parameters: {trainable_count}")
        
        # Rebuild optimizer with only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        logger.log(f"  Optimizer rebuilt with {len(trainable_params)} trainable parameters")
    
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    load_state_dict(
                        resume_checkpoint, map_location=dev()
                    )
                )

        sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = load_state_dict(
                    ema_checkpoint, map_location=dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = os.path.join(
            os.path.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = load_state_dict(
                opt_checkpoint, map_location=dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
           
            
            batch = next(iter(self.data))

            self.run_step(batch)
            
            
            if self.step % self.log_interval == 0:
                # Add explicit sketch loss logging for visibility
                if self.use_sketch:
                    logger.log(f"Step {self.step}: Sketch loss metrics available in output")
                logger.dumpkvs()
            
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch):
        self.forward_backward(batch)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch):
        self.mp_trainer.zero_grad()
        
        for i in range(0, batch['feature'].shape[0], self.microbatch):
            
            micro = batch['feature'][i : i + self.microbatch].to(dev())
            if self.use_drug_structure:
                micro_cond = {
                    'group': batch['group'][i : i + self.microbatch],
                    'drug_dose': batch['drug_dose'][i : i + self.microbatch].to(dev()),
                    'control_feature':batch['control_feature'][i : i + self.microbatch].to(dev()),
                }
            else:
                micro_cond = {
                    'group': batch['group'][i : i + self.microbatch],
                    'drug_dose':None,
                    'control_feature':None
                }
            
            last_batch = (i + self.microbatch) >= batch['feature'].shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            elif hasattr(self.ddp_model, 'no_sync'):
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            else:
                losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            
            # Two-stage training: Add pathway loss
            model_to_check = self.ddp_model if hasattr(self.ddp_model, 'use_two_stage') else self.model
            
            # Debug: Log two-stage status at step 0
            if self.step == 0:
                logger.log(f"[DEBUG] self.use_two_stage: {self.use_two_stage}")
                logger.log(f"[DEBUG] hasattr(model_to_check, 'use_two_stage'): {hasattr(model_to_check, 'use_two_stage')}")
                if hasattr(model_to_check, 'use_two_stage'):
                    logger.log(f"[DEBUG] model_to_check.use_two_stage: {model_to_check.use_two_stage}")
                logger.log(f"[DEBUG] losses keys: {list(losses.keys())}")
            
            if self.use_two_stage and hasattr(model_to_check, 'use_two_stage') and model_to_check.use_two_stage:
                delta_s_hat = losses.get("s_hat", None)  # Reuse s_hat key for delta_s_hat
            
                
                if delta_s_hat is not None:
                    # Get ground truth Δs_true from batch
                    if 'progeny_sketch' in batch:
                        # Use pre-computed PROGENy Δs = PROGENy(x_treated) - PROGENy(x_control)
                        delta_s_true = batch['progeny_sketch'][i : i + self.microbatch].to(dev())
                        
                        # Stage 1 loss: L_path = ||Δs_hat - Δs_true||^2
                        loss_pathway = th.nn.functional.mse_loss(delta_s_hat, delta_s_true)
                        loss = loss + self.lambda_pathway * loss_pathway
                        
                        # Log pathway loss
                        logger.logkv_mean("pathway_loss", loss_pathway.item())
                        logger.logkv_mean("pathway_loss_weighted", (self.lambda_pathway * loss_pathway).item())
                        logger.logkv_mean("loss_pathway", loss_pathway.item())
                        
                        # Log pathway-level statistics
                        if self.step % self.log_interval == 0:
                            delta_s_hat_mean = delta_s_hat.mean(dim=0)
                            delta_s_true_mean = delta_s_true.mean(dim=0)
                            logger.log(f"  Pathway Δs_hat mean: {delta_s_hat_mean.abs().mean().item():.4f}")
                            logger.log(f"  Pathway Δs_true mean: {delta_s_true_mean.abs().mean().item():.4f}")
                            logger.log(f"  Pathway prediction MAE: {(delta_s_hat - delta_s_true).abs().mean().item():.4f}")
                    else:
                        if self.step % self.log_interval == 0:
                            logger.log("Warning: two-stage enabled but progeny_sketch not in batch")
                else:
                    if self.step % self.log_interval == 0:
                        logger.log("Warning: two-stage enabled but delta_s_hat is None")
            
            # Legacy sketch loss (if not using two-stage)
            elif self.use_sketch:
                s_hat = losses.get("s_hat", None)
                if s_hat is not None:
                    # Get ground truth sketch from batch
                    s_true = None
                    
                    if self.use_progeny_sketch and 'progeny_sketch' in batch:
                        # Use pre-computed PROGENy sketches from data loader
                        s_true = batch['progeny_sketch'][i : i + self.microbatch].to(dev())
                    
                    if s_true is not None:
                        loss_sketch = th.nn.functional.mse_loss(s_hat, s_true)
                        loss = loss + self.lambda_sketch * loss_sketch
                        # Log sketch loss (always record for averaging)
                        logger.logkv_mean("sketch_loss", loss_sketch.item())
                        logger.logkv_mean("sketch_loss_weighted", (self.lambda_sketch * loss_sketch).item())
                        # Add to main loss dict for visibility
                        logger.logkv_mean("loss_sketch", loss_sketch.item())
                        if self.use_progeny_sketch:
                            logger.logkv_mean("progeny_sketch_loss", loss_sketch.item())
                            # Log pathway-level statistics
                            if self.step % self.log_interval == 0:
                                s_hat_mean = s_hat.mean(dim=0)
                                s_true_mean = s_true.mean(dim=0)
                                logger.log(f"  Sketch prediction mean: {s_hat_mean.abs().mean().item():.4f}")
                                logger.log(f"  Sketch target mean: {s_true_mean.abs().mean().item():.4f}")
                    else:
                        if self.step % self.log_interval == 0:
                            logger.log("Warning: sketch enabled but s_true is None")
                else:
                    if self.step % self.log_interval == 0:
                        logger.log("Warning: sketch enabled but s_hat is None")
            
            # Consistency loss (post-hoc PROGENy alignment)
            if self.use_consistency_loss and self.progeny_calculator is not None:
                # Get x_0 prediction from the model
                # We need to predict x_0 from the noisy x_t
                if 'pred_xstart' in losses:
                    x_pred = losses['pred_xstart']
                else:
                    # Compute x_0 prediction from model output
                    # This depends on the diffusion parameterization
                    # For epsilon prediction: x_0 = (x_t - sqrt(1-alpha_bar) * epsilon) / sqrt(alpha_bar)
                    # We'll use the diffusion's _predict_xstart_from_eps method if available
                    model_output = losses.get('model_output', None)
                    if model_output is not None and hasattr(self.diffusion, '_predict_xstart_from_eps'):
                        # Get alpha_bar for timestep t
                        x_pred = self.diffusion._predict_xstart_from_eps(micro, t, model_output)
                    else:
                        # Skip consistency loss if we can't get x_0 prediction
                        x_pred = None
                
                if x_pred is not None and 'control_feature' in batch:
                    # Get control and predicted treated expression
                    x_control = micro_cond['control_feature']
                    x_treated_pred = x_pred
                    
                    # Get gene names from batch if available
                    gene_names = batch.get('gene_names', None)
                    
                    # Compute pathway activities using PROGENy
                    # Δs_pred = PROGENy(x_treated_pred) - PROGENy(x_control)
                    pathway_delta_pred = self.progeny_calculator.compute_pathway_delta(
                        expression_treated=x_treated_pred,
                        expression_control=x_control,
                        gene_names=gene_names,
                        normalize=True
                    )
                    
                    # Get ground truth pathway delta from batch (if pre-computed)
                    if 'progeny_sketch' in batch:
                        pathway_delta_true = batch['progeny_sketch'][i : i + self.microbatch].to(dev())
                    else:
                        # Compute on-the-fly from actual treated expression
                        pathway_delta_true = self.progeny_calculator.compute_pathway_delta(
                            expression_treated=micro,  # Use actual x_start
                            expression_control=x_control,
                            gene_names=gene_names,
                            normalize=True
                        )
                    
                    # Consistency loss: align predicted pathway changes with ground truth
                    loss_consistency = th.nn.functional.mse_loss(pathway_delta_pred, pathway_delta_true)
                    loss = loss + self.lambda_consistency * loss_consistency
                    
                    # Log consistency loss
                    logger.logkv_mean("consistency_loss", loss_consistency.item())
                    logger.logkv_mean("consistency_loss_weighted", (self.lambda_consistency * loss_consistency).item())
                    logger.logkv_mean("loss_consistency", loss_consistency.item())
                    
                    # Log pathway-level statistics
                    if self.step % self.log_interval == 0:
                        pathway_pred_mean = pathway_delta_pred.mean(dim=0)
                        pathway_true_mean = pathway_delta_true.mean(dim=0)
                        logger.log(f"  Consistency Δs_pred mean: {pathway_pred_mean.abs().mean().item():.4f}")
                        logger.log(f"  Consistency Δs_true mean: {pathway_true_mean.abs().mean().item():.4f}")
                        logger.log(f"  Pathway consistency MAE: {(pathway_delta_pred - pathway_delta_true).abs().mean().item():.4f}")
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items() if k != "s_hat"}
            )
            
            # Log total loss (important for monitoring training progress)
            logger.logkv_mean("total_loss", loss.item())
            
            self.mp_trainer.backward(loss)
            
        self.loss_list.append(loss)
        #print('loss=',loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        # Log current learning rate
        if len(self.opt.param_groups) > 0:
            logger.logkv("lr", self.opt.param_groups[0]["lr"])

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params) if self.mp_trainer else self.model.state_dict()
            if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not os.path.exists(self.resume_checkpoint):
                    # Directory doesn't exist, so create it
                    os.makedirs(self.resume_checkpoint)
                if not rate: 
                    filepath = os.path.join(self.resume_checkpoint, "model.pt")
                else:
                    filepath = os.path.join(self.resume_checkpoint, f"model_{rate}.pt")
                with open(filepath, "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params if self.mp_trainer else self.model.parameters())
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # Commented out: Do not save intermediate optimizer checkpoints during training
        # if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        #     opt_filename = f"opt{(self.step+self.resume_step):06d}.pt"
        #     opt_filepath = os.path.join(get_blob_logdir(), opt_filename)
        #     with open(opt_filepath, "wb") as f:
        #         th.save(self.opt.state_dict(), f)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = os.path.join(os.path.dirname(main_checkpoint), filename)
    if os.path.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
