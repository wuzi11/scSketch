import sys
import os
import argparse
import warnings

import pandas as pd
import numpy as np
import torch
import scipy
from sklearn.metrics import r2_score

import sample_scSketch
from scSketch.scrna_datasets import Drug_dose_encoder, read_h5ad_compat

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings('ignore')


def cal_metric(x1, x2):
    """Calculate R2 metric."""
    return r2_score(x1, x2)


def r2_mean(data1, data2):
    """Calculate mean R2 score across samples."""
    sum_r2_1 = 0
    for i in range(data1.shape[0]):
        r2_score_ = r2_score(data1[i], data2[i])
        sum_r2_1 += r2_score_
    return sum_r2_1/data1.shape[0]


def run_sampler(model_path, train_adata_path, control_adata_path, test_adata_path,
                gene_size=2000,
                dit_hidden_size=768, dit_num_heads=12,
                dit_mlp_ratio=4.0, dit1d_use_value_embedding=True, 
                dit1d_patch_size=1,
                num_layers=3, batch_size=1024, device='cuda',
                timestep_respacing='',
                delta_s_override=None):
    """
    Run sampler to generate predictions.
    
    Args:
        model_path: Path to the model checkpoint
        train_adata_path: Path to training data (h5ad file)
        control_adata_path: Path to control data (h5ad file)
        test_adata_path: Path to test data (h5ad file)
        gene_size: Gene size (default: 2000)
        dit_hidden_size: DiT hidden size (default: 768)
        dit_num_heads: DiT number of heads (default: 12)
        dit_mlp_ratio: DiT MLP ratio (default: 4.0)
        dit1d_use_value_embedding: Whether to use value embedding for DiT1D (default: True)
        num_layers: Number of layers (default: 3)
        batch_size: Batch size for prediction (default: 1024)
        device: Device to use (default: 'cuda')
    
    Returns:
        sample_interp: Generated samples
        test_adata: Test data
    """
    sampler = sample_scSketch.sampler(
        model_path=model_path,
        gene_size=gene_size,
        output_dim=gene_size,
        dit_hidden_size=dit_hidden_size,
        dit_num_heads=dit_num_heads,
        dit_mlp_ratio=dit_mlp_ratio,
        dit1d_use_value_embedding=dit1d_use_value_embedding,
        dit1d_patch_size=dit1d_patch_size,
        num_layers=num_layers,
        timestep_respacing=timestep_respacing,
    )
    
    test_adata = read_h5ad_compat(test_adata_path)
    control_adata = read_h5ad_compat(control_adata_path)
    drug_type_list = test_adata.obs['SMILES'].to_list()
    dose_list = test_adata.obs['dose'].to_list()
    encode_drug_doses = torch.tensor(
        Drug_dose_encoder(drug_type_list, dose_list, comb_num=1),
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        # Handle both sparse matrices and numpy arrays
        test_X = test_adata.X.toarray() if scipy.sparse.issparse(test_adata.X) else test_adata.X
        control_X = control_adata.X.toarray() if scipy.sparse.issparse(control_adata.X) else control_adata.X
        
        # Convert to tensors with float32 dtype
        test_X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
        control_X_tensor = torch.tensor(control_X, dtype=torch.float32).to(device)
        
        z_sem = sampler.model.encoder(
            x_start=test_X_tensor,
            drug_dose=encode_drug_doses,
            control_feature=control_X_tensor
        )

    sample_interp = sampler.pred(
        z_sem=z_sem,
        gene_size=test_adata.shape[1],
        batch_size=batch_size,
        drug_dose=encode_drug_doses,
        control_feature=control_X_tensor,
        delta_s_override=delta_s_override
    )

    torch.cuda.empty_cache()

    return sample_interp, test_adata, control_adata


def main():
    parser = argparse.ArgumentParser(
        description='Test scSketch model on sciplex dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint file')
    parser.add_argument('--train_adata_path', type=str, required=True,
                        help='Path to training data (h5ad file)')
    parser.add_argument('--control_adata_path', type=str, required=True,
                        help='Path to control data (h5ad file)')
    parser.add_argument('--test_adata_path', type=str, required=True,
                        help='Path to test data (h5ad file)')
    
    # Model architecture arguments
    parser.add_argument('--gene_size', type=int, default=2000,
                        help='Gene size')
    parser.add_argument('--dit_hidden_size', type=int, default=768,
                        help='DiT hidden size')
    parser.add_argument('--dit_num_heads', type=int, default=12,
                        help='DiT number of attention heads')
    parser.add_argument('--dit_mlp_ratio', type=float, default=4.0,
                        help='DiT MLP ratio')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers')
    parser.add_argument('--dit1d_use_value_embedding', action='store_true', default=True,
                        help='Use value embedding for DiT1D')
    parser.add_argument('--dit1d_patch_size', type=int, default=1,
                        help='DiT1D patch size (1=per-gene, >1=patch mode)')
    
    # Training/inference arguments
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for prediction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    # Output arguments
    parser.add_argument('--output_prefix', type=str, default='sciplex_scSketch',
                        help='Prefix for output CSV files')
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save results to CSV files')
    
    # Sampling arguments
    parser.add_argument('--timestep_respacing', type=str, default='',
                        help='Timestep respacing for DDIM (e.g., "50" for 50 steps instead of 1000)')
    
    
    args = parser.parse_args()
    use_dit1d = True
    
    # Determine model name for output
    if use_dit1d:
        model_name = 'DiT1D_1D'
    else:
        model_name = 'scSketch'
    
    print("=" * 50)
    print(f"Testing {model_name} Model")
    print("=" * 50)
    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.test_adata_path}")
    print(f"Gene size: {args.gene_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Sampling: DDIM (timestep_respacing={args.timestep_respacing if args.timestep_respacing else '1000'})")
    print("=" * 50)
    
    try:
        sample_interp, test_adata, control_adata = run_sampler(
            model_path=args.model_path,
            train_adata_path=args.train_adata_path,
            control_adata_path=args.control_adata_path,
            test_adata_path=args.test_adata_path,
            gene_size=args.gene_size,
            dit_hidden_size=args.dit_hidden_size,
            dit_num_heads=args.dit_num_heads,
            dit_mlp_ratio=args.dit_mlp_ratio,
            dit1d_use_value_embedding=args.dit1d_use_value_embedding,
            dit1d_patch_size=args.dit1d_patch_size,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            device=args.device,
            timestep_respacing=args.timestep_respacing
        )
        
        # Convert to numpy arrays
        predicted = sample_interp.detach().cpu().numpy()
        # Handle both sparse matrices and numpy arrays
        true = test_adata.X.toarray() if scipy.sparse.issparse(test_adata.X) else test_adata.X
        control_true = control_adata.X.toarray() if scipy.sparse.issparse(control_adata.X) else control_adata.X  # Paper: use true control (μ_pre) for all calculations
        
        # Aggregate by condition (cell line × drug × dose)
        # Paper: "For each condition key c, let μ(c)_post,g and μ(c)_pre,g denote the aggregated means"
        print("\nAggregating data by condition...")
        
        # Build condition key from available columns
        condition_cols = []
        if 'cell_line' in test_adata.obs.columns:
            condition_cols.append('cell_line')
        if 'SMILES' in test_adata.obs.columns:
            condition_cols.append('SMILES')
        if 'dose' in test_adata.obs.columns:
            condition_cols.append('dose')
        
        if len(condition_cols) == 0:
            print("Warning: No condition columns found. Using all samples as single condition.")
            # If no condition info, treat all as one condition (already aggregated)
            predicted_agg = predicted.mean(axis=0, keepdims=True) if predicted.shape[0] > 1 else predicted
            true_agg = true.mean(axis=0, keepdims=True) if true.shape[0] > 1 else true
        else:
            # Create condition identifier
            test_adata.obs['condition'] = test_adata.obs[condition_cols].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )
            
            # Aggregate by condition: take mean within each condition
            conditions = test_adata.obs['condition'].unique()
            predicted_agg = []
            true_agg = []
            
            print(f"  Found {len(conditions)} unique conditions")
            
            for cond in conditions:
                mask = test_adata.obs['condition'] == cond
                predicted_agg.append(predicted[mask].mean(axis=0))
                true_agg.append(true[mask].mean(axis=0))
            
            predicted_agg = np.array(predicted_agg)  # (n_conditions, n_genes)
            true_agg = np.array(true_agg)
        
        # Control: if multiple samples, use mean (all conditions share same control)
        if control_true.shape[0] > 1:
            control_agg = control_true.mean(axis=0, keepdims=True)
        else:
            control_agg = control_true
        
        print(f"  Aggregated shape: predicted {predicted_agg.shape}, true {true_agg.shape}, control {control_agg.shape}")
        
        # Save aggregated data for later use
        if args.save_results:
            import os
            results_dir = 'results'
            os.makedirs(results_dir, exist_ok=True)
            
            # Save condition information
            if len(condition_cols) > 0:
                condition_info = []
                for cond in conditions:
                    cond_dict = {'condition': cond}
                    # Split condition back to components
                    cond_parts = cond.split('_')
                    for i, col in enumerate(condition_cols):
                        if i < len(cond_parts):
                            cond_dict[col] = cond_parts[i]
                    condition_info.append(cond_dict)
                condition_df = pd.DataFrame(condition_info)
            else:
                condition_df = pd.DataFrame({'condition': ['all_samples']})
            
            # Save to separate CSV files for easier loading
            # Format: each file has conditions as rows, genes as columns
            data_prefix = os.path.join(results_dir, f'{args.output_prefix}')
            
            # Save predicted data
            predicted_df = pd.DataFrame(
                predicted_agg,
                index=condition_df['condition'],
                columns=[f'gene_{i}' for i in range(predicted_agg.shape[1])]
            )
            predicted_file = f'{data_prefix}_predicted.csv'
            predicted_df.to_csv(predicted_file)
            
            # Save true data
            true_df = pd.DataFrame(
                true_agg,
                index=condition_df['condition'],
                columns=[f'gene_{i}' for i in range(true_agg.shape[1])]
            )
            true_file = f'{data_prefix}_true.csv'
            true_df.to_csv(true_file)
            
            # Save control data
            control_df = pd.DataFrame(
                control_agg,
                index=['control'],
                columns=[f'gene_{i}' for i in range(control_agg.shape[1])]
            )
            control_file = f'{data_prefix}_control.csv'
            control_df.to_csv(control_file)
            
            # Save condition info
            condition_file = f'{data_prefix}_conditions.csv'
            condition_df.to_csv(condition_file, index=False)
            
            print(f"  Data saved to:")
            print(f"    Predicted: {predicted_file}")
            print(f"    True: {true_file}")
            print(f"    Control: {control_file}")
            print(f"    Conditions: {condition_file}")
        
        # Calculate per-condition R2 metrics only
        print("\nCalculating per-condition R2 metrics...")
        
        n_cond = predicted_agg.shape[0]
        
        if len(condition_cols) > 0:
            conditions_list = conditions
        else:
            conditions_list = ['all_samples']
        
        # Store R2 for each condition
        r2_list = []
        
        for i in range(n_cond):
            cond_name = conditions_list[i]
            
            cond_r2 = cal_metric(
                predicted_agg[i].flatten(),
                true_agg[i].flatten()
            )
            r2_list.append(cond_r2)
        
        # Calculate mean and median across conditions
        r2_mean = np.mean(r2_list) if len(r2_list) > 0 else np.nan
        r2_median = np.median(r2_list) if len(r2_list) > 0 else np.nan
        
        print(f"\nPer-condition R2 summary:")
        print(f"  R2: mean={r2_mean:.6f}, median={r2_median:.6f}")
        
        # Store aggregated metrics for saving
        aggregated_metrics = {
            'R2_mean': r2_mean,
            'R2_median': r2_median
        }
        
        # Calculate basic metric (mean across samples, for reference) - only print, not save
        r2 = cal_metric(
            predicted.mean(axis=0).flatten(),
            true.mean(axis=0).flatten()
        )
        
        print(f"\n{model_name} Basic Results (all samples, not saved):")
        print(f"  R2 Score: {r2:.6f}")
        
        if args.save_results:
            # Save aggregated metrics (mean and median across conditions)
            metrics_df = pd.DataFrame([aggregated_metrics])
            metrics_df.insert(0, 'Model', model_name)
            
            # Save to results directory
            import os
            results_dir = 'results'
            os.makedirs(results_dir, exist_ok=True)
            
            metrics_file = os.path.join(results_dir, f'{args.output_prefix}_metrics.csv')
            metrics_df.to_csv(metrics_file, index=False)
            
            # Save R2 summary to separate report
            comprehensive_metrics_file = os.path.join(results_dir, f'{args.output_prefix}_r2_metrics.txt')
            with open(comprehensive_metrics_file, 'w') as f:
                f.write("="*60 + "\n")
                f.write(f"{model_name} - R2 Evaluation Report\n")
                f.write("="*60 + "\n\n")

                f.write("R2 METRICS\n")
                f.write("-"*60 + "\n")
                f.write(f"R2 (mean):                 {aggregated_metrics.get('R2_mean', np.nan):.6f}\n")
                f.write(f"R2 (median):               {aggregated_metrics.get('R2_median', np.nan):.6f}\n\n")

                f.write("="*60 + "\n")
                f.write("Metric Interpretation:\n")
                f.write("="*60 + "\n")
                f.write("R2:             Coefficient of determination (higher is better, close to 1.0 is best)\n")
            
            print(f"\n✓ Results saved to:")
            print(f"  Aggregated metrics (CSV):      {metrics_file}")
            print(f"  Comprehensive report (TXT):    {comprehensive_metrics_file}")
        
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()




