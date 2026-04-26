# scSketch

**scSketch** is a structure-aware, perturbation-conditioned diffusion model for predicting single-cell gene expression responses to unseen drug perturbations. It integrates gene-level modeling with pathway-level biological structure, enabling more consistent and biologically meaningful generation.

This repository provides training, sampling, and evaluation utilities for scRNA-seq perturbation datasets (e.g., sci-Plex, LINCS), built upon a DiT1D backbone, PROGENy-based pathway sketching, and a low-rank drug perturbation operator.

---

## Highlights

- **Structure-aware diffusion modeling** for predicting perturbation responses from control cells with drug and dosage conditions.
- **DiT1D backbone** for scalable modeling of high-dimensional gene expression.
- **PROGENy-based pathway sketch** for incorporating biological priors and enforcing pathway-level consistency.
- **Structured perturbation dynamics (SPD)** via low-rank transformations for modeling cell-specific drug effects.
- **Efficient sampling** with DPM-Solver++ / DDIM for fast inference.

## Requirements
This project relies on standard scientific Python packages and PyTorch. At minimum you will need:
- Python 3.9+
- PyTorch
- scanpy, anndata
- numpy, scipy, pandas, scikit-learn
- rdkit (for drug structure options)

## Data format
Training and testing use AnnData (.h5ad) files with at least:
- `X`: expression matrix (cells x genes)
- `obs['SMILES']`: drug string
- `obs['dose']`: dose value
- (optional) `obs['cell_line']` for condition aggregation

Control datasets should have matching gene order and compatible cell metadata.

## Dataset
- Perturbation dataset: [dataset-link](https://drive.google.com/drive/folders/1YUFf0OY_1NuOC4J18_79YtGw5RHjwsXu?usp=drive_link)

## Quick start
### 1) Train
Run training with required paths and a log directory:

```bash
python train_scSketch.py \
  --data_path /path/to/train.h5ad \
  --control_data_path /path/to/train_control.h5ad \
  --logger_path /path/to/logs \
  --gene_size 2000 \
  --output_dim 2000 \
  --batch_size 128
```

Notes:
- `--logger_path` and `--data_path` are required.
- DiT1D settings (`--dit_hidden_size`, `--dit_num_heads`, `--num_layers`) can be tuned.

### 2) Test / evaluate
```bash
python test_scSketch.py \
  --model_path /path/to/model.pt \
  --train_adata_path /path/to/train.h5ad \
  --control_adata_path /path/to/test_control.h5ad \
  --test_adata_path /path/to/test.h5ad \
  --gene_size 2000 \
  --batch_size 1024 \
  --device cuda \
  --timestep_respacing 50
```

Outputs are written to `results/` with the prefix set by `--output_prefix`.

### 3) One-click train + test (bash)
There is a helper script for sciplex-style datasets:

```bash
bash run_train_test.sh
```

Update dataset paths inside the script before running.

## Project layout
- `train_scSketch.py`: training entry point
- `test_scSketch.py`: inference + evaluation
- `sample_scSketch.py`: reusable sampler and diffusion helpers
- `scSketch/`: model, diffusion, and utilities
