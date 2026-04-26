#!/usr/bin/env bash
set -euo pipefail

# =========================
# scSketch train + test script
# =========================

# Optional: activate your environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
DATA_DIR="${PROJECT_DIR}/datasets"
DATASET_NAME="datasets_sciplex"  # options: datasets_sciplex / datasets_LINCS
DATASET_DIR="${DATA_DIR}/${DATASET_NAME}"
SPLIT_ID=0
CKPT_DIR="${PROJECT_DIR}/checkpoints/scSketch_split_${SPLIT_ID}"
RESULT_PREFIX="scSketch_split_${SPLIT_ID}"

# Train data
TRAIN_DATA_PATH="${DATASET_DIR}/sci_plex_train_drug_split_${SPLIT_ID}.h5ad"
TRAIN_CONTROL_DATA_PATH="${DATASET_DIR}/sci_plex_train_drug_split_${SPLIT_ID}_control.h5ad"

# Test data
TEST_DATA_PATH="${DATASET_DIR}/sci_plex_test_drug_split_${SPLIT_ID}.h5ad"
TEST_CONTROL_DATA_PATH="${DATASET_DIR}/sci_plex_test_drug_split_${SPLIT_ID}_control.h5ad"
# Processed metadata
PROCESSED_GENE_LIST="${DATA_DIR}/selected_genes_256_progeny.csv"

# ---------- Common model params ----------
GENE_SIZE=2000
BATCH_SIZE=128
LR=1e-4

# DiT1D is enabled by default in code
DIT_HIDDEN_SIZE=512
DIT_NUM_HEADS=8
NUM_LAYERS=12

# Sampling settings for test
TIMESTEP_RESPACING=50

mkdir -p "${CKPT_DIR}"
cd "${PROJECT_DIR}"

echo "Checking dataset files..."
REQUIRED_FILES=(
  "${TRAIN_DATA_PATH}"
  "${TRAIN_CONTROL_DATA_PATH}"
  "${TEST_DATA_PATH}"
  "${TEST_CONTROL_DATA_PATH}"
)

for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing required dataset file: ${f}"
    echo "Please place your processed .h5ad files under: ${DATA_DIR}"
    exit 1
  fi
done

if [[ -f "${PROCESSED_GENE_LIST}" ]]; then
  echo "Found processed gene list: ${PROCESSED_GENE_LIST}"
else
  echo "Warning: processed gene list not found: ${PROCESSED_GENE_LIST}"
fi

echo "========== [1/2] Training =========="

python train_scSketch.py \
  --data_path "${TRAIN_DATA_PATH}" \
  --control_data_path "${TRAIN_CONTROL_DATA_PATH}" \
  --resume_checkpoint "${CKPT_DIR}" \
  --logger_path "${CKPT_DIR}/logs" \
  --gene_size "${GENE_SIZE}" \
  --output_dim "${GENE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --dit_hidden_size "${DIT_HIDDEN_SIZE}" \
  --dit_num_heads "${DIT_NUM_HEADS}" \
  --num_layers "${NUM_LAYERS}"

MODEL_PATH="${CKPT_DIR}/model.pt"
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Training finished but model not found at: ${MODEL_PATH}"
  echo "Please check your save settings / logs in ${CKPT_DIR}"
  exit 1
fi

echo "========== [2/2] Testing =========="

python test_scSketch.py \
  --model_path "${MODEL_PATH}" \
  --train_adata_path "${TRAIN_DATA_PATH}" \
  --control_adata_path "${TEST_CONTROL_DATA_PATH}" \
  --test_adata_path "${TEST_DATA_PATH}" \
  --gene_size "${GENE_SIZE}" \
  --batch_size 1024 \
  --device cuda \
  --dit_hidden_size "${DIT_HIDDEN_SIZE}" \
  --dit_num_heads "${DIT_NUM_HEADS}" \
  --num_layers "${NUM_LAYERS}" \
  --output_prefix "${RESULT_PREFIX}" \
  --timestep_respacing "${TIMESTEP_RESPACING}"

echo "Done. Check outputs under:"
echo "  - ${PROJECT_DIR}/results"
echo "  - ${CKPT_DIR}"
