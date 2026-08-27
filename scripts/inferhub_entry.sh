#!/usr/bin/env bash
set -euo pipefail

: "${INFER_CODE_DIR:?InferHub must provide INFER_CODE_DIR}"
: "${INFER_WEIGHTS_DIR:?InferHub must provide INFER_WEIGHTS_DIR}"
: "${INFER_OUTPUT_DIR:?InferHub must provide INFER_OUTPUT_DIR}"

cd "${INFER_CODE_DIR}"

export WAN_MODEL_PATH="${WAN_MODEL_PATH:-${INFER_WEIGHTS_DIR}/model}"
export WAN_SPARSE_PYTHON_OVERLAYS="${WAN_SPARSE_PYTHON_OVERLAYS:-${INFER_WEIGHTS_DIR}/python-overlay}"
export XDG_CACHE_HOME="${INFER_OUTPUT_DIR}/cache/xdg"
export HF_HOME="${INFER_OUTPUT_DIR}/cache/huggingface"
export TORCH_HOME="${INFER_OUTPUT_DIR}/cache/torch"
export TRITON_CACHE_DIR="${INFER_OUTPUT_DIR}/cache/triton"

suite="${WAN_SPARSE_SUITE:-configs/formal_stage2_v2.json}"
num_shards="${WAN_SPARSE_NUM_SHARDS:-1}"
shard_index="${WAN_SPARSE_SHARD_INDEX:-0}"

python_bin="${WAN_SPARSE_PYTHON:-python3}"
"${python_bin}" scripts/run_matrix.py \
  --suite "${suite}" \
  --num-shards "${num_shards}" \
  --shard-index "${shard_index}" \
  --output-root "${INFER_OUTPUT_DIR}/videos" \
  --manifest-root "${INFER_OUTPUT_DIR}/manifests"
