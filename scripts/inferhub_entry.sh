#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_WEIGHTS_DIR:?missing INFER_WEIGHTS_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

library_paths=("${VIRTUAL_ENV}/lib")
for library_dir in \
  "${VIRTUAL_ENV}"/lib/python*/site-packages/nvidia/*/lib \
  "${VIRTUAL_ENV}"/lib/python*/site-packages/torch/lib \
  "${VIRTUAL_ENV}"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ -d "${library_dir}" ]] && library_paths+=("${library_dir}")
done
joined_library_paths=$(IFS=:; echo "${library_paths[*]}")
export LD_LIBRARY_PATH="${joined_library_paths}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
bundle_root=${LONGLIVE_INPUT_BUNDLE_ROOT:-${INFER_WEIGHTS_DIR}}
export PYTHONPATH="${bundle_root}/python-overlay:${INFER_CODE_DIR}:${INFER_CODE_DIR}/third_party/longlive-inferhub:${INFER_CODE_DIR}/third_party/LongLive-RAG${PYTHONPATH:+:${PYTHONPATH}}"
export LONGLIVE_BASE_SOURCE="${INFER_CODE_DIR}/third_party/longlive-inferhub"
export LONGLIVE_RAG_SOURCE="${INFER_CODE_DIR}/third_party/LongLive-RAG"
export LONGLIVE_WAN_MODELS_ROOT="${bundle_root}/model"
export LONGLIVE_GENERATOR_CKPT="${bundle_root}/checkpoints/longlive_init.pt"
export LONGLIVE_LORA_CKPT="${bundle_root}/checkpoints/longlive_lora_003000.pt"
export LONGLIVE_DISABLE_FA3=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

config_path="${LONGLIVE_CONFIG_PATH:-configs/inferhub/rag_dense_21.yaml}"
nvidia-smi
python scripts/run_longlive_sparse.py --config_path "${config_path}"
python third_party/longlive-inferhub/scripts/inspect_video.py "$(find "${INFER_OUTPUT_DIR}" -maxdepth 1 -name '*.mp4' -type f | sort | head -n1)" "${INFER_OUTPUT_DIR}/quality"
sha256sum "${INFER_OUTPUT_DIR}"/*.mp4 > "${INFER_OUTPUT_DIR}/SHA256SUMS.txt"
