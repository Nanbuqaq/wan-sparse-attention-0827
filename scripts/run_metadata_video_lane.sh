#!/usr/bin/env bash
set -Eeuo pipefail
metadata_gpu=${1:?physical GPU}
metadata_suite=${2:?frozen suite}
metadata_output=${3:?new output root}
metadata_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
metadata_workspace=/home/zhouhe08/MyProjects/0904-longlive-system
metadata_bundle=/kaimm-distill/zhouhe08/longlive/input_bundle
[[ "$metadata_gpu" == 0 || "$metadata_gpu" == 1 ]]
[[ -f "$metadata_suite" && ! -e "$metadata_output" ]]
mkdir -p "$metadata_output"
export LONGLIVE_BASE_SOURCE=$metadata_workspace/publish_repo/third_party/longlive-inferhub
export LONGLIVE_RAG_SOURCE=$metadata_workspace/publish_repo/third_party/LongLive-RAG
export LONGLIVE_PYTHON_OVERLAY=$metadata_bundle/python-overlay
export LONGLIVE_WAN_MODELS_ROOT=$metadata_bundle/model
export LONGLIVE_GENERATOR_CKPT=$metadata_bundle/checkpoints/longlive_init.pt
export LONGLIVE_LORA_CKPT=$metadata_bundle/checkpoints/longlive_lora_003000.pt
export PYTHONPATH="$LONGLIVE_PYTHON_OVERLAY:$metadata_repo:$LONGLIVE_BASE_SOURCE:$LONGLIVE_RAG_SOURCE"
export LONGLIVE_DISABLE_FA3=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LONGLIVE_CAPTURE_QKV=0 LONGLIVE_CAPTURE_COMPLETE_ATTENTION=0 LONGLIVE_NVTX=0
export INFER_OUTPUT_DIR=$metadata_output
cd "$metadata_repo"
exec /usr/bin/python3 scripts/run_on_free_gpu.py --physical-gpu "$metadata_gpu" -- \
  /usr/bin/python3 scripts/run_loaded_method_suite.py --suite "$metadata_suite" --shard-index 0 --shard-count 1 \
  >"$metadata_output/runner.log" 2>&1
