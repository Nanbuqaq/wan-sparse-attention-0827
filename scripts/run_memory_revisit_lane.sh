#!/usr/bin/env bash
set -Eeuo pipefail
revisit_gpu=${1:?physical GPU}
revisit_output=${2:?new output directory}
shift 2
revisit_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
revisit_workspace=/home/zhouhe08/MyProjects/0904-longlive-system
revisit_bundle=/kaimm-distill/zhouhe08/longlive/input_bundle
[[ ! -e "$revisit_output" && ! -e "$revisit_output.log" ]]
mkdir -p "$(dirname "$revisit_output")"
export LONGLIVE_BASE_SOURCE=$revisit_workspace/publish_repo/third_party/longlive-inferhub
export LONGLIVE_RAG_SOURCE=$revisit_workspace/publish_repo/third_party/LongLive-RAG
export LONGLIVE_PYTHON_OVERLAY=$revisit_bundle/python-overlay
export LONGLIVE_WAN_MODELS_ROOT=$revisit_bundle/model
export LONGLIVE_GENERATOR_CKPT=$revisit_bundle/checkpoints/longlive_init.pt
export LONGLIVE_LORA_CKPT=$revisit_bundle/checkpoints/longlive_lora_003000.pt
export PYTHONPATH="$LONGLIVE_PYTHON_OVERLAY:$revisit_repo:$LONGLIVE_BASE_SOURCE:$LONGLIVE_RAG_SOURCE"
export LONGLIVE_DISABLE_FA3=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$revisit_repo"
exec /usr/bin/python3 scripts/run_on_free_gpu.py --physical-gpu "$revisit_gpu" -- \
  /usr/bin/python3 scripts/run_memory_revisit_videos.py --output "$revisit_output" "$@" \
  >"$revisit_output.log" 2>&1
