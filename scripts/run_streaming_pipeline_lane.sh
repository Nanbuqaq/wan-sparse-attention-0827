#!/usr/bin/env bash
set -Eeuo pipefail
stream_gpu=${1:?physical GPU}
stream_output=${2:?new output root}
stream_method=${3:?method}
stream_prompt=${4:?prompt}
shift 4
stream_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stream_workspace=/home/zhouhe08/MyProjects/0904-longlive-system
stream_bundle=/kaimm-distill/zhouhe08/longlive/input_bundle
[[ ! -e "$stream_output" && ! -e "$stream_output.log" ]]
mkdir -p "$(dirname "$stream_output")"
export LONGLIVE_BASE_SOURCE=$stream_workspace/publish_repo/third_party/longlive-inferhub
export LONGLIVE_RAG_SOURCE=$stream_workspace/publish_repo/third_party/LongLive-RAG
export LONGLIVE_PYTHON_OVERLAY=$stream_bundle/python-overlay
export LONGLIVE_WAN_MODELS_ROOT=$stream_bundle/model
export LONGLIVE_GENERATOR_CKPT=$stream_bundle/checkpoints/longlive_init.pt
export LONGLIVE_LORA_CKPT=$stream_bundle/checkpoints/longlive_lora_003000.pt
export PYTHONPATH="$LONGLIVE_PYTHON_OVERLAY:$stream_repo:$LONGLIVE_BASE_SOURCE:$LONGLIVE_RAG_SOURCE"
export LONGLIVE_DISABLE_FA3=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$stream_repo"
exec /usr/bin/python3 scripts/run_on_free_gpu.py --physical-gpu "$stream_gpu" -- \
  /usr/bin/python3 scripts/profile_streaming_pipeline.py --output "$stream_output" \
  --method "$stream_method" --prompt "$stream_prompt" "$@" >"$stream_output.log" 2>&1
