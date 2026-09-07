#!/usr/bin/env bash
set -Eeuo pipefail
tail_gpu=${1:?physical GPU}
tail_output=${2:?new output directory}
tail_kind=${3:?motion or state}
shift 3
tail_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tail_workspace=/home/zhouhe08/MyProjects/0904-longlive-system
tail_bundle=/kaimm-distill/zhouhe08/longlive/input_bundle
[[ ! -e "$tail_output" && ! -e "$tail_output.log" ]]
mkdir -p "$(dirname "$tail_output")"
export LONGLIVE_BASE_SOURCE=$tail_workspace/publish_repo/third_party/longlive-inferhub
export LONGLIVE_RAG_SOURCE=$tail_workspace/publish_repo/third_party/LongLive-RAG
export LONGLIVE_PYTHON_OVERLAY=$tail_bundle/python-overlay
export LONGLIVE_WAN_MODELS_ROOT=$tail_bundle/model
export LONGLIVE_GENERATOR_CKPT=$tail_bundle/checkpoints/longlive_init.pt
export LONGLIVE_LORA_CKPT=$tail_bundle/checkpoints/longlive_lora_003000.pt
export PYTHONPATH="$LONGLIVE_PYTHON_OVERLAY:$tail_repo:$LONGLIVE_BASE_SOURCE:$LONGLIVE_RAG_SOURCE"
export LONGLIVE_DISABLE_FA3=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$tail_repo"
exec /usr/bin/python3 scripts/run_on_free_gpu.py --physical-gpu "$tail_gpu" -- \
  /usr/bin/python3 scripts/run_prototype_reference_videos.py --output "$tail_output" --kind "$tail_kind" "$@" \
  >"$tail_output.log" 2>&1
