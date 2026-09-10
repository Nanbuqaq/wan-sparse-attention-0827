#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
block_libs=$VIRTUAL_ENV/lib
for block_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$block_lib" ]] || block_libs="$block_libs:$block_lib"
done
export LD_LIBRARY_PATH="$block_libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0 LLV2_COMPILE_VAE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  python scripts/prepare_longlive2_reference.py --inputs "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2"
  python -c 'from adapters.longlive_sparse.native_causal_block_memory import CausalBlockConfig; print(CausalBlockConfig())'
  touch "$INFER_OUTPUT_DIR/prepared.ok"
  exit 0
fi
IFS=',' read -r -a block_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#block_gpus[@]} == 4 || ${#block_gpus[@]} == 8 ]] || { echo 'requires four or eight assigned GPUs' >&2; exit 2; }
python scripts/run_native_causal_block_wave.py --stage screen --run --required-gpu-name H200 \
  --assets "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2" \
  --output "$INFER_OUTPUT_DIR/screen"
