#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
visual_libs=$VIRTUAL_ENV/lib
for visual_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$visual_lib" ]] || visual_libs="$visual_libs:$visual_lib"
done
export LD_LIBRARY_PATH="$visual_libs:${LD_LIBRARY_PATH:-}"
visual_assets="$INFER_WEIGHTS_DIR/longlive2_native_inputs_v1"
export PYTHONPATH="$visual_assets/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0 LLV2_COMPILE_VAE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  python scripts/prepare_longlive2_reference.py --inputs "$visual_assets" --source "$INFER_CODE_DIR/third_party/LongLive2"
  python scripts/prepare_native_visual_inputs.py --root "$INFER_WEIGHTS_DIR" --cases "${VISUAL_SOURCE_CASES:?}" --output "$INFER_OUTPUT_DIR/inputs"
  python scripts/run_native_visual_restart_batch.py --assets "$visual_assets" --specs "$INFER_OUTPUT_DIR/inputs" --output "$INFER_OUTPUT_DIR/screen" > "$INFER_OUTPUT_DIR/frozen_plan.json"
  touch "$INFER_OUTPUT_DIR/prepared.ok"
  exit 0
fi
python scripts/run_native_visual_restart_batch.py --assets "$visual_assets" --specs "$INFER_OUTPUT_DIR/inputs" --output "$INFER_OUTPUT_DIR/screen" --run
