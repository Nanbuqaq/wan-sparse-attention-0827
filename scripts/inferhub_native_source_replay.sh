#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}" "${NATIVE_REPLAY_CASE:?}"
replay_libs=$VIRTUAL_ENV/lib
for replay_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$replay_lib" ]] || replay_libs="$replay_libs:$replay_lib"
done
export LD_LIBRARY_PATH="$replay_libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_WEIGHTS_DIR/longlive2_native_inputs_v1/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0 LLV2_COMPILE_VAE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  python scripts/prepare_longlive2_reference.py --inputs "$INFER_WEIGHTS_DIR/longlive2_native_inputs_v1" --source "$INFER_CODE_DIR/third_party/LongLive2"
  touch "$INFER_OUTPUT_DIR/prepared.ok"
  exit 0
fi
python scripts/check_native_hardware.py --expected-count 1 --required H200 --allow-h800 --output "$INFER_OUTPUT_DIR/hardware.json"
python scripts/replay_native_raw_source.py --case "$INFER_WEIGHTS_DIR/$NATIVE_REPLAY_CASE" \
  --assets "$INFER_WEIGHTS_DIR/longlive2_native_inputs_v1" --source "$INFER_CODE_DIR/third_party/LongLive2" --output "$INFER_OUTPUT_DIR/replay"
