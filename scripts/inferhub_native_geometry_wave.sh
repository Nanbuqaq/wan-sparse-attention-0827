#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
geometry_libs=$VIRTUAL_ENV/lib
for geometry_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$geometry_lib" ]] || geometry_libs="$geometry_libs:$geometry_lib"
done
export LD_LIBRARY_PATH="$geometry_libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_OUTPUT_DIR/geometry_inputs/sam2-runtime:$INFER_OUTPUT_DIR/geometry_inputs/sam2-python:$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0 LLV2_COMPILE_VAE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  python scripts/prepare_longlive2_reference.py --inputs "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2"
  python scripts/prepare_geometry_wave_inputs.py --output "$INFER_OUTPUT_DIR/geometry_inputs"
  python scripts/verify_geometry_wave_inputs.py --inputs "$INFER_OUTPUT_DIR/geometry_inputs"
  touch "$INFER_OUTPUT_DIR/prepared.ok"
  exit 0
fi
geometry_hardware_args=()
[[ ${NATIVE_ALLOW_H800:-0} != 1 ]] || geometry_hardware_args+=(--allow-h800)
geometry_pairs=4
if [[ ${GEOMETRY_RECOVERY_ONLY:-0} == 1 ]]; then
  geometry_pairs=2
  geometry_hardware_args+=(--geometry-recovery-only)
fi
python scripts/run_native_duration_wave.py --geometry-wave --latent-frames 128 --seed 20260913 \
  --scenario generated_patchwork_toy_cut_revisit --gpu-pairs "$geometry_pairs" --run \
  --required-gpu-name H200 "${geometry_hardware_args[@]}" \
  --geometry-inputs "$INFER_OUTPUT_DIR/geometry_inputs" \
  --assets "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2" --output "$INFER_OUTPUT_DIR/screen"
