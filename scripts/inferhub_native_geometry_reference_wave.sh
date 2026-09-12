#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}" "${NATIVE_GEOMETRY_INPUTS:?}" "${NATIVE_SOURCE_GEOMETRY:?}"
reference_libs=$VIRTUAL_ENV/lib
for reference_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$reference_lib" ]] || reference_libs="$reference_libs:$reference_lib"
done
export LD_LIBRARY_PATH="$reference_libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_WEIGHTS_DIR/$NATIVE_GEOMETRY_INPUTS/sam2-runtime:$INFER_WEIGHTS_DIR/$NATIVE_GEOMETRY_INPUTS/sam2-python:$INFER_WEIGHTS_DIR/longlive2_native_inputs_v1/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0 LLV2_COMPILE_VAE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$INFER_CODE_DIR"
reference_args=(--geometry-wave --geometry-reference-wave --latent-frames 128 --seed 20260913
  --scenario generated_patchwork_toy_cut_revisit --gpu-pairs 2 --required-gpu-name H200 --allow-h800
  --assets "$INFER_WEIGHTS_DIR/longlive2_native_inputs_v1" --source "$INFER_CODE_DIR/third_party/LongLive2"
  --geometry-inputs "$INFER_WEIGHTS_DIR/$NATIVE_GEOMETRY_INPUTS" --geometry-source-masks "$INFER_WEIGHTS_DIR/$NATIVE_SOURCE_GEOMETRY"
  --output "$INFER_OUTPUT_DIR/screen")
if [[ ${1:-} == --prepare-only ]]; then
  python scripts/prepare_longlive2_reference.py --inputs "$INFER_WEIGHTS_DIR/longlive2_native_inputs_v1" --source "$INFER_CODE_DIR/third_party/LongLive2"
  python scripts/verify_geometry_wave_inputs.py --inputs "$INFER_WEIGHTS_DIR/$NATIVE_GEOMETRY_INPUTS"
  python scripts/run_native_duration_wave.py "${reference_args[@]}" > "$INFER_OUTPUT_DIR/frozen_plan.json"
  touch "$INFER_OUTPUT_DIR/prepared.ok"
  exit 0
fi
python scripts/run_native_duration_wave.py "${reference_args[@]}" --run
