#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
duration_libs=$VIRTUAL_ENV/lib
for duration_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$duration_lib" ]] || duration_libs="$duration_libs:$duration_lib"
done
export LD_LIBRARY_PATH="$duration_libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0 LLV2_COMPILE_VAE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  python scripts/prepare_longlive2_reference.py --inputs "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2"
  python -c 'from adapters.longlive_sparse.native_duration_probe import duration_geometry; print(duration_geometry(728,128))'
  touch "$INFER_OUTPUT_DIR/prepared.ok"
  exit 0
fi
duration_hardware_args=()
[[ ${NATIVE_ALLOW_H800:-0} != 1 ]] || duration_hardware_args+=(--allow-h800)
python scripts/run_native_duration_wave.py --run --required-gpu-name H200 "${duration_hardware_args[@]}" \
  --latent-frames "${DURATION_LATENTS:-728}" --seed "${DURATION_SEED:-20261002}" \
  --scenario "${DURATION_SCENARIO:-generated_patchwork_toy_cut_revisit}" \
  --assets "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2" \
  --output "$INFER_OUTPUT_DIR/screen"
