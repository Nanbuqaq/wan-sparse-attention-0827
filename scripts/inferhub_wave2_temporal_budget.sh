#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
w2_libs=$VIRTUAL_ENV/lib
for w2_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$w2_lib" ]] || w2_libs="$w2_libs:$w2_lib"
done
export LD_LIBRARY_PATH="$w2_libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0 LLV2_COMPILE_VAE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$INFER_CODE_DIR"
w2_args=(--wave2-config configs/system/wave2_scenarios.json --wave2-stage "${WAVE2_STAGE:-native}"
  --seed "${WAVE2_SEED:-20261010}" --gpu-pairs "${WAVE2_GPU_PAIRS:-4}"
  --assets "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2" --output "$INFER_OUTPUT_DIR/screen")
if [[ -n ${WAVE2_VALID_SCENARIOS:-} ]]; then
  IFS=',' read -r -a w2_valid <<<"$WAVE2_VALID_SCENARIOS"
  w2_args+=(--wave2-valid-scenarios "${w2_valid[@]}")
fi
if [[ -n ${WAVE2_REQUIRED_GPU:-} ]]; then
  w2_args+=(--required-gpu-name "$WAVE2_REQUIRED_GPU")
else
  w2_args+=(--required-gpu-name H200 --allow-h800)
fi
if [[ ${1:-} == --prepare-only ]]; then
  python scripts/prepare_longlive2_reference.py --inputs "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2"
  python scripts/run_native_duration_wave.py "${w2_args[@]}" > "$INFER_OUTPUT_DIR/frozen_plan.json"
  touch "$INFER_OUTPUT_DIR/prepared.ok"
  exit 0
fi
python scripts/run_native_duration_wave.py "${w2_args[@]}" --run
