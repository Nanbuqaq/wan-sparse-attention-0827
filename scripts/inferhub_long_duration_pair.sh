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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/kaimm-distill/zhouhe08/.triton-cache/longlive2}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/kaimm-distill/zhouhe08/.inductor-cache/longlive2}"
export WAN_SPARSE_PHYSICAL_GPUS="${WAN_SPARSE_PHYSICAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
cd "$INFER_CODE_DIR"
duration_latents=${WAVE2_DURATION_LATENTS:-3608}
common=(--assets "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2" --seed "${WAVE2_SEED:-20261010}"
  --duration-probe-latents "$duration_latents" --duration-noise-alignment absolute --cut-scenario w2_rotating_wooden_bird
  --native-local-frames 32 --cfg1-positive-cache-only --native-inplace-cache --native-shared-conditioning
  --fixed-adaln-warps 16 --fixed-adaln-stages 1 --constructor-mode strict_checkpoint_no_parameter_init
  --pipeline-mode overlap --pipeline-encode-mode thread --pipeline-slots 4 --pipeline-pixel-slots 4
  --pipeline-pinned-mib 256 --native-inplace-gelu)
if [[ ${WAVE2_ROUTE_TIMELINE:-0} == 1 ]]; then
  common+=(--wave2-route-timeline --wave2-route-timeline-payload-hash "${WAVE2_ROUTE_TIMELINE_PAYLOAD_HASH:-checkpoint}")
fi
python scripts/run_longlive2_native_reference.py "${common[@]}" --output "$INFER_OUTPUT_DIR/screen/native" --wave2-method w2_native
python scripts/run_longlive2_native_reference.py "${common[@]}" --output "$INFER_OUTPUT_DIR/screen/shared125_transition_anchor" \
  --wave2-method w2_steady_sparse --wave2-steady-fraction .125 --wave2-selector shared_sum_block64 \
  --wave2-preparation geometry_cache --wave2-route-refresh first_only --wave2-transition-anchor
