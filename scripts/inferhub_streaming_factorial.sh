#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
factor_root=$INFER_OUTPUT_DIR
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  factor_libraries=$VIRTUAL_ENV/lib
  for factor_library in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
    [[ ! -d "$factor_library" ]] || factor_libraries="$factor_libraries:$factor_library"
  done
  export LD_LIBRARY_PATH="$factor_libraries:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
  python -c 'import torch,av,flash_attn; from scripts.profile_streaming_pipeline import build_repetition_schedule; assert len(build_repetition_schedule([(m,"current_stream",r) for m in ("batch","async_priority") for r in ("upstream","direct_output")],3,20260908))==12; print("STREAM_FACTORIAL_PREP_PASS",torch.__version__)'
  touch "$factor_root/prepared.ok"
  exit 0
fi
source scripts/inferhub_runtime_env.sh
IFS=',' read -r -a factor_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#factor_gpus[@]} == 4 ]] || { echo 'requires exactly four assigned GPUs' >&2; exit 2; }
factor_methods=(rag_dense transfer_vaware_hybrid_history rag_dense transfer_vaware_hybrid_history)
factor_prompts=(calibration_motion calibration_motion calibration_state calibration_state)
factor_pids=()
for factor_lane in "${!factor_gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=${factor_gpus[$factor_lane]} python scripts/profile_streaming_pipeline.py \
    --output "$factor_root/lane$factor_lane" --method "${factor_methods[$factor_lane]}" \
    --prompt "${factor_prompts[$factor_lane]}" --latent-frames 120 --warmup-latents 21 \
    --variants batch_current_stream,async_priority_current_stream --rope-factorial --repeats 3 \
    --order-seed "$((20260908+factor_lane))" >"$factor_root/lane$factor_lane.log" 2>&1 &
  factor_pids+=("$!")
done
factor_statuses=()
for factor_pid in "${factor_pids[@]}"; do
  factor_status=0
  wait "$factor_pid" || factor_status=$?
  factor_statuses+=("$factor_status")
done
python scripts/collect_streaming_factorial.py --root "$factor_root" --expected-lanes 4 --repeats 3 \
  --exit-codes "${factor_statuses[@]}"
