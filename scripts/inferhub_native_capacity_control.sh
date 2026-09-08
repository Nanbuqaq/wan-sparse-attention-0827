#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
capacity_libs=$VIRTUAL_ENV/lib
for capacity_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$capacity_lib" ]] || capacity_libs="$capacity_libs:$capacity_lib"
done
export LD_LIBRARY_PATH="$capacity_libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_WEIGHTS_DIR/private-triton331:$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0
export TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  python scripts/prepare_longlive2_reference.py --inputs "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2"
  python -c 'import triton; assert triton.__version__=="3.3.1"; from adapters.longlive_sparse.native_capacity import install_positive_only_allocator; print("PRIVATE_RUNTIME_CAPACITY_PREP_PASS")'
  touch "$INFER_OUTPUT_DIR/prepared.ok"
  exit 0
fi
: "${CAPACITY_SEEDS:?}" "${CAPACITY_GATE_SEEDS:?}" "${CAPACITY_SCENARIOS:?}"
IFS=',' read -r -a capacity_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
IFS=',' read -r -a capacity_seeds <<<"$CAPACITY_SEEDS"
IFS=',' read -r -a capacity_gate_seeds <<<"$CAPACITY_GATE_SEEDS"
IFS=',' read -r -a capacity_scenarios <<<"$CAPACITY_SCENARIOS"
[[ ${#capacity_gpus[@]} == 4 && ${#capacity_seeds[@]} == 2 && ${#capacity_gate_seeds[@]} == 2 && ${#capacity_scenarios[@]} == 2 ]]
capacity_pids=()
for capacity_lane in "${!capacity_gpus[@]}"; do
  capacity_extra=();if (( capacity_lane % 2 )); then capacity_extra+=(--reverse-full-order);fi
  CUDA_VISIBLE_DEVICES=${capacity_gpus[$capacity_lane]} python scripts/run_native_capacity_control.py \
    --assets "$INFER_WEIGHTS_DIR" --output "$INFER_OUTPUT_DIR/lane$capacity_lane" \
    --cut-scenario "${capacity_scenarios[$((capacity_lane/2))]}" \
    --seed "${capacity_seeds[$((capacity_lane%2))]}" --gate-seed "${capacity_gate_seeds[$((capacity_lane%2))]}" \
    "${capacity_extra[@]}" >"$INFER_OUTPUT_DIR/lane$capacity_lane.log" 2>&1 &
  capacity_pids+=("$!")
done
for capacity_pid in "${capacity_pids[@]}"; do wait "$capacity_pid" || true;done
python scripts/collect_native_capacity_control.py --root "$INFER_OUTPUT_DIR"
