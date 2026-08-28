#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"
: "${LONGLIVE_PARETO_SELECTION:?missing LONGLIVE_PARETO_SELECTION}"
: "${LONGLIVE_ROUTE_CAPTURE_DIR:?missing LONGLIVE_ROUTE_CAPTURE_DIR}"

source "${INFER_CODE_DIR}/scripts/inferhub_runtime_env.sh"
IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
lane_count=${#assigned_gpus[@]}
if [[ ${lane_count} -ne 4 && ${lane_count} -ne 8 ]]; then
  echo "Pareto route benchmarks require exactly four or eight assigned GPUs" >&2
  exit 2
fi
task_count=$(python - "${LONGLIVE_PARETO_SELECTION}" <<'PY'
import json
import sys
print(3 * len(json.load(open(sys.argv[1], encoding="utf-8"))["selected_methods"]))
PY
)
if (( task_count < lane_count )); then
  echo "${task_count} distinct route tasks cannot fill ${lane_count} GPU lanes" >&2
  exit 2
fi

pids=()
for ((lane=0; lane<lane_count; lane++)); do
  lane_root=${INFER_OUTPUT_DIR}/lane${lane}
  mkdir -p "${lane_root}"
  CUDA_VISIBLE_DEVICES=${assigned_gpus[${lane}]} \
    python scripts/run_route_benchmark_matrix.py \
      --selection "${LONGLIVE_PARETO_SELECTION}" \
      --capture-dir "${LONGLIVE_ROUTE_CAPTURE_DIR}" \
      --method-params "${INFER_CODE_DIR}/configs/formal/method_params.json" \
      --output-root "${lane_root}" \
      --cache-root "/tmp/wan_longlive_pareto_route_bench_lane${lane}" \
      --shard-index "${lane}" --shard-count "${lane_count}" \
      >"${lane_root}/runner.log" 2>&1 &
  pids+=("$!")
done
statuses=()
set +e
for pid in "${pids[@]}"; do
  wait "${pid}"
  statuses+=("$?")
done
set -e

state_args=()
for ((lane=0; lane<lane_count; lane++)); do
  state_args+=(--states "${INFER_OUTPUT_DIR}/lane${lane}/shard_${lane}_states.json")
done
python scripts/audit_route_benchmark_states.py \
  --selection "${LONGLIVE_PARETO_SELECTION}" \
  "${state_args[@]}" \
  --output "${INFER_OUTPUT_DIR}/route_benchmark_audit.json"
python - "${INFER_OUTPUT_DIR}/batch_status.json" "${statuses[@]}" <<'PY'
import json
import sys
from pathlib import Path
payload = {"lane_statuses": [int(value) for value in sys.argv[2:]], "terminal_audit_completed": True}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
