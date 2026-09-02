#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"
: "${LONGLIVE_PARETO_SELECTION:?missing LONGLIVE_PARETO_SELECTION}"
: "${LONGLIVE_PARETO_PARTITION:?missing LONGLIVE_PARETO_PARTITION}"

source "${INFER_CODE_DIR}/scripts/inferhub_runtime_env.sh"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#assigned_gpus[@]} -eq 8 ]] || {
  echo "Pareto partition requires exactly eight assigned GPUs" >&2
  exit 2
}
[[ "${LONGLIVE_PARETO_PARTITION}" =~ ^[0-3]$ ]] || {
  echo "Pareto partition must be 0, 1, 2, or 3" >&2
  exit 2
}

batch_root=${INFER_OUTPUT_DIR}
control_root=${batch_root}/control
experiment_commit=$(git -C "${INFER_CODE_DIR}" rev-parse HEAD)
cpu_threads_per_lane=${LONGLIVE_CPU_THREADS_PER_LANE:-8}
[[ "${cpu_threads_per_lane}" =~ ^[1-9][0-9]*$ ]] || {
  echo "LONGLIVE_CPU_THREADS_PER_LANE must be a positive integer" >&2
  exit 2
}
export OMP_NUM_THREADS=${cpu_threads_per_lane}
export MKL_NUM_THREADS=${cpu_threads_per_lane}
export OPENBLAS_NUM_THREADS=${cpu_threads_per_lane}
export NUMEXPR_NUM_THREADS=${cpu_threads_per_lane}
mkdir -p "${control_root}"

python scripts/build_pareto_suites.py \
  --frozen-prompts "${INFER_CODE_DIR}/configs/formal/frozen_prompts.json" \
  --selection "${LONGLIVE_PARETO_SELECTION}" \
  --calibration "${INFER_CODE_DIR}/configs/formal/method_params.json" \
  --commit "${experiment_commit}" \
  --output-dir "${control_root}/full"
python scripts/build_pareto_partition_plan.py \
  --expected "${control_root}/full/expected_pareto_expansion.json" \
  --dense-suite "${control_root}/full/rag_dense_pareto_expansion.json" \
  --sparse-suite "${control_root}/full/rag_pareto_expansion.json" \
  --output-dir "${control_root}/partitions" \
  --partitions 4 --lanes 8 --max-lane-hours 8

partition_root=${control_root}/partitions/partition${LONGLIVE_PARETO_PARTITION}
partition_plan=${partition_root}/partition_plan.json
partition_expected=${partition_root}/expected.json
[[ -f "${partition_plan}" && -f "${partition_expected}" ]]

run_lane() {
  local lane=$1
  local device=${assigned_gpus[${lane}]}
  local lane_root=${batch_root}/lane${lane}
  local lane_plan=${partition_root}/lane${lane}_plan.json
  mkdir -p "${lane_root}"
  mapfile -t task_rows < <(python - "${lane_plan}" <<'PY'
import json, sys
for index, task in enumerate(json.load(open(sys.argv[1], encoding="utf-8"))["tasks"]):
    print(f"{index}|{task['kind']}|{task['suite']}|{task['id']}")
PY
  )
  [[ ${#task_rows[@]} -gt 0 ]] || { echo "empty lane ${lane}" >&2; return 2; }
  local statuses=()
  local status
  for row in "${task_rows[@]}"; do
    IFS='|' read -r task_index kind suite_name task_id <<<"${row}"
    local task_root=${lane_root}/task${task_index}
    local state
    mkdir -p "${task_root}"
    if [[ ${kind} == dense ]]; then
      state=${task_root}/dense_screen_states.json
    else
      state=${task_root}/shard_0_states.json
    fi
    if [[ -f "${state}" ]] && python - "${state}" "${task_id}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
matches = [
    case for case in payload.get("cases", [])
    if case.get("id", case.get("case_id")) == sys.argv[2]
]
raise SystemExit(0 if len(matches) == 1 and matches[0].get("status") in {"pass", "fail", "negative"} else 1)
PY
    then
      status=0
      echo "reusing existing terminal state for ${task_id}" >"${lane_root}/task${task_index}.log"
    else
      set +e
      if [[ ${kind} == dense ]]; then
        CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${task_root} \
          python scripts/run_loaded_dense_screen.py \
            --runtime rag_dense \
            --base-config configs/inferhub/rag_dense_21.yaml \
            --candidates "${partition_root}/${suite_name}" \
            --experiment-commit "${experiment_commit}" \
            --shard-index 0 --shard-count 1 \
            >"${lane_root}/task${task_index}.log" 2>&1
        status=$?
      else
        CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${task_root} \
          python scripts/run_loaded_method_suite.py \
            --suite "${partition_root}/${suite_name}" \
            --experiment-commit "${experiment_commit}" \
            --shard-index 0 --shard-count 1 \
            >"${lane_root}/task${task_index}.log" 2>&1
        status=$?
      fi
      set -e
    fi
    if [[ ! -f "${state}" ]]; then
      python - "${state}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"cases": []}) + "\n", encoding="utf-8")
PY
    fi
    statuses+=("${status}")
    echo "task=${task_id} kind=${kind} status=${status}" >>"${lane_root}/progress.log"
  done
  python - "${lane_root}/lane_status.json" "${statuses[@]}" <<'PY'
import json, sys
from pathlib import Path
values = [int(value) for value in sys.argv[2:]]
Path(sys.argv[1]).write_text(json.dumps({"task_statuses": values}, indent=2) + "\n", encoding="utf-8")
PY
}

pids=()
for lane in {0..7}; do
  run_lane "${lane}" >"${batch_root}/lane${lane}.log" 2>&1 &
  pids+=("$!")
done
lane_statuses=()
set +e
for pid in "${pids[@]}"; do
  wait "${pid}"
  lane_statuses+=("$?")
done
set -e

state_inputs=()
for lane in {0..7}; do
  lane_plan=${partition_root}/lane${lane}_plan.json
  task_count=$(python - "${lane_plan}" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["tasks"]))
PY
  )
  for ((task=0; task<task_count; task++)); do
    kind=$(python - "${lane_plan}" "${task}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["tasks"][int(sys.argv[2])]["kind"])
PY
    )
    if [[ ${kind} == dense ]]; then
      state=${batch_root}/lane${lane}/task${task}/dense_screen_states.json
    else
      state=${batch_root}/lane${lane}/task${task}/shard_0_states.json
    fi
    state_inputs+=(--input "${state}")
  done
done

python scripts/merge_case_states.py \
  "${state_inputs[@]}" \
  --expected "${partition_expected}" \
  --fill-missing-reason "Pareto partition runner did not emit a terminal state" \
  --output "${batch_root}/merged_case_states.json"
python scripts/audit_case_states.py \
  --expected "${partition_expected}" \
  --states "${batch_root}/merged_case_states.json" \
  --output "${batch_root}/terminal_state_audit.json"
python - "${batch_root}/batch_status.json" "${LONGLIVE_PARETO_PARTITION}" "${cpu_threads_per_lane}" "${lane_statuses[@]}" <<'PY'
import json, sys
from pathlib import Path
payload = {
    "partition": int(sys.argv[2]),
    "cpu_threads_per_lane": int(sys.argv[3]),
    "lane_statuses": [int(value) for value in sys.argv[4:]],
    "terminal_audit_completed": True,
    "staggered_exact_task_lanes": True,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
find "${batch_root}" -type f \( -name '*.mp4' -o -name 'latents.pt' \) -print0 \
  | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"
