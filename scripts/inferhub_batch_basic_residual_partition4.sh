#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"
: "${LONGLIVE_RESIDUAL_CONTROL_DIR:?missing LONGLIVE_RESIDUAL_CONTROL_DIR}"
: "${LONGLIVE_RESIDUAL_PARTITION:?missing LONGLIVE_RESIDUAL_PARTITION}"

source "${INFER_CODE_DIR}/scripts/inferhub_runtime_env.sh"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ ${#assigned_gpus[@]} -ne 4 ]]; then
  echo "formal-basic residual partition requires exactly four assigned GPUs" >&2
  exit 2
fi
if [[ "${LONGLIVE_RESIDUAL_PARTITION}" != "0" && "${LONGLIVE_RESIDUAL_PARTITION}" != "1" ]]; then
  echo "residual partition must be 0 or 1" >&2
  exit 2
fi

batch_root=${INFER_OUTPUT_DIR}
control_root=${LONGLIVE_RESIDUAL_CONTROL_DIR}
global_plan=${control_root}/residual_plan.json
partition_plan=${control_root}/partition${LONGLIVE_RESIDUAL_PARTITION}_plan.json
partition_expected=${control_root}/partition${LONGLIVE_RESIDUAL_PARTITION}_expected.json
for required in "${global_plan}" "${partition_plan}" "${partition_expected}"; do
  [[ -f "${required}" ]] || { echo "missing residual partition control: ${required}" >&2; exit 3; }
done

experiment_commit=$(python - "${global_plan}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["experiment_commit"])
PY
)
actual_commit=$(git -C "${INFER_CODE_DIR}" rev-parse HEAD)
[[ "${actual_commit}" == "${experiment_commit}" ]] || {
  echo "residual runner checkout ${actual_commit} != experiment ${experiment_commit}" >&2
  exit 4
}
mapfile -t global_lanes < <(python - "${partition_plan}" <<'PY'
import json, sys
for lane in json.load(open(sys.argv[1], encoding="utf-8"))["lane_ids"]:
    print(lane)
PY
)
[[ ${#global_lanes[@]} -eq 4 ]] || { echo "partition must contain four global lanes" >&2; exit 3; }

run_lane() {
  local slot=$1
  local lane=${global_lanes[${slot}]}
  local device=${assigned_gpus[${slot}]}
  local lane_root=${batch_root}/lane${lane}
  local lane_plan=${control_root}/lane${lane}_plan.json
  mkdir -p "${lane_root}"
  mapfile -t suites < <(python - "${lane_plan}" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["suites"]:
    print(item["suite"])
PY
  )
  [[ ${#suites[@]} -gt 0 ]] || { echo "empty residual lane ${lane}" >&2; return 2; }
  local statuses=()
  local suite_index=0
  set +e
  for suite_name in "${suites[@]}"; do
    local run_root=${lane_root}/suite${suite_index}
    mkdir -p "${run_root}"
    CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${run_root} \
      python scripts/run_loaded_method_suite.py \
        --suite "${control_root}/${suite_name}" \
        --shard-index 0 --shard-count 1 \
        --experiment-commit "${experiment_commit}" \
        >"${lane_root}/suite${suite_index}.log" 2>&1
    statuses+=("$?")
    suite_index=$((suite_index + 1))
  done
  set -e
  python - "${lane_root}/lane_status.json" "${statuses[@]}" <<'PY'
import json, sys
from pathlib import Path
values = [int(value) for value in sys.argv[2:]]
Path(sys.argv[1]).write_text(json.dumps({"suite_statuses": values}, indent=2) + "\n", encoding="utf-8")
raise SystemExit(1 if any(values) else 0)
PY
}

pids=()
for slot in 0 1 2 3; do
  run_lane "${slot}" >"${batch_root}/slot${slot}.log" 2>&1 &
  pids+=("$!")
done
statuses=()
set +e
for pid in "${pids[@]}"; do
  wait "${pid}"
  statuses+=("$?")
done
set -e

state_inputs=()
for lane in "${global_lanes[@]}"; do
  lane_plan=${control_root}/lane${lane}_plan.json
  suite_count=$(python - "${lane_plan}" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["suites"]))
PY
  )
  for ((suite_index=0; suite_index<suite_count; suite_index++)); do
    state=${batch_root}/lane${lane}/suite${suite_index}/shard_0_states.json
    if [[ ! -f "${state}" ]]; then
      mkdir -p "$(dirname "${state}")"
      python - "${state}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"cases": []}) + "\n", encoding="utf-8")
PY
    fi
    state_inputs+=(--input "${state}")
  done
done

python scripts/merge_case_states.py \
  "${state_inputs[@]}" \
  --expected "${partition_expected}" \
  --fill-missing-reason "four-GPU residual partition did not emit a terminal state" \
  --output "${batch_root}/partial_merged_case_states.json"
python scripts/audit_case_states.py \
  --expected "${partition_expected}" \
  --states "${batch_root}/partial_merged_case_states.json" \
  --output "${batch_root}/partial_terminal_state_audit.json"
python - "${batch_root}/batch_status.json" "${statuses[@]}" <<'PY'
import json, sys
from pathlib import Path
payload = {
    "lane_statuses": [int(value) for value in sys.argv[2:]],
    "terminal_audit_completed": True,
    "residual_partition": True,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
find "${batch_root}" -type f \( -name '*.mp4' -o -name 'latents.pt' \) -print0 \
  | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"
