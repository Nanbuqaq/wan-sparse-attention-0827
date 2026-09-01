#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"
: "${LONGLIVE_RESIDUAL_CONTROL_DIR:?missing LONGLIVE_RESIDUAL_CONTROL_DIR}"

source "${INFER_CODE_DIR}/scripts/inferhub_runtime_env.sh"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
lane_count=${#assigned_gpus[@]}
if [[ ${lane_count} -ne 8 ]]; then
  echo "formal-basic residual batch requires exactly eight assigned GPUs" >&2
  exit 2
fi

batch_root=${INFER_OUTPUT_DIR}
control_root=${LONGLIVE_RESIDUAL_CONTROL_DIR}
plan=${control_root}/residual_plan.json
expected=${control_root}/expected_basic_477.json
checkpoint_states=${control_root}/checkpoint_states.json
for required in "${plan}" "${expected}" "${checkpoint_states}" \
  "${control_root}/checkpoint_terminal_audit.json"; do
  [[ -f "${required}" ]] || { echo "missing residual control: ${required}" >&2; exit 3; }
done

experiment_commit=$(python - "${plan}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["experiment_commit"])
PY
)
actual_commit=$(git -C "${INFER_CODE_DIR}" rev-parse HEAD)
[[ "${actual_commit}" == "${experiment_commit}" ]] || {
  echo "residual runner checkout ${actual_commit} != experiment ${experiment_commit}" >&2
  exit 4
}

run_lane() {
  local lane=$1
  local device=${assigned_gpus[${lane}]}
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
for ((lane=0; lane<lane_count; lane++)); do
  run_lane "${lane}" >"${batch_root}/lane${lane}.log" 2>&1 &
  pids+=("$!")
done
statuses=()
set +e
for pid in "${pids[@]}"; do
  wait "${pid}"
  statuses+=("$?")
done
set -e

state_inputs=(--input "${checkpoint_states}")
for ((lane=0; lane<lane_count; lane++)); do
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
  --expected "${expected}" \
  --fill-missing-reason "eight-GPU residual runner did not emit a terminal state" \
  --output "${batch_root}/merged_case_states.json"
python scripts/audit_case_states.py \
  --expected "${expected}" \
  --states "${batch_root}/merged_case_states.json" \
  --output "${batch_root}/terminal_state_audit.json"
python - "${batch_root}/batch_status.json" "${statuses[@]}" <<'PY'
import json, sys
from pathlib import Path
payload = {
    "lane_statuses": [int(value) for value in sys.argv[2:]],
    "terminal_audit_completed": True,
    "residual_batch": True,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
python - "${batch_root}/merged_case_states.json" "${batch_root}/SHA256SUMS.txt" <<'PY'
import hashlib, json, sys
from pathlib import Path
states = json.load(open(sys.argv[1], encoding="utf-8"))["cases"]
paths = set()
for case in states:
    if case.get("status") not in {"pass", "negative"}:
        continue
    video = Path(case["video"])
    latent = Path(case.get("latent", video.parent / "latents.pt"))
    paths.update((video, latent))
lines = []
for path in sorted(paths, key=str):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    lines.append(f"{digest.hexdigest()}  {path}")
Path(sys.argv[2]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
