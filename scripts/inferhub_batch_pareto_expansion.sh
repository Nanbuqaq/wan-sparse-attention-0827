#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"
: "${LONGLIVE_PARETO_SELECTION:?missing LONGLIVE_PARETO_SELECTION}"

source "${INFER_CODE_DIR}/scripts/inferhub_runtime_env.sh"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
lane_count=${#assigned_gpus[@]}
if [[ ${lane_count} -ne 4 && ${lane_count} -ne 8 ]]; then
  echo "Pareto expansion requires exactly four or eight assigned GPUs" >&2
  exit 2
fi

batch_root=${INFER_OUTPUT_DIR}
control_root=${batch_root}/control
experiment_commit=$(git -C "${INFER_CODE_DIR}" rev-parse HEAD)
python scripts/build_pareto_suites.py \
  --frozen-prompts "${INFER_CODE_DIR}/configs/formal/frozen_prompts.json" \
  --selection "${LONGLIVE_PARETO_SELECTION}" \
  --calibration "${INFER_CODE_DIR}/configs/formal/method_params.json" \
  --commit "${experiment_commit}" \
  --output-dir "${control_root}"

run_lane() {
  local lane=$1
  local device=${assigned_gpus[${lane}]}
  local lane_root=${batch_root}/lane${lane}
  local dense_status sparse_status
  mkdir -p "${lane_root}/dense" "${lane_root}/sparse"
  set +e
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${lane_root}/dense \
    python scripts/run_loaded_dense_screen.py \
      --runtime rag_dense \
      --base-config configs/inferhub/rag_dense_21.yaml \
      --candidates "${control_root}/rag_dense_pareto_expansion.json" \
      --experiment-commit "${experiment_commit}" \
      --shard-index "${lane}" --shard-count "${lane_count}" \
      >"${lane_root}/dense.log" 2>&1
  dense_status=$?
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${lane_root}/sparse \
    python scripts/run_loaded_method_suite.py \
      --suite "${control_root}/rag_pareto_expansion.json" \
      --experiment-commit "${experiment_commit}" \
      --shard-axis case \
      --shard-index "${lane}" --shard-count "${lane_count}" \
      >"${lane_root}/sparse.log" 2>&1
  sparse_status=$?
  set -e
  python - "${lane_root}/lane_status.json" "${dense_status}" "${sparse_status}" <<'PY'
import json
import sys
from pathlib import Path
payload = {"dense_status": int(sys.argv[2]), "sparse_status": int(sys.argv[3])}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  [[ ${dense_status} -eq 0 && ${sparse_status} -eq 0 ]]
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

state_inputs=()
for ((lane=0; lane<lane_count; lane++)); do
  for state in \
    "${batch_root}/lane${lane}/dense/dense_screen_states.json" \
    "${batch_root}/lane${lane}/sparse/shard_${lane}_states.json"; do
    if [[ ! -f "${state}" ]]; then
      python - "${state}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"cases": []}) + "\n", encoding="utf-8")
PY
    fi
    state_inputs+=(--input "${state}")
  done
done
python scripts/merge_case_states.py \
  "${state_inputs[@]}" \
  --expected "${control_root}/expected_pareto_expansion.json" \
  --fill-missing-reason "Pareto expansion runner did not emit a terminal state" \
  --output "${batch_root}/merged_case_states.json"
python scripts/audit_case_states.py \
  --expected "${control_root}/expected_pareto_expansion.json" \
  --states "${batch_root}/merged_case_states.json" \
  --output "${batch_root}/terminal_state_audit.json"
python - "${batch_root}/batch_status.json" "${statuses[@]}" <<'PY'
import json
import sys
from pathlib import Path
statuses = [int(value) for value in sys.argv[2:]]
payload = {"lane_statuses": statuses, "terminal_audit_completed": True}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
find "${batch_root}" -type f \( -name '*.mp4' -o -name 'latents.pt' \) -print0 \
  | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"
