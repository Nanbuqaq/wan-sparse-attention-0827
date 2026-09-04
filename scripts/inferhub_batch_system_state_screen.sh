#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

source "${INFER_CODE_DIR}/scripts/inferhub_runtime_env.sh"
IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ ${#assigned_gpus[@]} -ne 4 ]]; then
  echo "System state prompt screen requires exactly four GPUs" >&2
  exit 2
fi

batch_root=${INFER_OUTPUT_DIR}
control_root=${batch_root}/control
experiment_commit=$(git -C "${INFER_CODE_DIR}" rev-parse HEAD)
mkdir -p "${control_root}"
python scripts/build_dense_screen_expected.py \
  --candidates configs/system/state_prompt_candidates.json \
  --commit "${experiment_commit}" --runtime rag_dense --latent-frames 120 \
  --output "${control_root}/state_screen_expected.json"

pids=()
for lane in 0 1 2 3; do
  lane_root=${batch_root}/lane${lane}
  mkdir -p "${lane_root}"
  CUDA_VISIBLE_DEVICES=${assigned_gpus[${lane}]} INFER_OUTPUT_DIR=${lane_root} \
    python scripts/run_loaded_dense_screen.py \
      --runtime rag_dense --base-config configs/inferhub/rag_dense_21.yaml \
      --candidates configs/system/state_prompt_candidates.json \
      --experiment-commit "${experiment_commit}" \
      --shard-axis case --shard-index "${lane}" --shard-count 4 \
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
for lane in 0 1 2 3; do
  state=${batch_root}/lane${lane}/dense_screen_states.json
  if [[ ! -f "${state}" ]]; then
    mkdir -p "$(dirname "${state}")"
    echo '{"cases": []}' >"${state}"
  fi
  state_args+=(--input "${state}")
done
python scripts/merge_case_states.py \
  "${state_args[@]}" \
  --expected "${control_root}/state_screen_expected.json" \
  --fill-missing-reason "state prompt screen lane emitted no terminal state" \
  --output "${batch_root}/merged_state_screen_states.json"
python scripts/audit_case_states.py \
  --expected "${control_root}/state_screen_expected.json" \
  --states "${batch_root}/merged_state_screen_states.json" \
  --output "${batch_root}/state_screen_terminal_audit.json"
python - "${batch_root}/batch_status.json" "${statuses[@]}" <<'PY'
import json
import sys
from pathlib import Path
statuses = [int(value) for value in sys.argv[2:]]
payload = {"lane_statuses": statuses, "status": "pass" if not any(statuses) else "fail"}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
find "${batch_root}" -type f \( -name '*.mp4' -o -name 'latents.pt' \) -print0 \
  | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"
if printf '%s\n' "${statuses[@]}" | grep -qv '^0$'; then
  exit 1
fi
