#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

source "${INFER_CODE_DIR}/scripts/inferhub_runtime_env.sh"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
lane_count=${#assigned_gpus[@]}
if [[ ${lane_count} -ne 4 ]]; then
  echo "final 957-frame confirmation requires exactly four assigned GPUs" >&2
  exit 2
fi

batch_root=${INFER_OUTPUT_DIR}
control_root=${batch_root}/control
experiment_commit=$(git -C "${INFER_CODE_DIR}" rev-parse HEAD)
python scripts/build_final_long_confirmation_suite.py \
  --frozen-prompts "${INFER_CODE_DIR}/configs/formal/frozen_prompts.json" \
  --method-params "${INFER_CODE_DIR}/configs/formal/method_params.json" \
  --commit "${experiment_commit}" \
  --seed 20260826 \
  --output-dir "${control_root}"

dense_manifest=${control_root}/final_long_dense.json
sparse_suite=${control_root}/final_long_sparse.json
expected_manifest=${control_root}/expected_final_long.json

run_dense() {
  local runtime=$1
  local config=$2
  local device=$3
  local lane=$4
  local output=$5
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${output} \
    python scripts/run_loaded_dense_screen.py \
      --runtime "${runtime}" \
      --base-config "${config}" \
      --candidates "${dense_manifest}" \
      --latent-frames 240 \
      --experiment-commit "${experiment_commit}" \
      --shard-index "${lane}" --shard-count "${lane_count}"
}

run_lane() {
  local lane=$1
  local device=${assigned_gpus[${lane}]}
  local lane_root=${batch_root}/lane${lane}
  local native_dense_status=0
  local native_block_status=0
  local rag_dense_status=0
  local sparse_status=0
  mkdir -p "${lane_root}"
  set +e
  run_dense native_dense configs/inferhub/native_dense_21.yaml \
    "${device}" "${lane}" "${lane_root}/native_dense" \
    >"${lane_root}/native_dense.log" 2>&1
  native_dense_status=$?
  run_dense native_block configs/inferhub/native_block_21.yaml \
    "${device}" "${lane}" "${lane_root}/native_block" \
    >"${lane_root}/native_block.log" 2>&1
  native_block_status=$?
  run_dense rag_dense configs/inferhub/rag_dense_21.yaml \
    "${device}" "${lane}" "${lane_root}/rag_dense" \
    >"${lane_root}/rag_dense.log" 2>&1
  rag_dense_status=$?
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${lane_root}/sparse \
    python scripts/run_loaded_method_suite.py \
      --suite "${sparse_suite}" \
      --experiment-commit "${experiment_commit}" \
      --shard-axis case \
      --shard-index "${lane}" --shard-count "${lane_count}" \
      >"${lane_root}/sparse.log" 2>&1
  sparse_status=$?
  set -e
  python - "${lane_root}/lane_status.json" \
    "${native_dense_status}" "${native_block_status}" \
    "${rag_dense_status}" "${sparse_status}" <<'PY'
import json
import sys
from pathlib import Path
payload = {
    "native_dense_status": int(sys.argv[2]),
    "native_block_status": int(sys.argv[3]),
    "rag_dense_status": int(sys.argv[4]),
    "sparse_status": int(sys.argv[5]),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  [[ ${native_dense_status} -eq 0 && ${native_block_status} -eq 0 && \
     ${rag_dense_status} -eq 0 && ${sparse_status} -eq 0 ]]
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

ensure_state() {
  local state=$1
  if [[ ! -f "${state}" ]]; then
    mkdir -p "$(dirname "${state}")"
    python - "${state}" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"cases": []}) + "\n", encoding="utf-8")
PY
  fi
}

state_inputs=()
for ((lane=0; lane<lane_count; lane++)); do
  for state in \
    "${batch_root}/lane${lane}/native_dense/dense_screen_states.json" \
    "${batch_root}/lane${lane}/native_block/dense_screen_states.json" \
    "${batch_root}/lane${lane}/rag_dense/dense_screen_states.json" \
    "${batch_root}/lane${lane}/sparse/shard_${lane}_states.json"; do
    ensure_state "${state}"
    state_inputs+=(--input "${state}")
  done
done

python scripts/merge_case_states.py \
  "${state_inputs[@]}" \
  --expected "${expected_manifest}" \
  --fill-missing-reason "final 957-frame runner did not emit a terminal state" \
  --output "${batch_root}/merged_case_states.json"
python scripts/audit_case_states.py \
  --expected "${expected_manifest}" \
  --states "${batch_root}/merged_case_states.json" \
  --output "${batch_root}/terminal_state_audit.json"
python - "${batch_root}/batch_status.json" "${statuses[@]}" <<'PY'
import json
import sys
from pathlib import Path
payload = {
    "lane_statuses": [int(value) for value in sys.argv[2:]],
    "terminal_audit_completed": True,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
find "${batch_root}" -type f \( -name '*.mp4' -o -name 'latents.pt' \) -print0 \
  | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"
