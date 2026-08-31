#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

source "${INFER_CODE_DIR}/scripts/inferhub_runtime_env.sh"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
lane_count=${#assigned_gpus[@]}
if [[ ${lane_count} -ne 4 && ${lane_count} -ne 8 ]]; then
  echo "basic 477-frame matrix requires exactly four or eight assigned GPUs" >&2
  exit 2
fi

batch_root=${INFER_OUTPUT_DIR}
experiment_commit=$(git -C "${INFER_CODE_DIR}" rev-parse HEAD)
control_dir=${LONGLIVE_FORMAL_CONTROL_DIR:-${batch_root}/control}
if [[ -z "${LONGLIVE_FORMAL_CONTROL_DIR:-}" ]]; then
  frozen_prompts=${INFER_CODE_DIR}/configs/formal/frozen_prompts.json
  method_params=${INFER_CODE_DIR}/configs/formal/method_params.json
  [[ -f "${frozen_prompts}" ]] || { echo "missing frozen prompts: ${frozen_prompts}" >&2; exit 3; }
  [[ -f "${method_params}" ]] || { echo "missing method parameters: ${method_params}" >&2; exit 3; }
  python scripts/build_formal_suites.py \
    --frozen-prompts "${frozen_prompts}" \
    --calibration "${method_params}" \
    --commit "${experiment_commit}" \
    --output-dir "${control_dir}"
fi

dense_manifest=${control_dir}/dense_basic_477.json
rag_suite=${control_dir}/rag_basic_477.json
expected_manifest=${control_dir}/expected_basic_477.json
for control in "${dense_manifest}" "${rag_suite}" "${expected_manifest}"; do
  [[ -f "${control}" ]] || { echo "missing formal control: ${control}" >&2; exit 3; }
done

run_baseline() {
  local lane=$1
  local device=$2
  local lane_root=$3
  case "${lane}" in
    0)
      CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${lane_root}/native_dense \
        python scripts/run_loaded_dense_screen.py \
          --runtime native_dense \
          --base-config configs/inferhub/native_dense_21.yaml \
          --candidates "${dense_manifest}" \
          --latent-frames 120 \
          --experiment-commit "${experiment_commit}" \
          >"${lane_root}/native_dense.log" 2>&1
      ;;
    1)
      CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${lane_root}/rag_dense \
        LONGLIVE_CAPTURE_QKV=1 \
        LONGLIVE_CAPTURE_LAYERS=0,9,19,29 \
        LONGLIVE_CAPTURE_STARTS=28080,93600,177840 \
        LONGLIVE_CAPTURE_MAX_PER_LAYER=3 \
        python scripts/run_loaded_dense_screen.py \
          --runtime rag_dense \
          --base-config configs/inferhub/rag_dense_21.yaml \
          --candidates "${dense_manifest}" \
          --latent-frames 120 \
          --experiment-commit "${experiment_commit}" \
          >"${lane_root}/rag_dense.log" 2>&1
      ;;
    2)
      CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${lane_root}/native_block \
        python scripts/run_loaded_dense_screen.py \
          --runtime native_block \
          --base-config configs/inferhub/native_block_21.yaml \
          --candidates "${dense_manifest}" \
          --latent-frames 120 \
          --experiment-commit "${experiment_commit}" \
          >"${lane_root}/native_block.log" 2>&1
      ;;
    *) return 0 ;;
  esac
}

run_lane() {
  local lane=$1
  local device=${assigned_gpus[${lane}]}
  local lane_root=${batch_root}/lane${lane}
  local baseline_status=0
  local rag_status=0
  mkdir -p "${lane_root}"
  set +e
  run_baseline "${lane}" "${device}" "${lane_root}"
  baseline_status=$?
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${lane_root}/rag_methods \
    python scripts/run_loaded_method_suite.py \
      --suite "${rag_suite}" \
      --shard-index "${lane}" --shard-count "${lane_count}" \
      --experiment-commit "${experiment_commit}" \
      >"${lane_root}/rag_methods.log" 2>&1
  rag_status=$?
  set -e
  python - "${lane_root}/lane_status.json" "${baseline_status}" "${rag_status}" <<'PY'
import json
import sys
from pathlib import Path
payload = {"baseline_status": int(sys.argv[2]), "rag_status": int(sys.argv[3])}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  [[ ${baseline_status} -eq 0 && ${rag_status} -eq 0 ]]
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
for state in \
  "${batch_root}/lane0/native_dense/dense_screen_states.json" \
  "${batch_root}/lane1/rag_dense/dense_screen_states.json" \
  "${batch_root}/lane2/native_block/dense_screen_states.json"; do
  ensure_state "${state}"
  state_inputs+=(--input "${state}")
done
for ((lane=0; lane<lane_count; lane++)); do
  state=${batch_root}/lane${lane}/rag_methods/shard_${lane}_states.json
  ensure_state "${state}"
  state_inputs+=(--input "${state}")
done

python scripts/merge_case_states.py \
  "${state_inputs[@]}" \
  --expected "${expected_manifest}" \
  --fill-missing-reason "basic 477 runner failed before emitting a terminal state" \
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
