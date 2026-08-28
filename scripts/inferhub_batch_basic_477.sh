#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

source "${INFER_CODE_DIR}/scripts/inferhub_runtime_env.sh"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ ${#assigned_gpus[@]} -lt 4 ]]; then
  echo "basic 477-frame matrix requires four assigned GPUs" >&2
  exit 2
fi

batch_root=${INFER_OUTPUT_DIR}
for lane in 0 1 2 3; do
  mkdir -p "${batch_root}/lane${lane}"
done

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

run_lane0() {
  local device=${assigned_gpus[0]}
  local native_dense_status rag_status
  set +e
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${batch_root}/lane0/native_dense \
    python scripts/run_loaded_dense_screen.py \
      --runtime native_dense \
      --base-config configs/inferhub/native_dense_21.yaml \
      --candidates "${dense_manifest}" \
      --latent-frames 120 \
      --experiment-commit "${experiment_commit}" \
      >"${batch_root}/lane0/native_dense.log" 2>&1
  native_dense_status=$?
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${batch_root}/lane0/rag_methods \
    python scripts/run_loaded_method_suite.py \
      --suite "${rag_suite}" \
      --shard-index 0 --shard-count 4 \
      --experiment-commit "${experiment_commit}" \
      >"${batch_root}/lane0/rag_methods.log" 2>&1
  rag_status=$?
  set -e
  python - <<PY
import json
from pathlib import Path
payload = {
    "native_dense_status": ${native_dense_status},
    "rag_shard0_status": ${rag_status},
}
Path("${batch_root}/lane0/lane_status.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
  [[ ${native_dense_status} -eq 0 && ${rag_status} -eq 0 ]]
}

run_lane1() {
  local device=${assigned_gpus[1]}
  local rag_dense_status rag_status
  set +e
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${batch_root}/lane1/rag_dense \
    LONGLIVE_CAPTURE_QKV=1 \
    LONGLIVE_CAPTURE_LAYERS=0 \
    LONGLIVE_CAPTURE_STARTS=28080,93600,177840 \
    LONGLIVE_CAPTURE_MAX_PER_LAYER=3 \
    python scripts/run_loaded_dense_screen.py \
      --runtime rag_dense \
      --base-config configs/inferhub/rag_dense_21.yaml \
      --candidates "${dense_manifest}" \
      --latent-frames 120 \
      --experiment-commit "${experiment_commit}" \
      >"${batch_root}/lane1/rag_dense.log" 2>&1
  rag_dense_status=$?
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${batch_root}/lane1/rag_methods \
    python scripts/run_loaded_method_suite.py \
      --suite "${rag_suite}" \
      --shard-index 1 --shard-count 4 \
      --experiment-commit "${experiment_commit}" \
      >"${batch_root}/lane1/rag_methods.log" 2>&1
  rag_status=$?
  set -e
  python - <<PY
import json
from pathlib import Path
payload = {"rag_dense_status": ${rag_dense_status}, "rag_shard1_status": ${rag_status}}
Path("${batch_root}/lane1/lane_status.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
  [[ ${rag_dense_status} -eq 0 && ${rag_status} -eq 0 ]]
}

run_lane2() {
  local device=${assigned_gpus[2]}
  local native_block_status rag_status
  set +e
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${batch_root}/lane2/native_block \
    python scripts/run_loaded_dense_screen.py \
      --runtime native_block \
      --base-config configs/inferhub/native_block_21.yaml \
      --candidates "${dense_manifest}" \
      --latent-frames 120 \
      --experiment-commit "${experiment_commit}" \
      >"${batch_root}/lane2/native_block.log" 2>&1
  native_block_status=$?
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${batch_root}/lane2/rag_methods \
    python scripts/run_loaded_method_suite.py \
      --suite "${rag_suite}" \
      --shard-index 2 --shard-count 4 \
      --experiment-commit "${experiment_commit}" \
      >"${batch_root}/lane2/rag_methods.log" 2>&1
  rag_status=$?
  set -e
  python - <<PY
import json
from pathlib import Path
payload = {"native_block_status": ${native_block_status}, "rag_shard2_status": ${rag_status}}
Path("${batch_root}/lane2/lane_status.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
  [[ ${native_block_status} -eq 0 && ${rag_status} -eq 0 ]]
}

run_lane3() {
  local device=${assigned_gpus[3]}
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${batch_root}/lane3/rag_methods \
    python scripts/run_loaded_method_suite.py \
      --suite "${rag_suite}" \
      --shard-index 3 --shard-count 4 \
      --experiment-commit "${experiment_commit}" \
      >"${batch_root}/lane3/rag_methods.log" 2>&1
}

run_lane0 >"${batch_root}/lane0.log" 2>&1 & pid0=$!
run_lane1 >"${batch_root}/lane1.log" 2>&1 & pid1=$!
run_lane2 >"${batch_root}/lane2.log" 2>&1 & pid2=$!
run_lane3 >"${batch_root}/lane3.log" 2>&1 & pid3=$!
set +e
wait "${pid0}"; status0=$?
wait "${pid1}"; status1=$?
wait "${pid2}"; status2=$?
wait "${pid3}"; status3=$?
set -e
python - <<PY
import json
from pathlib import Path
payload = {
    "lane0_status": ${status0},
    "lane1_status": ${status1},
    "lane2_status": ${status2},
    "lane3_status": ${status3},
}
Path("${batch_root}/batch_status.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
find "${batch_root}" -type f \( -name '*.mp4' -o -name 'latents.pt' \) -print0 \
  | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"

state_inputs=()
for state in \
  "${batch_root}/lane0/native_dense/dense_screen_states.json" \
  "${batch_root}/lane0/rag_methods/shard_0_states.json" \
  "${batch_root}/lane1/rag_dense/dense_screen_states.json" \
  "${batch_root}/lane1/rag_methods/shard_1_states.json" \
  "${batch_root}/lane2/native_block/dense_screen_states.json" \
  "${batch_root}/lane2/rag_methods/shard_2_states.json" \
  "${batch_root}/lane3/rag_methods/shard_3_states.json"; do
  if [[ ! -f "${state}" ]]; then
    mkdir -p "$(dirname "${state}")"
    python - "${state}" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"cases": []}) + "\n", encoding="utf-8")
PY
  fi
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
