#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_WEIGHTS_DIR:?missing INFER_WEIGHTS_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ ${#assigned_gpus[@]} -lt 2 ]]; then
  echo "batch gate v2 requires two assigned GPUs" >&2
  exit 2
fi

batch_root=${INFER_OUTPUT_DIR}
native_dense_output=${batch_root}/native_dense_21
native_block_output=${batch_root}/native_block_21
gate_output=${batch_root}/gpu_correctness
block_output=${batch_root}/block64_100pct_21
mkdir -p "${native_dense_output}" "${native_block_output}" "${gate_output}" "${block_output}"

run_native_cases() {
  CUDA_VISIBLE_DEVICES=${assigned_gpus[0]} INFER_OUTPUT_DIR=${native_dense_output} \
  LONGLIVE_CONFIG_PATH=configs/inferhub/native_dense_21.yaml \
    bash scripts/inferhub_entry.sh
  CUDA_VISIBLE_DEVICES=${assigned_gpus[0]} INFER_OUTPUT_DIR=${native_block_output} \
  LONGLIVE_CONFIG_PATH=configs/inferhub/native_block_21.yaml \
    bash scripts/inferhub_entry.sh
}

run_gate_and_block() {
  CUDA_VISIBLE_DEVICES=${assigned_gpus[1]} INFER_OUTPUT_DIR=${gate_output} \
    bash scripts/inferhub_gpu_gate.sh
  CUDA_VISIBLE_DEVICES=${assigned_gpus[1]} INFER_OUTPUT_DIR=${block_output} \
  LONGLIVE_CONFIG_PATH=configs/inferhub/block64_100pct_21.yaml \
    bash scripts/inferhub_entry.sh
}

run_native_cases >"${batch_root}/native_cases.log" 2>&1 & pid0=$!
run_gate_and_block >"${batch_root}/gate_and_block.log" 2>&1 & pid1=$!
set +e
wait "${pid0}"; status0=$?
wait "${pid1}"; status1=$?
set -e
python - <<PY
import json
from pathlib import Path
payload = {"native_cases_status": ${status0}, "gate_and_block_status": ${status1}}
Path("${batch_root}/batch_status.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
find "${batch_root}" -type f -name '*.mp4' -print0 | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"
if [[ ${status0} -ne 0 || ${status1} -ne 0 ]]; then exit 1; fi

