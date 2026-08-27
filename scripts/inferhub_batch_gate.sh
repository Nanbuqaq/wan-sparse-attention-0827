#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_WEIGHTS_DIR:?missing INFER_WEIGHTS_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ ${#assigned_gpus[@]} -lt 2 ]]; then
  echo "batch gate requires two assigned GPUs, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

batch_root=${INFER_OUTPUT_DIR}
dense_output=${batch_root}/rag_dense_21
gate_output=${batch_root}/gpu_correctness
block_output=${batch_root}/block64_100pct_21
mkdir -p "${dense_output}" "${gate_output}" "${block_output}"

run_dense() {
  CUDA_VISIBLE_DEVICES=${assigned_gpus[0]} \
  INFER_OUTPUT_DIR=${dense_output} \
  LONGLIVE_CONFIG_PATH=configs/inferhub/rag_dense_21.yaml \
    bash scripts/inferhub_entry.sh
}

run_gate_and_block() {
  CUDA_VISIBLE_DEVICES=${assigned_gpus[1]} \
  INFER_OUTPUT_DIR=${gate_output} \
    bash scripts/inferhub_gpu_gate.sh
  CUDA_VISIBLE_DEVICES=${assigned_gpus[1]} \
  INFER_OUTPUT_DIR=${block_output} \
  LONGLIVE_CONFIG_PATH=configs/inferhub/block64_100pct_21.yaml \
    bash scripts/inferhub_entry.sh
}

run_dense >"${batch_root}/rag_dense_21.log" 2>&1 &
dense_pid=$!
run_gate_and_block >"${batch_root}/gate_and_block.log" 2>&1 &
gate_pid=$!

set +e
wait "${dense_pid}"
dense_status=$?
wait "${gate_pid}"
gate_status=$?
set -e

python - <<PY
import json
from pathlib import Path
payload = {
    "assigned_gpus": ${#assigned_gpus[@]},
    "dense_status": ${dense_status},
    "gate_and_block_status": ${gate_status},
    "cases": {
        "rag_dense_21": "${dense_output}",
        "gpu_correctness": "${gate_output}",
        "block64_100pct_21": "${block_output}",
    },
}
Path("${batch_root}/batch_status.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

find "${batch_root}" -type f -name '*.mp4' -print0 | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"
if [[ ${dense_status} -ne 0 || ${gate_status} -ne 0 ]]; then
  exit 1
fi

