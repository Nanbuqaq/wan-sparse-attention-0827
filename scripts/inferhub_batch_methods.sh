#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_WEIGHTS_DIR:?missing INFER_WEIGHTS_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ ${#assigned_gpus[@]} -lt 2 ]]; then
  echo "method batch requires two assigned GPUs" >&2
  exit 2
fi

library_paths=("${VIRTUAL_ENV}/lib")
for library_dir in \
  "${VIRTUAL_ENV}"/lib/python*/site-packages/nvidia/*/lib \
  "${VIRTUAL_ENV}"/lib/python*/site-packages/torch/lib \
  "${VIRTUAL_ENV}"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ -d "${library_dir}" ]] && library_paths+=("${library_dir}")
done
joined_library_paths=$(IFS=:; echo "${library_paths[*]}")
export LD_LIBRARY_PATH="${joined_library_paths}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${INFER_WEIGHTS_DIR}/python-overlay:${INFER_CODE_DIR}:${INFER_CODE_DIR}/third_party/longlive-inferhub:${INFER_CODE_DIR}/third_party/LongLive-RAG${PYTHONPATH:+:${PYTHONPATH}}"
export LONGLIVE_BASE_SOURCE="${INFER_CODE_DIR}/third_party/longlive-inferhub"
export LONGLIVE_RAG_SOURCE="${INFER_CODE_DIR}/third_party/LongLive-RAG"
export LONGLIVE_WAN_MODELS_ROOT="${INFER_WEIGHTS_DIR}/model"
export LONGLIVE_GENERATOR_CKPT="${INFER_WEIGHTS_DIR}/checkpoints/longlive_init.pt"
export LONGLIVE_LORA_CKPT="${INFER_WEIGHTS_DIR}/checkpoints/longlive_lora_003000.pt"
export LONGLIVE_DISABLE_FA3=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

batch_root=${INFER_OUTPUT_DIR}
mkdir -p "${batch_root}/shard0" "${batch_root}/shard1"

CUDA_VISIBLE_DEVICES=${assigned_gpus[0]} INFER_OUTPUT_DIR=${batch_root}/shard0 \
  python scripts/run_loaded_method_suite.py --shard-index 0 --shard-count 2 \
  >"${batch_root}/shard0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=${assigned_gpus[1]} INFER_OUTPUT_DIR=${batch_root}/shard1 \
  python scripts/run_loaded_method_suite.py --shard-index 1 --shard-count 2 \
  >"${batch_root}/shard1.log" 2>&1 &
pid1=$!

set +e
wait "${pid0}"; status0=$?
wait "${pid1}"; status1=$?
set -e
python - <<PY
import json
from pathlib import Path
payload = {"shard0_status": ${status0}, "shard1_status": ${status1}}
Path("${batch_root}/batch_status.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
find "${batch_root}" -type f -name '*.mp4' -print0 | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"
if [[ ${status0} -ne 0 || ${status1} -ne 0 ]]; then exit 1; fi

