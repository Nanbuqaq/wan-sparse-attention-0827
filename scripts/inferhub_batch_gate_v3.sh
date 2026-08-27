#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_WEIGHTS_DIR:?missing INFER_WEIGHTS_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ ${#assigned_gpus[@]} -lt 2 ]]; then
  echo "batch gate v3 requires two assigned GPUs" >&2
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

batch_root=${INFER_OUTPUT_DIR}
gate_output=${batch_root}/gpu_correctness
block_output=${batch_root}/block64_100pct_21
rag39_output=${batch_root}/rag_dense_39_capture
mkdir -p "${gate_output}" "${block_output}" "${rag39_output}"

run_gate_and_block() {
  local gate_device=${assigned_gpus[0]}
  local bundle=${gate_output}/gate_bundle.pt
  local routing_status grouped_status fixed_status varlen_status block_status
  CUDA_VISIBLE_DEVICES=${gate_device} nvidia-smi

  set +e
  CUDA_VISIBLE_DEVICES=${gate_device} python scripts/gpu_correctness_gate.py \
    --mode prepare --bundle "${bundle}" --output "${gate_output}/routing_gate.json" \
    >"${gate_output}/routing_gate.log" 2>&1
  routing_status=$?
  set -e

  grouped_status=125
  fixed_status=125
  varlen_status=125
  if [[ ${routing_status} -eq 0 ]]; then
    for backend in grouped_fa2 fixed64_rect varlen_triton; do
      set +e
      CUDA_VISIBLE_DEVICES=${gate_device} python scripts/gpu_correctness_gate.py \
        --mode backend --backend "${backend}" --bundle "${bundle}" \
        --output "${gate_output}/backend_${backend}.json" \
        >"${gate_output}/backend_${backend}.log" 2>&1
      backend_status=$?
      set -e
      case "${backend}" in
        grouped_fa2) grouped_status=${backend_status} ;;
        fixed64_rect) fixed_status=${backend_status} ;;
        varlen_triton) varlen_status=${backend_status} ;;
      esac
    done
  fi

  set +e
  CUDA_VISIBLE_DEVICES=${gate_device} INFER_OUTPUT_DIR=${block_output} \
    LONGLIVE_CONFIG_PATH=configs/inferhub/block64_100pct_21.yaml \
    bash scripts/inferhub_entry.sh >"${batch_root}/block64_100pct_21.log" 2>&1
  block_status=$?
  set -e

  python - <<PY
import json
from pathlib import Path
payload = {
    "routing_status": ${routing_status},
    "grouped_fa2_status": ${grouped_status},
    "fixed64_rect_status": ${fixed_status},
    "varlen_triton_status": ${varlen_status},
    "block64_100pct_21_status": ${block_status},
    "kernel_failures_are_non_blocking": True,
}
Path("${batch_root}/gate_lane_status.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
  [[ ${routing_status} -eq 0 && ${grouped_status} -eq 0 && ${block_status} -eq 0 ]]
}

run_rag39_capture() {
  local rag_device=${assigned_gpus[1]}
  local rag_status
  CUDA_VISIBLE_DEVICES=${rag_device} nvidia-smi
  set +e
  CUDA_VISIBLE_DEVICES=${rag_device} INFER_OUTPUT_DIR=${rag39_output} \
    LONGLIVE_CONFIG_PATH=configs/inferhub/rag_dense_21.yaml \
    LONGLIVE_NUM_OUTPUT_FRAMES=39 \
    LONGLIVE_CAPTURE_QKV=1 \
    LONGLIVE_CAPTURE_LAYERS=0 \
    LONGLIVE_CAPTURE_MAX_PER_LAYER=1 \
    bash scripts/inferhub_entry.sh >"${batch_root}/rag_dense_39_capture.log" 2>&1
  rag_status=$?
  set -e
  python - <<PY
import json
from pathlib import Path
payload = {"rag_dense_39_capture_status": ${rag_status}}
Path("${batch_root}/rag_lane_status.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
  [[ ${rag_status} -eq 0 ]]
}

run_gate_and_block >"${batch_root}/gate_lane.log" 2>&1 & pid0=$!
run_rag39_capture >"${batch_root}/rag_lane.log" 2>&1 & pid1=$!
set +e
wait "${pid0}"; status0=$?
wait "${pid1}"; status1=$?
set -e
python - <<PY
import json
from pathlib import Path
payload = {"gate_lane_status": ${status0}, "rag_lane_status": ${status1}}
Path("${batch_root}/batch_status.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
find "${batch_root}" -type f \( -name '*.mp4' -o -name 'latents.pt' -o -path '*/qkv_captures/*.pt' \) -print0 \
  | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"
if [[ ${status0} -ne 0 || ${status1} -ne 0 ]]; then
  exit 1
fi
