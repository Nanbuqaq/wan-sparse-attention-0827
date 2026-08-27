#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_WEIGHTS_DIR:?missing INFER_WEIGHTS_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ ${#assigned_gpus[@]} -lt 2 ]]; then
  echo "calibration/method smoke batch requires two assigned GPUs" >&2
  exit 2
fi

bundle_root=${INFER_WEIGHTS_DIR}/input_bundle
previous_capture_root=${INFER_WEIGHTS_DIR}/outputs/sparse-batch-gate-v3-20260827-r1/rag_dense_39_capture/qkv_captures

library_paths=("${VIRTUAL_ENV}/lib")
for library_dir in \
  "${VIRTUAL_ENV}"/lib/python*/site-packages/nvidia/*/lib \
  "${VIRTUAL_ENV}"/lib/python*/site-packages/torch/lib \
  "${VIRTUAL_ENV}"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ -d "${library_dir}" ]] && library_paths+=("${library_dir}")
done
joined_library_paths=$(IFS=:; echo "${library_paths[*]}")
export LD_LIBRARY_PATH="${joined_library_paths}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${bundle_root}/python-overlay:${INFER_CODE_DIR}:${INFER_CODE_DIR}/third_party/longlive-inferhub:${INFER_CODE_DIR}/third_party/LongLive-RAG${PYTHONPATH:+:${PYTHONPATH}}"
export LONGLIVE_BASE_SOURCE="${INFER_CODE_DIR}/third_party/longlive-inferhub"
export LONGLIVE_RAG_SOURCE="${INFER_CODE_DIR}/third_party/LongLive-RAG"
export LONGLIVE_WAN_MODELS_ROOT="${bundle_root}/model"
export LONGLIVE_GENERATOR_CKPT="${bundle_root}/checkpoints/longlive_init.pt"
export LONGLIVE_LORA_CKPT="${bundle_root}/checkpoints/longlive_lora_003000.pt"
export LONGLIVE_DISABLE_FA3=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

batch_root=${INFER_OUTPUT_DIR}
paper_root=${batch_root}/paper_lane
nonpaper_root=${batch_root}/nonpaper_lane
calibration_root=${paper_root}/calibration
rag39_root=${paper_root}/rag_dense_39
block100_root=${paper_root}/block64_100pct_21
mkdir -p "${calibration_root}" "${paper_root}/methods" "${nonpaper_root}/methods" "${rag39_root}" "${block100_root}"

run_paper_lane() {
  local device=${assigned_gpus[0]}
  local rag_status block_status calibration_status paper_status benchmark_status capture
  set +e
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${rag39_root} \
    LONGLIVE_INPUT_BUNDLE_ROOT=${bundle_root} \
    LONGLIVE_CONFIG_PATH=configs/inferhub/rag_dense_21.yaml \
    LONGLIVE_NUM_OUTPUT_FRAMES=39 \
    LONGLIVE_CAPTURE_QKV=1 \
    LONGLIVE_CAPTURE_LAYERS=0 \
    LONGLIVE_CAPTURE_MAX_PER_LAYER=1 \
    bash scripts/inferhub_entry.sh >"${paper_root}/rag_dense_39.log" 2>&1
  rag_status=$?
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${block100_root} \
    LONGLIVE_INPUT_BUNDLE_ROOT=${bundle_root} \
    LONGLIVE_CONFIG_PATH=configs/inferhub/block64_100pct_21.yaml \
    bash scripts/inferhub_entry.sh >"${paper_root}/block64_100pct_21.log" 2>&1
  block_status=$?
  set -e
  capture=$(find "${rag39_root}/qkv_captures" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | sort | head -n 1 || true)
  if [[ -z "${capture}" ]]; then
    capture=$(find "${previous_capture_root}" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | sort | head -n 1 || true)
  fi
  if [[ -z "${capture}" ]]; then
    echo "no QKV capture is available for calibration" >"${calibration_root}/calibration.log"
    calibration_status=3
  else
    set +e
  CUDA_VISIBLE_DEVICES=${device} python scripts/calibrate_methods_from_trace.py \
    --capture "${capture}" \
    --output "${calibration_root}/calibration.json" \
    --freeze-output "${calibration_root}/method_params.json" \
    --methods svg2_ar,adacluster_ar,svoo_ar,scope_ar \
    --head-limit 2 --recall-queries 16 --device cuda \
    >"${calibration_root}/calibration.log" 2>&1
    calibration_status=$?
    set -e
  fi
  set +e
  if [[ ${calibration_status} -eq 0 ]]; then
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${paper_root}/methods \
    python scripts/run_loaded_method_suite.py \
      --suite configs/rag_smoke_paper.json \
      --method-params-file "${calibration_root}/method_params.json" \
      --shard-index 0 --shard-count 1 \
      >"${paper_root}/methods.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${paper_root}/methods \
      python scripts/run_loaded_method_suite.py \
        --suite configs/rag_smoke_paper.json \
        --shard-index 0 --shard-count 1 \
        >"${paper_root}/methods.log" 2>&1
  fi
  paper_status=$?
  CUDA_VISIBLE_DEVICES=${device} python scripts/benchmark_route_backends.py \
    --capture "${capture}" \
    --output "${paper_root}/backend_benchmark.json" \
    --method block64_history --density 0.25 \
    --warmup 2 --iterations 5 \
    >"${paper_root}/backend_benchmark.log" 2>&1
  benchmark_status=$?
  set -e
  python - <<PY
import json
from pathlib import Path
payload = {
    "rag_dense_39_status": ${rag_status},
    "block64_100pct_21_status": ${block_status},
    "calibration_status": ${calibration_status},
    "paper_methods_status": ${paper_status},
    "backend_benchmark_status": ${benchmark_status},
}
Path("${paper_root}/lane_status.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
  [[ ${rag_status} -eq 0 && ${block_status} -eq 0 && ${calibration_status} -eq 0 && ${paper_status} -eq 0 && ${benchmark_status} -eq 0 ]]
}

run_nonpaper_lane() {
  local device=${assigned_gpus[1]}
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${nonpaper_root}/methods \
    python scripts/run_loaded_method_suite.py \
      --suite configs/rag_smoke_nonpaper.json \
      --shard-index 0 --shard-count 1 \
      >"${nonpaper_root}/methods.log" 2>&1
}

run_paper_lane >"${batch_root}/paper_lane.log" 2>&1 & pid0=$!
run_nonpaper_lane >"${batch_root}/nonpaper_lane.log" 2>&1 & pid1=$!
set +e
wait "${pid0}"; status0=$?
wait "${pid1}"; status1=$?
set -e
python - <<PY
import json
from pathlib import Path
payload = {"paper_lane_status": ${status0}, "nonpaper_lane_status": ${status1}}
Path("${batch_root}/batch_status.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
find "${batch_root}" -type f \( -name '*.mp4' -o -name 'latents.pt' \) -print0 \
  | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"
if [[ ${status0} -ne 0 || ${status1} -ne 0 ]]; then
  exit 1
fi
