#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_WEIGHTS_DIR:?missing INFER_WEIGHTS_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ ${#assigned_gpus[@]} -lt 4 ]]; then
  echo "method completion v2 requires four assigned GPUs with distinct lanes" >&2
  exit 2
fi

bundle_root=${INFER_WEIGHTS_DIR}/input_bundle
batch_root=${INFER_OUTPUT_DIR}
paper_root=${batch_root}/paper_lane
calibration_file=${paper_root}/calibration_v2/method_params.json
capture=$(find "${paper_root}/rag_dense_39/qkv_captures" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | sort | head -n 1 || true)

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

python - "${calibration_file}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {"svg2_ar", "adacluster_ar", "svoo_ar", "scope_ar"}
if set(payload.get("method_params", {})) != expected:
    raise SystemExit("method calibration prep is incomplete")
PY
[[ -n "${capture}" ]] || { echo "missing RAG39 QKV capture" >&2; exit 3; }

correctness_root=${paper_root}/matched_100pct_v2
mkdir -p "${correctness_root}" "${paper_root}/method_shards_v2"

run_correctness_lane() {
  local device=${assigned_gpus[0]}
  local rag_status block_status compare_status benchmark_status
  set +e
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${correctness_root}/rag_dense_21 \
    LONGLIVE_INPUT_BUNDLE_ROOT=${bundle_root} \
    LONGLIVE_CONFIG_PATH=configs/inferhub/rag_dense_21.yaml \
    bash scripts/inferhub_entry.sh >"${correctness_root}/rag_dense_21.log" 2>&1
  rag_status=$?
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${correctness_root}/block64_100pct_21 \
    LONGLIVE_INPUT_BUNDLE_ROOT=${bundle_root} \
    LONGLIVE_CONFIG_PATH=configs/inferhub/block64_100pct_21.yaml \
    bash scripts/inferhub_entry.sh >"${correctness_root}/block64_100pct_21.log" 2>&1
  block_status=$?
  if [[ ${rag_status} -eq 0 && ${block_status} -eq 0 ]]; then
    python scripts/compare_tensors.py \
      --reference "${correctness_root}/rag_dense_21/latents.pt" \
      --candidate "${correctness_root}/block64_100pct_21/latents.pt" \
      --max-relative-l2 0.01 \
      --output "${correctness_root}/latent_equivalence.json" \
      >"${correctness_root}/latent_equivalence.log" 2>&1
    compare_status=$?
  else
    compare_status=4
  fi
  CUDA_VISIBLE_DEVICES=${device} python scripts/benchmark_route_backends.py \
    --capture "${capture}" \
    --output "${correctness_root}/backend_benchmark.json" \
    --method block64_history --density 0.25 \
    --warmup 5 --iterations 20 \
    >"${correctness_root}/backend_benchmark.log" 2>&1
  benchmark_status=$?
  set -e
  python - <<PY
import json
from pathlib import Path
payload = {
    "rag_dense_21_status": ${rag_status},
    "block64_100pct_21_status": ${block_status},
    "latent_equivalence_status": ${compare_status},
    "backend_benchmark_status": ${benchmark_status},
}
Path("${correctness_root}/lane_status.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY
  [[ ${rag_status} -eq 0 && ${block_status} -eq 0 && ${compare_status} -eq 0 && ${benchmark_status} -eq 0 ]]
}

run_paper_shard() {
  local shard_index=$1
  local device=${assigned_gpus[$((shard_index + 1))]}
  local output=${paper_root}/method_shards_v2/shard${shard_index}
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES=${device} INFER_OUTPUT_DIR=${output} \
    python scripts/run_loaded_method_suite.py \
      --suite configs/rag_smoke_paper.json \
      --method-params-file "${calibration_file}" \
      --shard-index "${shard_index}" --shard-count 3 \
      >"${output}/methods.log" 2>&1
}

run_correctness_lane >"${correctness_root}/lane.log" 2>&1 & pids=("$!")
for shard in 0 1 2; do
  run_paper_shard "${shard}" \
    >"${paper_root}/method_shards_v2/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

statuses=()
set +e
for pid in "${pids[@]}"; do
  wait "${pid}"
  statuses+=("$?")
done
set -e

python scripts/merge_case_states.py \
  --input "${paper_root}/method_shards_v2/shard0/shard_0_states.json" \
  --input "${paper_root}/method_shards_v2/shard1/shard_1_states.json" \
  --input "${paper_root}/method_shards_v2/shard2/shard_2_states.json" \
  --output "${paper_root}/paper_method_states_v2.json"

python - "${batch_root}/batch_status_v2.json" "${statuses[@]}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
statuses = [int(value) for value in sys.argv[2:]]
payload = {
    "correctness_lane_status": statuses[0],
    "paper_shard_statuses": statuses[1:],
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
if any(statuses):
    raise SystemExit(1)
PY

find "${batch_root}" -type f \( -name '*.mp4' -o -name 'latents.pt' \) -print0 \
  | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS_v2.txt"
