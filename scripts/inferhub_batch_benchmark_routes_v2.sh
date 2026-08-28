#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_WEIGHTS_DIR:?missing INFER_WEIGHTS_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ ${#assigned_gpus[@]} -lt 4 ]]; then
  echo "route benchmark batch requires four assigned GPUs" >&2
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
export PYTHONPATH="${INFER_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

capture=$(find "${INFER_OUTPUT_DIR}/paper_lane/rag_dense_39/qkv_captures" \
  -maxdepth 1 -type f -name '*.pt' 2>/dev/null | sort | head -n 1 || true)
[[ -n "${capture}" ]] || { echo "missing verified RAG39 capture" >&2; exit 3; }

output_root=${INFER_OUTPUT_DIR}/paper_lane/backend_benchmarks_v2
mkdir -p "${output_root}"
densities=(0.10 0.15 0.25 1.00)

run_lane() {
  local lane=$1
  local device=${assigned_gpus[${lane}]}
  local density=${densities[${lane}]}
  local lane_root=${output_root}/lane${lane}_d${density/./}
  mkdir -p "${lane_root}/cache/triton" "${lane_root}/cache/torchinductor"
  CUDA_VISIBLE_DEVICES=${device} \
    TRITON_CACHE_DIR=${lane_root}/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=${lane_root}/cache/torchinductor \
    python scripts/benchmark_route_backends.py \
      --capture "${capture}" \
      --output "${lane_root}/benchmark.json" \
      --method block64_history --density "${density}" \
      --warmup 5 --iterations 20 \
      >"${lane_root}/benchmark.log" 2>&1
}

pids=()
for lane in 0 1 2 3; do
  run_lane "${lane}" >"${output_root}/lane${lane}.log" 2>&1 &
  pids+=("$!")
done

statuses=()
set +e
for pid in "${pids[@]}"; do
  wait "${pid}"
  statuses+=("$?")
done
set -e

python - "${output_root}/benchmark_summary.json" "${statuses[@]}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
statuses = [int(value) for value in sys.argv[2:]]
benchmarks = []
for lane in range(4):
    matches = sorted(output.parent.glob(f"lane{lane}_d*/benchmark.json"))
    if len(matches) == 1:
        benchmarks.append(json.loads(matches[0].read_text(encoding="utf-8")))
payload = {
    "status": "pass" if not any(statuses) and len(benchmarks) == 4 else "fail",
    "lane_statuses": statuses,
    "benchmarks": benchmarks,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
if payload["status"] != "pass":
    raise SystemExit(1)
PY

