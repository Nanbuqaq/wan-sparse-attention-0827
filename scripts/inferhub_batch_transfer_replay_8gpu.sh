#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_WEIGHTS_DIR:?missing INFER_WEIGHTS_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

export LONGLIVE_INPUT_BUNDLE_ROOT="${INFER_WEIGHTS_DIR}/input_bundle"
source "${INFER_CODE_DIR}/scripts/inferhub_runtime_env.sh"
IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ ${#assigned_gpus[@]} -ne 8 ]]; then
  echo "transfer replay batch requires exactly eight GPUs" >&2
  exit 2
fi

capture_root=${INFER_WEIGHTS_DIR}/outputs/sparse-formal-basic-694da9e-v1/lane1/rag_dense/qkv_captures
captures=(
  layer00_start00028080.pt
  layer00_start00093600.pt
  layer09_start00177840.pt
  layer19_start00028080.pt
  layer19_start00093600.pt
  layer29_start00177840.pt
  layer09_start00093600.pt
  layer29_start00028080.pt
)
splits=(calibration calibration calibration calibration calibration calibration holdout holdout)
batch_root=${INFER_OUTPUT_DIR}
mkdir -p "${batch_root}/control"

python - "${batch_root}/control/expected.json" "$(git rev-parse HEAD)" <<'PY'
import json
import sys
from pathlib import Path

captures = [
    "layer00_start00028080.pt",
    "layer00_start00093600.pt",
    "layer09_start00177840.pt",
    "layer19_start00028080.pt",
    "layer19_start00093600.pt",
    "layer29_start00177840.pt",
    "layer09_start00093600.pt",
    "layer29_start00028080.pt",
]
splits = ["calibration"] * 6 + ["holdout"] * 2
payload = {
    "artifact_id": "h200_transfer_replay_8gpu_v1",
    "commit": sys.argv[2],
    "expected_lanes": 8,
    "warmup": 5,
    "iterations": 30,
    "cases": [
        {"lane": index, "capture": capture, "split": splits[index]}
        for index, capture in enumerate(captures)
    ],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

run_lane() {
  local lane=$1
  local lane_root=${batch_root}/lane${lane}_${splits[${lane}]}
  local capture=${capture_root}/${captures[${lane}]}
  [[ -f "${capture}" ]] || { echo "missing capture: ${capture}" >&2; return 3; }
  mkdir -p "${lane_root}/cache/triton" "${lane_root}/cache/torchinductor"
  CUDA_VISIBLE_DEVICES=${assigned_gpus[${lane}]} \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    TRITON_CACHE_DIR=${lane_root}/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=${lane_root}/cache/torchinductor \
    python scripts/benchmark_transfer_layouts.py \
      --capture "${capture}" \
      --warmup 5 --iterations 30 \
      --output "${lane_root}/benchmark.json" \
      >"${lane_root}/benchmark.log" 2>&1
}

pids=()
for lane in 0 1 2 3 4 5 6 7; do
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

python - "${batch_root}/terminal_audit.json" "${statuses[@]}" <<'PY'
import json
import sys
from pathlib import Path

statuses = [int(value) for value in sys.argv[2:]]
cases = []
for lane, status in enumerate(statuses):
    matches = sorted(Path(sys.argv[1]).parent.glob(f"lane{lane}_*/benchmark.json"))
    cases.append(
        {
            "lane": lane,
            "returncode": status,
            "state": "pass" if status == 0 and len(matches) == 1 else "fail",
            "benchmark": str(matches[0]) if len(matches) == 1 else None,
        }
    )
payload = {
    "status": "pass" if all(case["state"] == "pass" for case in cases) else "fail",
    "expected": 8,
    "observed": len(cases),
    "missing": sum(case["benchmark"] is None for case in cases),
    "cases": cases,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
if payload["status"] != "pass" or payload["missing"] != 0:
    raise SystemExit(1)
PY

find "${batch_root}" -type f \( -name '*.json' -o -name '*.log' \) -print0 \
  | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"
