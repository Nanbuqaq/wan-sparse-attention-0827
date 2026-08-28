#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_WEIGHTS_DIR:?missing INFER_WEIGHTS_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"

batch_root=${INFER_OUTPUT_DIR}
calibration_root=${batch_root}/paper_lane/calibration_v2
capture_root=${batch_root}/paper_lane/rag_dense_39/qkv_captures
fallback_capture_root=${INFER_WEIGHTS_DIR}/outputs/sparse-batch-gate-v3-20260827-r1/rag_dense_39_capture/qkv_captures
mkdir -p "${calibration_root}"

capture=$(find "${capture_root}" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | sort | head -n 1 || true)
if [[ -z "${capture}" ]]; then
  capture=$(find "${fallback_capture_root}" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | sort | head -n 1 || true)
fi
if [[ -z "${capture}" ]]; then
  echo "no verified QKV capture is available for method calibration" >&2
  exit 3
fi

python scripts/calibrate_methods_from_trace.py \
  --capture "${capture}" \
  --output "${calibration_root}/calibration.json" \
  --freeze-output "${calibration_root}/method_params.json" \
  --methods svg2_ar,adacluster_ar,svoo_ar,scope_ar \
  --head-limit 2 --recall-queries 16 --device cpu \
  >"${calibration_root}/calibration.log" 2>&1

python - "${calibration_root}/method_params.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
expected = {"svg2_ar", "adacluster_ar", "svoo_ar", "scope_ar"}
actual = set(payload.get("method_params", {}))
if payload.get("status") != "frozen_before_method_smoke" or actual != expected:
    raise SystemExit(
        f"incomplete method calibration: status={payload.get('status')!r} "
        f"methods={sorted(actual)} expected={sorted(expected)}"
    )
print(json.dumps({"status": "pass", "methods": sorted(actual)}, indent=2))
PY

