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
if [[ ${#assigned_gpus[@]} -ne 5 ]]; then
  echo "system routing calibration requires exactly five GPUs" >&2
  exit 2
fi

batch_root=${INFER_OUTPUT_DIR}
control_root=${batch_root}/control
mkdir -p "${control_root}"
python scripts/validate_system_holdout_prompts.py
reuse_args=()
if [[ -n ${REUSE_DENSE_MOTION_COMMIT:-} ]]; then
  reuse_args=(--reuse-dense-motion-commit "$REUSE_DENSE_MOTION_COMMIT")
fi
python scripts/build_system_routing_calibration_suites.py \
  --commit "$(git rev-parse HEAD)" "${reuse_args[@]}" --output-dir "${control_root}"
config_ids=(
  rag_dense
  legacy_final
  legacy_final_top_p095
  system_utility_peak
  system_utility_count_uniform
)

run_lane() {
  local lane=$1
  local config_id=${config_ids[${lane}]}
  local lane_root=${batch_root}/lane${lane}_${config_id}
  mkdir -p "${lane_root}/cache/triton" "${lane_root}/cache/torchinductor"
  CUDA_VISIBLE_DEVICES=${assigned_gpus[${lane}]} \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    TRITON_CACHE_DIR=${lane_root}/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=${lane_root}/cache/torchinductor \
    INFER_OUTPUT_DIR=${lane_root} \
    python scripts/run_loaded_method_suite.py \
      --suite "${control_root}/suite_${config_id}.json" \
      --base-config configs/inferhub/rag_method_21.yaml \
      --shard-axis case --shard-index 0 --shard-count 1 \
      --experiment-commit "$(git rev-parse HEAD)" \
      >"${lane_root}/runner.log" 2>&1
}

pids=()
for lane in 0 1 2 3 4; do
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

state_args=()
for lane in 0 1 2 3 4; do
  matches=("${batch_root}"/lane${lane}_*/shard_0_states.json)
  if [[ ${#matches[@]} -ne 1 || ! -f ${matches[0]} ]]; then
    fallback=${batch_root}/lane${lane}_missing_states.json
    printf '%s\n' '{"cases": []}' >"${fallback}"
    state_args+=(--input "${fallback}")
  else
    state_args+=(--input "${matches[0]}")
  fi
done
python scripts/merge_case_states.py \
  "${state_args[@]}" --expected "${control_root}/expected.json" \
  --fill-missing-reason "routing calibration lane emitted no terminal state" \
  --output "${batch_root}/merged_states.json"
python scripts/audit_case_states.py \
  --expected "${control_root}/expected.json" \
  --states "${batch_root}/merged_states.json" \
  --output "${batch_root}/terminal_audit.json"
python - "${batch_root}/batch_status.json" "${statuses[@]}" <<'PY'
import json
import sys
from pathlib import Path

statuses = [int(value) for value in sys.argv[2:]]
payload = {"status": "pass" if not any(statuses) else "fail", "lane_statuses": statuses}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
if payload["status"] != "pass":
    raise SystemExit(1)
PY
find "${batch_root}" -type f \( -name '*.json' -o -name '*.mp4' -o -name 'latents.pt' -o -name '*.log' \) -print0 \
  | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"
