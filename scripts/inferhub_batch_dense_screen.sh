#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

source "${INFER_CODE_DIR}/scripts/inferhub_runtime_env.sh"

IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ ${#assigned_gpus[@]} -lt 4 ]]; then
  echo "Dense prompt screen requires four assigned GPUs" >&2
  exit 2
fi

batch_root=${INFER_OUTPUT_DIR}
experiment_commit=$(git -C "${INFER_CODE_DIR}" rev-parse HEAD)
mkdir -p "${batch_root}/native_dense" "${batch_root}/rag_dense"
control_root=${batch_root}/control
python scripts/build_dense_screen_expected.py \
  --commit "${experiment_commit}" --runtime native_dense \
  --output "${control_root}/native_dense_expected.json"
python scripts/build_dense_screen_expected.py \
  --commit "${experiment_commit}" --runtime rag_dense \
  --output "${control_root}/rag_dense_expected.json"
python scripts/build_dense_screen_expected.py \
  --commit "${experiment_commit}" --runtime all \
  --output "${control_root}/dense_screen_expected.json"

pids=()
for lane in 0 1 2 3; do
  if (( lane < 2 )); then
    runtime=native_dense
    shard=${lane}
    base_config=configs/inferhub/native_dense_21.yaml
  else
    runtime=rag_dense
    shard=$((lane - 2))
    base_config=configs/inferhub/rag_dense_21.yaml
  fi
  lane_root=${batch_root}/${runtime}/shard${shard}
  mkdir -p "${lane_root}"
  CUDA_VISIBLE_DEVICES=${assigned_gpus[${lane}]} INFER_OUTPUT_DIR=${lane_root} \
    python scripts/run_loaded_dense_screen.py \
      --runtime "${runtime}" --base-config "${base_config}" \
      --experiment-commit "${experiment_commit}" \
      --shard-index "${shard}" --shard-count 2 \
      >"${lane_root}/runner.log" 2>&1 &
  pids+=("$!")
done

statuses=()
set +e
for pid in "${pids[@]}"; do
  wait "${pid}"
  statuses+=("$?")
done
set -e
python - "${batch_root}/batch_status.json" "${statuses[@]}" <<'PY'
import json
import sys
from pathlib import Path
statuses = [int(value) for value in sys.argv[2:]]
payload = {"lane_statuses": statuses, "status": "pass" if not any(statuses) else "fail"}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
for state in \
  "${batch_root}/native_dense/shard0/dense_screen_states.json" \
  "${batch_root}/native_dense/shard1/dense_screen_states.json" \
  "${batch_root}/rag_dense/shard0/dense_screen_states.json" \
  "${batch_root}/rag_dense/shard1/dense_screen_states.json"; do
  if [[ ! -f "${state}" ]]; then
    mkdir -p "$(dirname "${state}")"
    python - "${state}" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"cases": []}) + "\n", encoding="utf-8")
PY
  fi
done
python scripts/merge_case_states.py \
  --input "${batch_root}/native_dense/shard0/dense_screen_states.json" \
  --input "${batch_root}/native_dense/shard1/dense_screen_states.json" \
  --expected "${control_root}/native_dense_expected.json" \
  --fill-missing-reason "Native Dense screen runner did not emit a terminal state" \
  --output "${batch_root}/native_dense/dense_screen_states.json"
python scripts/merge_case_states.py \
  --input "${batch_root}/rag_dense/shard0/dense_screen_states.json" \
  --input "${batch_root}/rag_dense/shard1/dense_screen_states.json" \
  --expected "${control_root}/rag_dense_expected.json" \
  --fill-missing-reason "RAG Dense screen runner did not emit a terminal state" \
  --output "${batch_root}/rag_dense/dense_screen_states.json"
python scripts/merge_case_states.py \
  --input "${batch_root}/native_dense/dense_screen_states.json" \
  --input "${batch_root}/rag_dense/dense_screen_states.json" \
  --expected "${control_root}/dense_screen_expected.json" \
  --output "${batch_root}/merged_dense_screen_states.json"
python scripts/audit_case_states.py \
  --expected "${control_root}/dense_screen_expected.json" \
  --states "${batch_root}/merged_dense_screen_states.json" \
  --output "${batch_root}/dense_screen_terminal_audit.json"
find "${batch_root}" -type f \( -name '*.mp4' -o -name 'latents.pt' \) -print0 \
  | sort -z | xargs -0 -r sha256sum >"${batch_root}/SHA256SUMS.txt"
if (( ${statuses[0]} != 0 || ${statuses[1]} != 0 || ${statuses[2]} != 0 || ${statuses[3]} != 0 )); then
  exit 1
fi
