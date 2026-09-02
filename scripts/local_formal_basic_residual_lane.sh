#!/usr/bin/env bash
set -Eeuo pipefail

lane=${1:?lane index required}
control_root=${2:?control root required}
method_repo=${3:?method repo required}
output_root=${4:?output root required}
log_root=${5:?log root required}
experiment_commit=${6:?experiment commit required}

[[ "${lane}" == 0 || "${lane}" == 1 ]] || { echo "local residual lane must be 0 or 1" >&2; exit 2; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "CUDA_VISIBLE_DEVICES is required" >&2; exit 2; }
[[ $(git -C "${method_repo}" rev-parse HEAD) == "${experiment_commit}" ]] || {
  echo "method worktree commit mismatch" >&2
  exit 3
}
[[ -z $(git -C "${method_repo}" status --porcelain) ]] || {
  echo "method worktree is dirty" >&2
  exit 3
}

bundle_root=${LONGLIVE_INPUT_BUNDLE_ROOT:?LONGLIVE_INPUT_BUNDLE_ROOT is required}
runtime_source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export LONGLIVE_INPUT_BUNDLE_ROOT=${bundle_root}
export LONGLIVE_BASE_SOURCE=${runtime_source_root}/third_party/longlive-inferhub
export LONGLIVE_RAG_SOURCE=${runtime_source_root}/third_party/LongLive-RAG
export LONGLIVE_PYTHON_OVERLAY=${bundle_root}/python-overlay
export LONGLIVE_WAN_MODELS_ROOT=${bundle_root}/model
export LONGLIVE_GENERATOR_CKPT=${bundle_root}/checkpoints/longlive_init.pt
export LONGLIVE_LORA_CKPT=${bundle_root}/checkpoints/longlive_lora_003000.pt
export PYTHONPATH="${LONGLIVE_PYTHON_OVERLAY}:${method_repo}:${LONGLIVE_BASE_SOURCE}:${LONGLIVE_RAG_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export LONGLIVE_DISABLE_FA3=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

[[ -f "${LONGLIVE_BASE_SOURCE}/pipeline/__init__.py" ]] || {
  echo "verified LongLive pipeline source missing: ${LONGLIVE_BASE_SOURCE}" >&2
  exit 3
}

mkdir -p "${output_root}/lane${lane}" "${log_root}"
mapfile -t suites < <(python3 - "${control_root}/lane${lane}_plan.json" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["suites"]:
    print(item["suite"])
PY
)
statuses=()
suite_index=0
set +e
for suite_name in "${suites[@]}"; do
  run_root=${output_root}/lane${lane}/suite${suite_index}
  mkdir -p "${run_root}"
  (
    cd "${method_repo}"
    INFER_OUTPUT_DIR=${run_root} \
      /usr/bin/python3 scripts/run_loaded_method_suite.py \
        --suite "${control_root}/${suite_name}" \
        --base-config configs/inferhub/rag_method_21.yaml \
        --shard-index 0 --shard-count 1 \
        --experiment-commit "${experiment_commit}"
  ) >"${log_root}/lane${lane}_suite${suite_index}.log" 2>&1
  statuses+=("$?")
  suite_index=$((suite_index + 1))
done
set -e
python3 - "${output_root}/lane${lane}/lane_status.json" "${statuses[@]}" <<'PY'
import json, sys
from pathlib import Path
values = [int(value) for value in sys.argv[2:]]
Path(sys.argv[1]).write_text(json.dumps({"suite_statuses": values}, indent=2) + "\n", encoding="utf-8")
raise SystemExit(1 if any(values) else 0)
PY
