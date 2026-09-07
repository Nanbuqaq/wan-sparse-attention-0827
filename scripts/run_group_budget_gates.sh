#!/usr/bin/env bash
set -Eeuo pipefail
budget_output=${1:?new output directory}
budget_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
budget_workspace=/home/zhouhe08/MyProjects/0904-longlive-system
budget_bundle=/kaimm-distill/zhouhe08/longlive/input_bundle
[[ ! -e "$budget_output" ]]
mkdir -p "$budget_output"
export LONGLIVE_BASE_SOURCE=$budget_workspace/publish_repo/third_party/longlive-inferhub
export LONGLIVE_RAG_SOURCE=$budget_workspace/publish_repo/third_party/LongLive-RAG
export PYTHONPATH="$budget_bundle/python-overlay:$budget_repo:$LONGLIVE_BASE_SOURCE:$LONGLIVE_RAG_SOURCE"
export LONGLIVE_DISABLE_FA3=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd "$budget_repo"
for budget_method in transfer_vaware_hybrid_history group_relation_history; do
  INFER_OUTPUT_DIR=$budget_output/$budget_method /usr/bin/python3 scripts/system_runtime_regression.py \
    --output "$budget_output/$budget_method.json" --repeats 2 --backend-comparison \
    --method-filter "$budget_method" --history-density .5 --relation-admission shared --group-start-layer 0
done
