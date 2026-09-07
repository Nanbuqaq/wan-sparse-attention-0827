#!/usr/bin/env bash
set -Eeuo pipefail
joint_output=${1:?new output directory}
joint_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
joint_workspace=/home/zhouhe08/MyProjects/0904-longlive-system
joint_bundle=/kaimm-distill/zhouhe08/longlive/input_bundle
[[ ! -e "$joint_output" ]]
mkdir -p "$joint_output"
export LONGLIVE_BASE_SOURCE=$joint_workspace/publish_repo/third_party/longlive-inferhub
export LONGLIVE_RAG_SOURCE=$joint_workspace/publish_repo/third_party/LongLive-RAG
export PYTHONPATH="$joint_bundle/python-overlay:$joint_repo:$LONGLIVE_BASE_SOURCE:$LONGLIVE_RAG_SOURCE"
export LONGLIVE_DISABLE_FA3=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd "$joint_repo"
/usr/bin/python3 scripts/gate_dense_rope.py --output "$joint_output/rope_operator.json"
for joint_method in rag_dense transfer_vaware_hybrid_history; do
  INFER_OUTPUT_DIR=$joint_output/$joint_method /usr/bin/python3 scripts/system_runtime_regression.py \
    --output "$joint_output/$joint_method.json" --repeats 3 --rope-comparison --method-filter "$joint_method"
done
/usr/bin/python3 scripts/gate_committed_moment_runtime.py --output "$joint_output/committed_small.json"
/usr/bin/python3 scripts/gate_committed_moment_runtime.py --large --output "$joint_output/committed_large.json"
