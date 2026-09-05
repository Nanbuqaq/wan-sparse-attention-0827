#!/usr/bin/env bash
# Local physically-locked synthetic regression; distinct output root per run.
set -Eeuo pipefail
gpu=${1:?physical GPU required}
replay_root=${2:?absolute output root required}
shift 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ -z $(git -C "$repo" status --porcelain) ]] || { echo 'requires clean source' >&2; exit 2; }
[[ ! -e ${replay_root}/runtime_regression.json ]] || { echo 'existing terminal result; do not duplicate' >&2; exit 2; }
mkdir -p "$replay_root"
export PYTHONPATH="${LONGLIVE_INPUT_BUNDLE_ROOT:-/kaimm-distill/zhouhe08/longlive/input_bundle}/python-overlay:$repo:$repo/third_party/longlive-inferhub:$repo/third_party/LongLive-RAG"
export LONGLIVE_BASE_SOURCE=$repo/third_party/longlive-inferhub
export LONGLIVE_RAG_SOURCE=$repo/third_party/LongLive-RAG
export LONGLIVE_DISABLE_FA3=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 INFER_OUTPUT_DIR=$replay_root
command=(/usr/bin/python3 "$repo/scripts/system_runtime_regression.py"
         --output "$replay_root/runtime_regression.json" "$@")
if [[ -n ${NSYS_PROFILE_POLICY:-} ]]; then
  command=(/usr/local/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
      --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false
      --output="$replay_root/timeline" "${command[@]}" --profile-policy "$NSYS_PROFILE_POLICY")
fi
cd "$repo"
/usr/bin/python3 scripts/run_on_free_gpu.py --physical-gpu "$gpu" -- "${command[@]}" 2>&1 | tee "$replay_root/runner.log"
