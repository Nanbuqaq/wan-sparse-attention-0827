#!/usr/bin/env bash
# Small mechanism probe: placement also changes displaced context and lifetime.
set -Eeuo pipefail
placement_gpu=${1:?physical GPU}
placement_output=${2:?new cohort output root}
placement_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
placement_work=/home/zhouhe08/MyProjects/0904-longlive-system
if [[ $placement_gpu == 0 ]]; then
  placement_scenario=generated_patchwork_toy_cut_revisit
  placement_order=(none global shot)
elif [[ $placement_gpu == 1 ]]; then
  placement_scenario=generated_bead_state_cut_revisit
  placement_order=(shot global none)
else exit 2; fi
export PYTHONPATH="/kaimm-distill/zhouhe08/longlive/input_bundle/python-overlay:$placement_repo"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0
cd "$placement_repo"
mkdir -p "$placement_output/lane$placement_gpu"
placement_failed=0
for placement_arm in "${placement_order[@]}"; do
  placement_mode=raw_reveal; placement_destination=$placement_arm
  if [[ $placement_arm == none ]]; then placement_mode=none; placement_destination=global; fi
  placement_case=$placement_output/lane$placement_gpu/$placement_arm
  [[ ! -e "$placement_case" && ! -e "$placement_case.log" ]]
  /usr/bin/python3 scripts/run_on_free_gpu.py --physical-gpu "$placement_gpu" -- \
    /usr/bin/python3 scripts/run_longlive2_native_reference.py --gate \
    --cut-scenario "$placement_scenario" --episode-memory-mode "$placement_mode" \
    --episode-destination "$placement_destination" --fixed-adaln-warps 16 --fixed-adaln-stages 1 \
    --source "$placement_work/publish_repo/third_party/LongLive2" \
    --assets "$placement_work/.runtime/models/longlive2-native-bf16" \
    --output "$placement_case" --seed 20260904 \
    >"$placement_case.log" 2>&1 || placement_failed=1
done
exit "$placement_failed"
