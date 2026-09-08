#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
blackwell_root=$INFER_OUTPUT_DIR
blackwell_libs=$VIRTUAL_ENV/lib
for blackwell_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$blackwell_lib" ]] || blackwell_libs="$blackwell_libs:$blackwell_lib"
done
export LD_LIBRARY_PATH="$blackwell_libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_WEIGHTS_DIR/private-triton331:$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  python scripts/prepare_longlive2_reference.py --inputs "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2"
  python -c 'import triton; assert triton.__version__=="3.3.1"; from adapters.longlive_sparse.native_episode_memory import NativeEpisodeMemory; print("BLACKWELL_NATIVE_PREP_PASS")'
  touch "$blackwell_root/prepared.ok";exit 0
fi
IFS=',' read -r -a blackwell_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#blackwell_gpus[@]} == 8 ]] || { echo 'requires eight assigned GPUs' >&2; exit 2; }
blackwell_scenarios=(generated_patchwork_toy_cut_revisit generated_bead_state_cut_revisit)
blackwell_modes=(none raw_reveal raw_away log_reveal)
blackwell_pids=()
mkdir -p "$blackwell_root/gates/toy" "$blackwell_root/gates/state"
for blackwell_lane in "${!blackwell_gpus[@]}"; do
  blackwell_scenario=$((blackwell_lane/4));blackwell_mode=${blackwell_modes[$((blackwell_lane%4))]}
  blackwell_group=toy;[[ $blackwell_scenario == 0 ]] || blackwell_group=state
  CUDA_VISIBLE_DEVICES=${blackwell_gpus[$blackwell_lane]} python scripts/run_longlive2_native_reference.py \
    --assets "$INFER_WEIGHTS_DIR" --output "$blackwell_root/gates/$blackwell_group/$blackwell_mode" \
    --gate --seed 20260904 --cut-scenario "${blackwell_scenarios[$blackwell_scenario]}" \
    --episode-memory-mode "$blackwell_mode" --fixed-adaln-warps 16 --fixed-adaln-stages 1 \
    >"$blackwell_root/gate$blackwell_lane.log" 2>&1 &
  blackwell_pids+=("$!")
done
blackwell_gate_failed=0
for blackwell_pid in "${blackwell_pids[@]}"; do wait "$blackwell_pid" || blackwell_gate_failed=1; done
for blackwell_group in toy state; do
  python scripts/collect_native_episode_batch.py --gate --root "$blackwell_root/gates/$blackwell_group" || blackwell_gate_failed=1
done
if [[ $blackwell_gate_failed != 0 ]]; then
  python scripts/record_native_hardware_gate_stop.py --root "$blackwell_root"
  exit 2
fi
blackwell_pids=()
for blackwell_lane in "${!blackwell_gpus[@]}"; do
  (
    blackwell_scenario=$((blackwell_lane/4));blackwell_mode=${blackwell_modes[$((blackwell_lane%4))]}
    for blackwell_seed_index in 0 1; do
      blackwell_seed=$((20260913+blackwell_seed_index));blackwell_case_group=$((blackwell_scenario*2+blackwell_seed_index))
      mkdir -p "$blackwell_root/lane$blackwell_case_group"
      blackwell_code=0
      CUDA_VISIBLE_DEVICES=${blackwell_gpus[$blackwell_lane]} python scripts/run_longlive2_native_reference.py \
        --assets "$INFER_WEIGHTS_DIR" --output "$blackwell_root/lane$blackwell_case_group/$blackwell_mode" \
        --seed "$blackwell_seed" --cut-scenario "${blackwell_scenarios[$blackwell_scenario]}" \
        --episode-memory-mode "$blackwell_mode" --fixed-adaln-warps 16 --fixed-adaln-stages 1 \
        >"$blackwell_root/lane$blackwell_case_group/$blackwell_mode.log" 2>&1 || blackwell_code=$?
      echo "seed=$blackwell_seed mode=$blackwell_mode exit=$blackwell_code"
    done
  ) >"$blackwell_root/physical_lane$blackwell_lane.log" 2>&1 &
  blackwell_pids+=("$!")
done
for blackwell_pid in "${blackwell_pids[@]}"; do wait "$blackwell_pid" || true; done
python scripts/collect_native_episode_batch.py --root "$blackwell_root"
