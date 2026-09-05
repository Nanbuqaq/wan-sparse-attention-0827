#!/usr/bin/env bash
set -Eeuo pipefail
source "${INFER_CODE_DIR:?}/scripts/inferhub_runtime_env.sh"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LONGLIVE_COMPLETE_CAPTURE_STARTS=28080,46800
export LONGLIVE_COMPLETE_CAPTURE_LAYERS=0,19
IFS=',' read -r -a devices <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#devices[@]} == 4 ]] || { echo 'requires four GPUs' >&2; exit 2; }
batch=${INFER_OUTPUT_DIR}
python scripts/build_readiness_repair_calibration.py --output-dir "$batch/control"
names=(rag_dense legacy_final system_utility_peak system_utility_count_uniform)
pids=()
for lane in 0 1 2 3; do
  root=$batch/lane${lane}_${names[$lane]}
  mkdir -p "$root/cache"
  CUDA_VISIBLE_DEVICES=${devices[$lane]} INFER_OUTPUT_DIR=$root \
    TRITON_CACHE_DIR=$root/cache/triton TORCHINDUCTOR_CACHE_DIR=$root/cache/inductor \
    python scripts/run_loaded_method_suite.py --suite "$batch/control/suite_${names[$lane]}.json" \
    --shard-axis case --shard-index 0 --shard-count 1 >"$root/runner.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
python scripts/merge_case_states.py \
    --input "$batch/lane0_rag_dense/shard_0_states.json" \
    --input "$batch/lane1_legacy_final/shard_0_states.json" \
    --input "$batch/lane2_system_utility_peak/shard_0_states.json" \
    --input "$batch/lane3_system_utility_count_uniform/shard_0_states.json" \
    --expected "$batch/control/expected.json" --output "$batch/states.json"
python scripts/audit_case_states.py --expected "$batch/control/expected.json" \
    --states "$batch/states.json" --output "$batch/terminal_audit.json"
exit "$failed"
