#!/usr/bin/env bash
# Four distinct Dense-only system configurations on an isolated development prompt.
set -Eeuo pipefail
source "${INFER_CODE_DIR:?}/scripts/inferhub_runtime_env.sh"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
IFS=',' read -r -a devices <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#devices[@]} == 4 ]] || { echo 'requires four GPUs' >&2; exit 2; }
batch_root=${INFER_OUTPUT_DIR}
python scripts/build_dense_system_validation.py --output-dir "${batch_root}/control"
pids=()
for lane in 0 1 2 3; do
  mkdir -p "${batch_root}/lane${lane}"
  CUDA_VISIBLE_DEVICES=${devices[$lane]} INFER_OUTPUT_DIR=${batch_root}/lane${lane} \
    python scripts/run_loaded_method_suite.py --suite "${batch_root}/control/lane${lane}.json" \
      --shard-axis case --shard-index 0 --shard-count 1 \
      >"${batch_root}/lane${lane}/runner.log" 2>&1 &
  pids+=("$!")
done
failure=0
for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
python scripts/merge_case_states.py \
  --input "${batch_root}/lane0/shard_0_states.json" \
  --input "${batch_root}/lane1/shard_0_states.json" \
  --input "${batch_root}/lane2/shard_0_states.json" \
  --input "${batch_root}/lane3/shard_0_states.json" \
  --expected "${batch_root}/control/expected.json" --output "${batch_root}/states.json"
python scripts/audit_case_states.py --expected "${batch_root}/control/expected.json" \
  --states "${batch_root}/states.json" --output "${batch_root}/terminal_audit.json"
exit "$failure"
