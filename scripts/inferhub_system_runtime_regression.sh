#!/usr/bin/env bash
set -Eeuo pipefail
source "${INFER_CODE_DIR:?}/scripts/inferhub_runtime_env.sh"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/system_runtime_regression.py --output "${INFER_OUTPUT_DIR}/runtime_regression.json"
python scripts/system_runtime_regression.py --large --output "${INFER_OUTPUT_DIR}/runtime_regression_large.json"
