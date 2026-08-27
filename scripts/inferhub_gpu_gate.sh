#!/usr/bin/env bash
set -Eeuo pipefail

: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_WEIGHTS_DIR:?missing INFER_WEIGHTS_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

library_paths=("${VIRTUAL_ENV}/lib")
for library_dir in \
  "${VIRTUAL_ENV}"/lib/python*/site-packages/nvidia/*/lib \
  "${VIRTUAL_ENV}"/lib/python*/site-packages/torch/lib \
  "${VIRTUAL_ENV}"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ -d "${library_dir}" ]] && library_paths+=("${library_dir}")
done
joined_library_paths=$(IFS=:; echo "${library_paths[*]}")
export LD_LIBRARY_PATH="${joined_library_paths}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${INFER_WEIGHTS_DIR}/python-overlay:${INFER_CODE_DIR}:${INFER_CODE_DIR}/third_party/longlive-inferhub${PYTHONPATH:+:${PYTHONPATH}}"
export LONGLIVE_DISABLE_FA3=1
export PYTHONUNBUFFERED=1
nvidia-smi
python scripts/gpu_correctness_gate.py

