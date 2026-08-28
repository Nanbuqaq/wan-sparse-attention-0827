#!/usr/bin/env bash

# Resolve the InferHub input bundle explicitly. This file is sourced by batch
# entrypoints so no public config needs to contain cluster-local absolute paths.
configure_longlive_runtime() {
  : "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
  : "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
  : "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
  : "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"

  local bundle_root=${LONGLIVE_INPUT_BUNDLE_ROOT:-}
  if [[ -z "${bundle_root}" && -n "${INFER_WEIGHTS_DIR:-}" ]]; then
    if [[ -d "${INFER_WEIGHTS_DIR}/input_bundle" ]]; then
      bundle_root=${INFER_WEIGHTS_DIR}/input_bundle
    else
      bundle_root=${INFER_WEIGHTS_DIR}
    fi
  fi
  [[ -n "${bundle_root}" ]] || {
    echo "missing LONGLIVE_INPUT_BUNDLE_ROOT (or legacy INFER_WEIGHTS_DIR)" >&2
    return 2
  }
  [[ -d "${bundle_root}/python-overlay" ]] || {
    echo "input bundle missing python-overlay: ${bundle_root}" >&2
    return 2
  }
  [[ -d "${bundle_root}/model" ]] || {
    echo "input bundle missing model directory: ${bundle_root}" >&2
    return 2
  }
  [[ -f "${bundle_root}/checkpoints/longlive_init.pt" ]] || {
    echo "input bundle missing generator checkpoint: ${bundle_root}" >&2
    return 2
  }
  [[ -f "${bundle_root}/checkpoints/longlive_lora_003000.pt" ]] || {
    echo "input bundle missing LoRA checkpoint: ${bundle_root}" >&2
    return 2
  }

  export LONGLIVE_INPUT_BUNDLE_ROOT=${bundle_root}
  export LONGLIVE_BASE_SOURCE="${INFER_CODE_DIR}/third_party/longlive-inferhub"
  export LONGLIVE_RAG_SOURCE="${INFER_CODE_DIR}/third_party/LongLive-RAG"
  export LONGLIVE_PYTHON_OVERLAY="${bundle_root}/python-overlay"
  export LONGLIVE_WAN_MODELS_ROOT="${bundle_root}/model"
  export LONGLIVE_GENERATOR_CKPT="${bundle_root}/checkpoints/longlive_init.pt"
  export LONGLIVE_LORA_CKPT="${bundle_root}/checkpoints/longlive_lora_003000.pt"
  export PYTHONPATH="${LONGLIVE_PYTHON_OVERLAY}:${INFER_CODE_DIR}:${LONGLIVE_BASE_SOURCE}:${LONGLIVE_RAG_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"

  local library_paths=("${VIRTUAL_ENV}/lib")
  local library_dir
  for library_dir in \
    "${VIRTUAL_ENV}"/lib/python*/site-packages/nvidia/*/lib \
    "${VIRTUAL_ENV}"/lib/python*/site-packages/torch/lib \
    "${VIRTUAL_ENV}"/lib/python*/site-packages/triton/backends/nvidia/lib; do
    [[ -d "${library_dir}" ]] && library_paths+=("${library_dir}")
  done
  local joined_library_paths
  joined_library_paths=$(IFS=:; echo "${library_paths[*]}")
  export LD_LIBRARY_PATH="${joined_library_paths}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export LONGLIVE_DISABLE_FA3=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONUNBUFFERED=1
}

configure_longlive_runtime
