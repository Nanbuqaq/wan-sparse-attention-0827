"""Construction of the pinned LongLive-RAG runtime with sparse history hooks."""

from __future__ import annotations

import types
from typing import Any

import torch

from .archive import HistoryArchive
from .config import SparseHistoryConfig
from .runtime_attention import install_sparse_history_attention
from .stats import SparseRunStats
from .upstreams import (
    configure_upstream_paths,
    load_latentmem_module,
    load_rag_pipeline_module,
)


def _sparse_config_from_args(args: Any) -> SparseHistoryConfig:
    value = getattr(args, "sparse_history", None)
    if value is None:
        raise ValueError("sparse LongLive config requires a top-level sparse_history section")
    if hasattr(value, "items"):
        value = dict(value.items())
    return SparseHistoryConfig.from_mapping(value)


def build_sparse_pipeline(args: Any, device: torch.device | str):
    """Instantiate the read-only RAG pipeline and replace only self-attention."""

    if not torch.cuda.is_available():
        raise RuntimeError("LongLive sparse pipeline requires a CUDA-enabled process")
    configure_upstream_paths()
    import utils.wan_wrapper as base_wrapper

    latentmem_module = load_latentmem_module()
    rag_pipeline_module = load_rag_pipeline_module()
    sparse_config = _sparse_config_from_args(args)

    class SparseWanDiffusionWrapper(base_wrapper.WanDiffusionWrapper):
        def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            memory_size=0,
            use_latentmem=False,
            **unused,
        ):
            del use_latentmem, unused
            torch.nn.Module.__init__(self)
            if not is_causal:
                raise ValueError("LongLive sparse runtime requires is_causal=True")
            model_dir = base_wrapper._wan_model_dir(model_name)
            self.model = latentmem_module.CausalWanModel.from_pretrained(
                str(model_dir),
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                memory_size=memory_size,
            )
            self.model.eval()
            self.uniform_timestep = False
            self.scheduler = base_wrapper.FlowMatchScheduler(
                shift=timestep_shift,
                sigma_min=0.0,
                extra_one_step=True,
            )
            self.scheduler.set_timesteps(1000, training=True)
            self.seq_len = 1560 * local_attn_size if local_attn_size > 21 else 32760
            self.post_init()

    rag_pipeline_module.WanDiffusionWrapper = SparseWanDiffusionWrapper
    rag_pipeline_module.WanTextEncoder = base_wrapper.WanTextEncoder
    rag_pipeline_module.WanVAEWrapper = base_wrapper.WanVAEWrapper
    pipeline = rag_pipeline_module.CausalInferencePipeline(args, device=torch.device(device))
    archive = HistoryArchive(sparse_config, spatial_height=30, spatial_width=52)
    installed = install_sparse_history_attention(
        pipeline.generator.model,
        archive,
        sparse_config,
    )
    pipeline.sparse_history_archive = archive
    pipeline.sparse_history_modules = installed
    pipeline.sparse_history_config = sparse_config
    pipeline.sparse_history_completed_runs = []
    pipeline.sparse_history_aggregate_stats = SparseRunStats(method=sparse_config.method)

    original_inference = pipeline.inference

    def inference_with_reset(self, *inference_args, **inference_kwargs):
        self.sparse_history_archive.clear_frames()
        self.sparse_history_archive.stats = SparseRunStats(method=sparse_config.method)
        for module in self.sparse_history_modules:
            module.clear_selection_cache()
        result = original_inference(*inference_args, **inference_kwargs)
        self.sparse_history_aggregate_stats.merge(self.sparse_history_archive.stats)
        self.sparse_history_completed_runs.append(
            self.sparse_history_archive.stats.as_dict()
        )
        return result

    pipeline.inference = types.MethodType(inference_with_reset, pipeline)
    return pipeline
