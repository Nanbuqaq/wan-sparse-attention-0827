"""Construction of the pinned LongLive-RAG runtime with sparse history hooks."""

from __future__ import annotations

import types
from typing import Any

import torch

from .archive import HistoryArchive
from .config import SparseHistoryConfig
from .history_cache import HistoryUnionCache, RawHistoryBlockCache
from .runtime_attention import install_sparse_history_attention
from .stats import SparseRunStats
from .staging import PinnedStagingPool
from .system_config import LongLiveSystemConfig
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


def _system_config_from_args(args: Any) -> LongLiveSystemConfig:
    value = getattr(args, "longlive_system", None)
    if value is not None and hasattr(value, "items"):
        value = dict(value.items())
    return LongLiveSystemConfig.from_mapping(value)


def _build_history_union_cache(
    system_config: LongLiveSystemConfig,
) -> HistoryUnionCache | RawHistoryBlockCache | None:
    if system_config.gpu_union_cache == "off":
        return None
    if system_config.gpu_union_cache_budget_mib <= 0:
        raise ValueError("enabled gpu_union_cache requires a positive explicit budget")
    budget_bytes = system_config.gpu_union_cache_budget_mib * 1024 * 1024
    if system_config.gpu_union_cache == "cross_chunk":
        return RawHistoryBlockCache(budget_bytes)
    return HistoryUnionCache(budget_bytes)


def _build_staging_pool(
    system_config: LongLiveSystemConfig,
) -> PinnedStagingPool | None:
    if not system_config.staging_mode.startswith("persistent_"):
        return None
    if system_config.host_pinned_budget_mib <= 0:
        raise ValueError("persistent staging requires a positive host pinned budget")
    return PinnedStagingPool(
        slots=system_config.pinned_buffer_slots,
        budget_bytes=system_config.host_pinned_budget_mib * 1024 * 1024,
        pin_memory=torch.cuda.is_available(),
    )


def configure_pipeline_system(
    pipeline: Any, system_config: LongLiveSystemConfig
) -> HistoryUnionCache | RawHistoryBlockCache | None:
    """Apply one frozen system configuration to an already loaded pipeline."""

    history_union_cache = _build_history_union_cache(system_config)
    staging_pool = _build_staging_pool(system_config)
    pipeline.longlive_system_config = system_config
    pipeline.history_union_cache = history_union_cache
    pipeline.history_staging_pool = staging_pool
    for module in pipeline.sparse_history_modules:
        module.system_config = system_config
        module.history_union_cache = history_union_cache
        module.history_staging_pool = staging_pool
    return history_union_cache


def build_sparse_pipeline(args: Any, device: torch.device | str):
    """Instantiate the read-only RAG pipeline and replace only self-attention."""

    if not torch.cuda.is_available():
        raise RuntimeError("LongLive sparse pipeline requires a CUDA-enabled process")
    configure_upstream_paths()
    import utils.wan_wrapper as base_wrapper

    latentmem_module = load_latentmem_module()
    rag_pipeline_module = load_rag_pipeline_module()
    sparse_config = _sparse_config_from_args(args)
    system_config = _system_config_from_args(args)

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

        def forward(
            self,
            noisy_image_or_video,
            conditional_dict,
            timestep,
            kv_cache=None,
            crossattn_cache=None,
            current_start=None,
            classify_mode=False,
            concat_time_embeddings=False,
            clean_x=None,
            aug_t=None,
            cache_start=None,
            sink_recache_after_switch=False,
            memory_indices=None,
        ):
            prompt_embeds = conditional_dict["prompt_embeds"]
            input_timestep = timestep[:, 0] if self.uniform_timestep else timestep
            logits = None
            if kv_cache is not None:
                kwargs = {
                    "t": input_timestep,
                    "context": prompt_embeds,
                    "seq_len": self.seq_len,
                    "kv_cache": kv_cache,
                    "crossattn_cache": crossattn_cache,
                    "current_start": current_start,
                    "cache_start": cache_start,
                    "sink_recache_after_switch": sink_recache_after_switch,
                    "memory_indices": memory_indices,
                }
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4), **kwargs
                ).permute(0, 2, 1, 3, 4)
            elif clean_x is not None:
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                    sink_recache_after_switch=sink_recache_after_switch,
                ).permute(0, 2, 1, 3, 4)
            elif classify_mode:
                flow_pred, logits = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=self.seq_len,
                    classify_mode=True,
                    register_tokens=self._register_tokens,
                    cls_pred_branch=self._cls_pred_branch,
                    gan_ca_blocks=self._gan_ca_blocks,
                    concat_time_embeddings=concat_time_embeddings,
                )
                flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
            else:
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=self.seq_len,
                ).permute(0, 2, 1, 3, 4)
            pred_x0 = self._convert_flow_pred_to_x0(
                flow_pred=flow_pred.flatten(0, 1),
                xt=noisy_image_or_video.flatten(0, 1),
                timestep=timestep.flatten(0, 1),
            ).unflatten(0, flow_pred.shape[:2])
            if logits is not None:
                return flow_pred, pred_x0, logits
            return flow_pred, pred_x0

    rag_pipeline_module.WanDiffusionWrapper = SparseWanDiffusionWrapper
    rag_pipeline_module.WanTextEncoder = base_wrapper.WanTextEncoder
    rag_pipeline_module.WanVAEWrapper = base_wrapper.WanVAEWrapper
    pipeline = rag_pipeline_module.CausalInferencePipeline(args, device=torch.device(device))
    archive = HistoryArchive(sparse_config, spatial_height=30, spatial_width=52)
    history_union_cache = _build_history_union_cache(system_config)
    history_staging_pool = _build_staging_pool(system_config)
    installed = install_sparse_history_attention(
        pipeline.generator.model,
        archive,
        sparse_config,
        system_config=system_config,
        history_union_cache=history_union_cache,
        history_staging_pool=history_staging_pool,
    )
    pipeline.sparse_history_archive = archive
    pipeline.sparse_history_modules = installed
    pipeline.sparse_history_config = sparse_config
    pipeline.longlive_system_config = system_config
    pipeline.history_union_cache = history_union_cache
    pipeline.history_staging_pool = history_staging_pool
    pipeline.sparse_history_completed_runs = []
    pipeline.sparse_history_aggregate_stats = SparseRunStats(method=sparse_config.method)

    original_inference = pipeline.inference

    def inference_with_reset(self, *inference_args, **inference_kwargs):
        self.sparse_history_archive.clear_frames()
        current_method = self.sparse_history_config.method
        self.sparse_history_archive.stats = SparseRunStats(method=current_method)
        for module in self.sparse_history_modules:
            module.clear_selection_cache()
        if self.history_union_cache is not None:
            self.history_union_cache.reset()
        result = original_inference(*inference_args, **inference_kwargs)
        if self.sparse_history_aggregate_stats.method != current_method:
            self.sparse_history_aggregate_stats = SparseRunStats(method=current_method)
        self.sparse_history_aggregate_stats.merge(self.sparse_history_archive.stats)
        self.sparse_history_completed_runs.append(
            self.sparse_history_archive.stats.as_dict()
        )
        return result

    pipeline.inference = types.MethodType(inference_with_reset, pipeline)
    return pipeline
