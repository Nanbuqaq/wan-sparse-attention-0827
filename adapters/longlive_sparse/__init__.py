"""Training-free sparse history retrieval for LongLive and LongLive-RAG."""

from .archive import HistoryArchive, MaterializedHistory
from .attention_bias import AttentionBiasPlan
from .config import SparseHistoryConfig
from .contexts import OfflineTeacherContext, OnlineRoutingContext
from .cost_model import HardwareCostProfile, SystemCostModel
from .history_cache import (
    CachedRawHistoryBlock,
    HistoryKVCacheKey,
    HistoryUnionCache,
    RawHistoryBlockCache,
    RawHistoryBlockCacheKey,
)
from .memory_roles import build_three_role_probabilities
from .methods import METHOD_SPECS, MethodSpec, method_spec
from .novelty import causal_prototype_novelty
from .offline_eval import (
    dense_history_attention,
    output_error_metrics,
    routed_history_attention,
)
from .prefetch import VerifiedPrefetchPlan, build_verified_prefetch_plan
from .profiling import DeferredCudaEventCollector, classify_bottleneck
from .route_plan import HistoryRoutePlan
from .reuse import RouteReuseTracker
from .sensitivity import history_head_sensitivity
from .selectors import SparseSelection
from .stats import SparseRunStats
from .staging import PinnedStagingPool
from .system_config import LongLiveSystemConfig
from .system_utility_route import (
    SystemUtilityRouteConfig,
    build_cost_model_set_cost_factory,
    build_system_utility_route,
)
from .system_trace import SystemTraceRecord
from .timeline import TimelineInterval
from .tethermem import soft_region_age_prior, solve_context_weight
from .transfer_plan import (
    TransferExecutionPlan,
    TransferPlan,
    TransferRun,
    build_transfer_execution_plan,
    build_transfer_plan,
)
from .utility import OnlineUtilityProxy, compute_online_utility_proxy
from .utility import apply_query_group_policy

__all__ = [
    "AttentionBiasPlan",
    "HardwareCostProfile",
    "HistoryArchive",
    "HistoryKVCacheKey",
    "HistoryRoutePlan",
    "HistoryUnionCache",
    "RawHistoryBlockCache",
    "RawHistoryBlockCacheKey",
    "CachedRawHistoryBlock",
    "history_head_sensitivity",
    "LongLiveSystemConfig",
    "METHOD_SPECS",
    "MaterializedHistory",
    "MethodSpec",
    "OfflineTeacherContext",
    "OnlineUtilityProxy",
    "OnlineRoutingContext",
    "PinnedStagingPool",
    "DeferredCudaEventCollector",
    "RouteReuseTracker",
    "SparseHistoryConfig",
    "SparseRunStats",
    "SparseSelection",
    "SystemCostModel",
    "SystemUtilityRouteConfig",
    "SystemTraceRecord",
    "TimelineInterval",
    "TransferPlan",
    "TransferRun",
    "TransferExecutionPlan",
    "VerifiedPrefetchPlan",
    "build_verified_prefetch_plan",
    "apply_query_group_policy",
    "classify_bottleneck",
    "build_transfer_plan",
    "build_transfer_execution_plan",
    "build_three_role_probabilities",
    "build_cost_model_set_cost_factory",
    "build_system_utility_route",
    "compute_online_utility_proxy",
    "dense_history_attention",
    "causal_prototype_novelty",
    "method_spec",
    "output_error_metrics",
    "routed_history_attention",
    "soft_region_age_prior",
    "solve_context_weight",
]
