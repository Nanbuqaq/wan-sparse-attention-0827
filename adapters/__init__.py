"""Wan short-video sparse-attention adapters."""

from .types import MethodConfig, RoutePlan, SparseRunStats
from .wan_sparse import install_sparse_processors

__all__ = ["MethodConfig", "RoutePlan", "SparseRunStats", "install_sparse_processors"]
