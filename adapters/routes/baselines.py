"""Baseline route entry points.

The implementation currently delegates to the reviewed unified router.  Each
public entry point is kept separate so task dependency hashes remain scoped to
the selected method while the implementation is migrated incrementally.
"""

from __future__ import annotations


def _legacy(*args, **kwargs):
    from adapters.routing import _route_attention_legacy

    return _route_attention_legacy(*args, **kwargs)


def route_original_block(*args, **kwargs):
    return _legacy(*args, **kwargs)


def route_random_block(*args, **kwargs):
    from adapters.routing import _route_random_block

    return _route_random_block(*args, **kwargs)


def route_local_3d(*args, **kwargs):
    from adapters.routing import _route_local_3d

    return _route_local_3d(*args, **kwargs)


def route_fixed_k(*args, **kwargs):
    return _legacy(*args, **kwargs)


def route_qsort_local8(*args, **kwargs):
    from adapters.routing import _route_qsort_local8

    return _route_qsort_local8(*args, **kwargs)


def route_token_oracle(*args, **kwargs):
    return _legacy(*args, **kwargs)
