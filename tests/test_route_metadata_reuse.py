import copy

import pytest
import torch

from adapters.longlive_sparse.ar_routing import build_route_plan
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.route_metadata import RouteIdentityCache
from adapters.longlive_sparse.system_config import LongLiveSystemConfig


def plan_fixture():
    return build_route_plan(method='test', routing_stage='pre-transfer',
        query_labels=torch.zeros(1, 2, 8, dtype=torch.long),
        selections=[[[torch.arange(5)] for _ in range(2)]],
        history_frame_ids=torch.zeros(1, 2, 7, dtype=torch.long),
        history_token_ids=torch.arange(7).view(1, 1, -1).expand(1, 2, -1),
        candidate_history_tokens=7, exact_k_tokens=3, density=1., metadata={})


@pytest.mark.parametrize('inference', [False, True])
def test_digest_matches_recomputation_and_detects_numpy_alias(inference):
    with torch.inference_mode(inference):
        plan = plan_fixture()
        expected = plan.digest()
        plan.enable_verified_digest_reuse()
        assert plan.digest() == plan.digest() == expected
        assert plan.digest_snapshot_bytes > 0
        # This does NOT reliably change a Torch version counter.
        plan.union_token_ids.numpy()[0, 0, 0] = 6
        new_digest = plan.digest()
        plan.disable_verified_digest_reuse()
        assert new_digest == plan.digest() != expected
        assert plan.digest_snapshot_bytes == 0


@pytest.mark.parametrize('field', ['method', 'routing_stage', 'routing_identity', 'dtype', 'tensor_replace'])
def test_digest_invalidates_all_semantic_inputs(field):
    plan = plan_fixture()
    plan.enable_verified_digest_reuse()
    old = plan.digest()
    if field in ('method', 'routing_stage'):
        setattr(plan, field, 'changed')
    elif field == 'routing_identity':
        plan.metadata['routing_identity'] = {'role': ['subject']}
    elif field == 'dtype':
        plan.query_labels = plan.query_labels.to(torch.int32)
    else:
        plan.group_history_counts = plan.group_history_counts + 1
    new = plan.digest()
    plan.disable_verified_digest_reuse()
    assert new == plan.digest() != old


def test_position_key_and_single_plan_retention():
    plan, cache = plan_fixture(), RouteIdentityCache()
    kwargs = dict(current_frame_id=10, spatial_width=7, rope_policy='clipped_relative_age',
                  max_relative_age=100, candidate_frame_ids=(0,))
    first = cache.prepare(plan, **kwargs)
    assert cache.prepare(plan, **kwargs) is first
    assert first.position_sha256 == tensor_sha256(first.positions)
    newer = cache.prepare(plan, **(kwargs | {'current_frame_id': 11}))
    assert newer.position_sha256 != first.position_sha256
    plan.union_token_ids[0, 0, 0] = 6
    edited = cache.prepare(plan, **kwargs)
    assert edited.coordinate_sha256 != first.coordinate_sha256
    other = copy.deepcopy(plan)
    cache.prepare(other, **kwargs)
    assert plan.digest_snapshot_bytes == 0
    assert other.digest_snapshot_bytes > 0 and cache.retained_CPU_bytes > 0
    cache.clear()
    assert other.digest_snapshot_bytes == 0 and cache.retained_CPU_bytes == 0


def test_configuration_is_explicit_and_identity_distinguishes_reuse():
    old = LongLiveSystemConfig()
    new = LongLiveSystemConfig(route_metadata_mode='validated_reuse')
    assert old.identity_dict() != new.identity_dict()
    assert LongLiveSystemConfig.from_mapping(new.as_dict()) == new
    with pytest.raises(ValueError):
        LongLiveSystemConfig(route_metadata_mode='unchecked_identity_only')
