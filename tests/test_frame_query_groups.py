import pytest
import torch

from adapters.longlive_sparse.frame_query_groups import (
    choose_frames, spatial_query_geometry, whole_frame_prototypes,
)


def test_real_5b_partition_and_inverse_cover_every_query_once():
    groups, sites, samples = spatial_query_geometry(8, 22, 40)
    assert groups.shape == (4, 1760)
    assert torch.equal(groups.flatten().sort().values, torch.arange(7040))
    assert samples.shape == (4, 32)
    for g in range(4):
        assert torch.isin(sites[samples[g]], groups[g]).all()
        assert set((groups[g]//880).tolist()) == set(range(8))
    inverse = groups.flatten().argsort()
    assert torch.equal(groups.flatten()[inverse], torch.arange(7040))


def test_weighted_frame_prototype_does_not_overweight_the_48_token_tail():
    torch.manual_seed(1)
    raw = torch.randn(880, 2, 8)
    chunks = raw.split(64)
    means = torch.stack([x.mean(0) for x in chunks])
    count = torch.tensor([len(x) for x in chunks])
    k, v = whole_frame_prototypes([(means, means*2, count)], 880)
    torch.testing.assert_close(k[0], raw.mean(0))
    torch.testing.assert_close(v[0], raw.mean(0)*2)


def test_group_specific_routes_follow_different_current_queries_and_shared_control_is_equal():
    groups, sites, samples = spatial_query_geometry(1, 4, 4)
    q = torch.zeros(16, 2, 4)
    for g in range(4): q[groups[g], :, g] = 10
    k = torch.eye(4)[:, None].expand(-1, 2, -1)
    v = torch.ones_like(k)
    independent, shared = choose_frames(q, k, v, sites, samples, 1, 'split_specific')
    assert independent[:, :, 0].tolist() == [[g, g] for g in range(4)]
    split, shared2 = choose_frames(q, k, v, sites, samples, 1, 'split_shared')
    whole, shared3 = choose_frames(q, k, v, sites, samples, 1, 'shared')
    assert torch.equal(shared, shared2) and torch.equal(shared, shared3)
    assert torch.equal(split, whole.expand(4, -1, -1))
    _, tied = choose_frames(torch.zeros_like(q), k, v, sites, samples, 1, 'shared')
    assert tied.tolist() == [[0], [0]]  # Exact ties use the first history frame.


def test_unregistered_geometry_and_policy_are_rejected():
    with pytest.raises(ValueError): spatial_query_geometry(8, 21, 40)
    with pytest.raises(ValueError):
        choose_frames(torch.zeros(16,1,4),torch.zeros(4,1,4),torch.zeros(4,1,4),
                      torch.arange(16),torch.arange(16).reshape(4,4),1,'unknown')


def test_owner_version_change_invalidates_prototypes_without_rebuilding_geometry():
    from adapters.longlive_sparse.frame_query_groups import FrameQueryRouter
    router=FrameQueryRouter('shared',(4,4))
    q=torch.zeros(1,16,1,4,dtype=torch.bfloat16)
    k=torch.zeros(1,64,1,4,dtype=torch.bfloat16)
    summaries=[(torch.ones(1,1,4),torch.ones(1,1,4),torch.tensor([16])) for _ in range(2)]
    first=router.prepare(q,k,[0,1],[2,3],summaries,16,prototype_key=(0,('owner_v1',)))
    router.prepare(q,k,[0,1],[2,3],summaries,16,prototype_key=(0,('owner_v1',)))
    assert router.prototype_builds==1 and router.prototype_hits==1
    changed=[(x[0]*2,x[1],x[2]) for x in summaries]
    router.prepare(q,k,[0,1],[2,3],changed,16,prototype_key=(0,('owner_v2',)))
    assert router.prototype_builds==2 and router.builds==1 and router.hits==2
    assert torch.equal(router.prototype_cache[0][1],torch.full((2,1,4),2.))
    assert first['frame_ids'].shape==(1,1,3)


def test_closed_cohort_keeps_all_five_controls_on_the_same_task_pair(tmp_path):
    import json
    from pathlib import Path
    from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    cases=build_wave2_cases(spec,'query_groups',tmp_path,tmp_path,tmp_path,20261010)
    assert len(cases)==10
    lanes=case_lane_indices(cases,2)
    assert list(map(len,lanes))==[5,5]
    for indices in lanes:
        assert len({cases[i]['scenario'] for i in indices})==1
        assert {cases[i]['method'] for i in indices}=={'native','old_block64_fast','shared','split_shared','split_specific'}
