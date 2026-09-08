from copy import deepcopy

import pytest

from scripts.run_native_capacity_control import validate_gate
from scripts.collect_native_capacity_control import validate_full_pair


def report(window=32,gate=False,positive=True):
    tokens=128 if gate else 880
    return dict(status='pass',latent_shape=[1,64,48,16,32] if gate else [1,128,48,44,80],
        pixels=dict(frames=253 if gate else 509,raw_RGB_sha256='rgb'),local_frames=window,
        triton_version='3.3.1',torch_dynamo_disabled=True,attention_backend='native_FA2',fallback_allowed=False,
        noise_sha256='noise',latent_sha256='latent',seed=1,cut_scenario='toy',gpu='GPU',runner_commit='source',
        assets_manifest_sha256='assets',episode_memory_mode=None,
        native_KV_allocation_policy='CFG1_positive_only' if positive else 'native_positive_and_negative',
        native_positive_and_negative_KV_bytes=window*tokens*24*128*2*2*30*(1 if positive else 2),
        fixed_native_adaln_recipe=dict(num_warps=16,num_stages=1))


def test_same_compiler_allocation_gate():
    assert validate_gate(report(gate=True,positive=False),report(gate=True))['same_compiler_native_positive_latent_RGB_exact']


def test_capacity_allows_own_trajectory_not_a_false_exactness_requirement():
    a,b=report(),report(128);b['latent_sha256']='different'
    assert validate_full_pair(a,b)['own_source_trajectory_required']


@pytest.mark.parametrize('mutation',['compiler','noise','shape','KV','policy','fallback'])
def test_rejects_cross_runtime_or_fake_capacity(mutation):
    a,b=report(),deepcopy(report(128))
    if mutation=='compiler':b['triton_version']='3.2.0'
    if mutation=='noise':b['noise_sha256']='other'
    if mutation=='shape':b['latent_shape']=[1,64,48,16,32]
    if mutation=='KV':b['native_positive_and_negative_KV_bytes']//=4
    if mutation=='policy':b['native_KV_allocation_policy']='native_positive_and_negative'
    if mutation=='fallback':b['fallback_allowed']=True
    with pytest.raises(ValueError):validate_full_pair(a,b)
