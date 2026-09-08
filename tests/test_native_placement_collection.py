from copy import deepcopy

import pytest

from scripts.review_native_placement_gate import validate_group


def reports():
    result={}
    for arm in ('none','global','shot'):
        active=arm!='none'
        result[arm]=dict(status='pass',latent_shape=[1,64,48,16,32],pixels={'frames':253},
            local_frames=32,sink_frames=8,attention_backend='native_FA2',fallback_allowed=False,
            episode_memory_mode='raw_reveal' if active else 'none',seed=20260904,cut_scenario='toy',gpu='test',
            noise_sha256='noise',pre_return_latent_sha256='prefix',runner_commit='source',
            fixed_native_adaln_recipe=dict(num_warps=16,num_stages=1),
            episode_memory=dict(destination_role=arm,K_positions_preserved_not_rebased=True,
                capture=dict(source_frames=list(range(8,16))),
                installation=dict(at_latent=48,cache_metadata_unchanged=True,admission_plan_sha256=arm,
                    admission_plan=dict(frame_tokens=128,source_frames=list(range(8,16)),
                        destination_token_range=[1024,2048] if arm=='shot' else [0,1024])),
                ledger=dict(archive_D2H_payload_bytes=100 if active else 0,
                    demand_H2D_payload_bytes=100 if active else 0,CPU_archive_peak_bytes=100 if active else 0)))
    return result


def test_matched_destination_probe():
    assert validate_group(reports())['not_pure_layout_optimization']


@pytest.mark.parametrize('mutation', ['prefix','recipe','budget','source','sha','destination'])
def test_rejects_confounding_or_incomplete_contract(mutation):
    data=deepcopy(reports());shot=data['shot'];memory=shot['episode_memory']
    if mutation=='prefix':shot['pre_return_latent_sha256']='other'
    if mutation=='recipe':shot['fixed_native_adaln_recipe']['num_warps']=8
    if mutation=='budget':memory['ledger']['demand_H2D_payload_bytes']=101
    if mutation=='source':memory['capture']['source_frames']=list(range(16,24))
    if mutation=='sha':memory['installation']['admission_plan_sha256']='global'
    if mutation=='destination':memory['installation']['admission_plan']['destination_token_range']=[0,1024]
    with pytest.raises(ValueError):validate_group(data)
