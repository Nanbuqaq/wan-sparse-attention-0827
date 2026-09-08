from copy import deepcopy

import pytest

from scripts.collect_native_memory_study import validate_group
from scripts.run_native_memory_study import ARMS


def reports():
    result={}
    for arm in ARMS:
        active=arm in ('global','global_one_chunk','shot');ttl=arm=='global_one_chunk'
        record=dict(status='pass',latent_shape=[1,64,48,16,32],pixels={'frames':253},sink_frames=8,
            attention_backend='native_FA2',fallback_allowed=False,local_frames=128 if arm=='window128' else 32,
            fixed_native_adaln_recipe=dict(num_warps=16,num_stages=1),
            upstream_source_SHA='6b36d20ec6f7958d29d11a704dfa64611a9f2572',
            strict_generator_load=dict(missing_keys=[],unexpected_keys=[]),
            seed=20260904,cut_scenario='toy',noise_sha256='noise',runner_commit='source',assets_manifest_sha256='assets',
            gpu='test',pre_return_latent_sha256='own' if arm=='window128' else 'prefix',first_return_latent_sha256='first',
            episode_memory_mode=None if arm=='window128' else ('raw_reveal' if active else 'none'),
            native_positive_and_negative_KV_bytes=128*128*24*128*2*2*30*2)
        if arm!='window128':
            record['episode_memory']=dict(
                capture=dict(source_frames=list(range(8,16))),K_positions_preserved_not_rebased=True,
                installation=dict(at_latent=48,cache_metadata_unchanged=True,admission_plan_sha256=arm,
                    admission_plan=dict(destination_token_range=[1024,2048] if arm=='shot' else [0,1024])),
                restore_after_frames=8 if ttl else 0,
                restoration=dict(at_latent=56,original_pre_return_global_restored=True,cache_metadata_unchanged=True) if ttl else None,
                ledger=dict(CPU_archive_peak_bytes=128 if ttl else (64 if active else 0),
                    archive_D2H_payload_bytes=64 if active else 0,demand_H2D_payload_bytes=64 if active else 0,
                    rollback_D2H_payload_bytes=64 if ttl else 0,restore_H2D_payload_bytes=64 if ttl else 0))
        result[arm]=record
    return result


def test_accepts_own_large_window_trajectory_but_exact_preexpiry():
    assert validate_group(reports(),gate=True)['global_vs_ttl_first_return_exact']


@pytest.mark.parametrize('mutation',['first_return','prefix','unpaid_backup','unpaid_restore','unpaid_peak','no_restore',
                                    'source','position','recipe','allocation','capacity','window_hook','sha'])
def test_rejects_hidden_confounding_or_cost(mutation):
    data=deepcopy(reports());ttl=data['global_one_chunk'];memory=ttl['episode_memory']
    if mutation=='first_return':ttl['first_return_latent_sha256']='different'
    if mutation=='prefix':data['shot']['pre_return_latent_sha256']='different'
    if mutation=='unpaid_backup':memory['ledger']['rollback_D2H_payload_bytes']=0
    if mutation=='unpaid_restore':memory['ledger']['restore_H2D_payload_bytes']=0
    if mutation=='unpaid_peak':memory['ledger']['CPU_archive_peak_bytes']=64
    if mutation=='no_restore':memory['restore_after_frames']=0
    if mutation=='source':memory['capture']['source_frames']=list(range(16,24))
    if mutation=='position':memory['K_positions_preserved_not_rebased']=False
    if mutation=='recipe':ttl['fixed_native_adaln_recipe']['num_warps']=8
    if mutation=='allocation':data['window128']['native_positive_and_negative_KV_bytes']//=4
    if mutation=='capacity':data['window128']['local_frames']=32
    if mutation=='window_hook':data['window128']['episode_memory_mode']='none'
    if mutation=='sha':memory['installation']['admission_plan_sha256']='global'
    with pytest.raises(ValueError):validate_group(data,gate=True)
