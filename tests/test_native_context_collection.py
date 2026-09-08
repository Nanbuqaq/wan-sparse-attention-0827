from copy import deepcopy

import pytest

from scripts.collect_native_context_study import validate_group
from scripts.run_native_context_study import ARMS


def reports():
    result={}
    for arm in ARMS:
        reset=arm.startswith('reset_');source=list(range(56,64)) if arm=='reset_away' else list(range(40,48))
        before=dict(global_end_index=84480,local_end_index=28160,pinned_start=7040,pinned_len=7040)
        result[arm]=dict(status='pass',latent_shape=[1,128,48,44,80],pixels={'frames':509},local_frames=32,sink_frames=8,
            attention_backend='native_FA2',fallback_allowed=False,native_KV_allocation_policy='CFG1_positive_only',
            native_positive_and_negative_KV_bytes=10380902400,fixed_native_adaln_recipe=dict(num_warps=16,num_stages=1),
            triton_version='3.2.0',scene_context_reset=reset,seed=20260913,cut_scenario='toy',noise_sha256='noise',
            pre_return_latent_sha256='prefix',runner_commit='code',assets_manifest_sha256='assets',gpu='GPU',
            episode_memory=dict(mode='none' if arm=='none' else 'raw_reveal',capture={'source_frames':source},
                K_positions_preserved_not_rebased=True,ledger=dict(CPU_archive_peak_bytes=0 if arm=='none' else 2595225600,
                    demand_H2D_payload_bytes=0 if arm=='none' else 2595225600),
                installation=dict(at_latent=96,cache_metadata_unchanged=not reset,
                    admission_plan=dict(source_frames=source,destination_token_range=[7040,14080]),
                    cache_metadata_transition=dict(before=before,after=dict(before,local_end_index=14080))),
                observed_return_attention_shapes=[dict(query_start_latent=f,Q=7040,K=21120 if f==96 else 28160,calls=150) for f in (96,104,112,120)]))
    return result


def test_valid_context_intervention_contract():
    assert validate_group(reports())['first_return_K_24_then_32']


@pytest.mark.parametrize('mutation',['prefix','window','allocation','clock','source','H2D','shape','calls','compiler','policy'])
def test_rejects_misleading_context_accounting(mutation):
    data=deepcopy(reports());r=data['reset_reveal'];memory=r['episode_memory']
    if mutation=='prefix':r['pre_return_latent_sha256']='other'
    if mutation=='window':r['local_frames']=24
    if mutation=='allocation':r['native_positive_and_negative_KV_bytes']*=2
    if mutation=='clock':memory['installation']['cache_metadata_transition']['after']['global_end_index']=14080
    if mutation=='source':memory['capture']['source_frames']=list(range(80,88))
    if mutation=='H2D':memory['ledger']['demand_H2D_payload_bytes']=0
    if mutation=='shape':memory['observed_return_attention_shapes'][0]['K']=28160
    if mutation=='calls':memory['observed_return_attention_shapes'][0]['calls']=30
    if mutation=='compiler':r['triton_version']='3.3.1'
    if mutation=='policy':r['scene_context_reset']=False
    with pytest.raises(ValueError):validate_group(data)
