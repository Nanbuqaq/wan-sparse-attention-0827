from copy import deepcopy

import pytest

from scripts.collect_native_episode_batch import validate_group,MODES


def test_episode_storage_gate_requires_same_prefix_and_complete_output():
    reports={}
    for mode in MODES:
        enabled=mode!='none';log=mode=='log_reveal'
        reports[mode]=dict(status='pass',episode_memory_mode=mode,latent_shape=[1,64,48,16,32],local_frames=32,sink_frames=8,
            attention_backend='native_FA2',fallback_allowed=False,pixels={'frames':253,'raw_RGB_sha256':'pixels'},
            seed=1,cut_scenario='toy',gpu='test',noise_sha256='noise',pre_return_latent_sha256='prefix',latent_sha256='latent',
            episode_memory=dict(capture={'source_frames':list(range(24,32)) if mode=='raw_away' else list(range(8,16))},
                installation={'admission_plan_sha256':'correct','cache_metadata_unchanged':True},
                ledger={'demand_H2D_payload_bytes':10 if log else (100 if enabled else 0),
                        'CPU_archive_peak_bytes':10 if log else (100 if enabled else 0)}))
    assert validate_group(reports,gate=True)['raw_log_full_latent_RGB_exact']
    bad=deepcopy(reports);bad['log_reveal']['latent_sha256']='different'
    with pytest.raises(ValueError):validate_group(bad,gate=True)
