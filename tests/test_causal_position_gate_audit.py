from copy import deepcopy

import pytest

from scripts.audit_causal_original_gate import validate_memory_record


def record(policy):
    return dict(position_policy=policy,
        installations=[dict(at_latent=48,installation=dict(admission_plan=dict(
            source_frames=list(range(8,16)),temporal_delta=0 if policy=='original' else 48)))],
        archives=[dict(archive_version=2,source_end=16)],
        decisions=[dict(at_latent=48,selected_version=2)],
        ledger=dict(history_H2D_KV_bytes=377487360,archive_D2H_KV_bytes=1132462080,
            condition_summary_D2H_bytes=147456,CPU_archive_peak_tensor_bytes=1132511232,evicted_archives=0),
        raw_source_frames_or_target_frames_supplied_to_selector=False,
        future_text_or_generated_outputs_read_by_selector=False)


@pytest.mark.parametrize('policy',['original','recent_virtual'])
def test_registered_position_and_causal_inputs_checked(policy):
    memory=record(policy);reference=deepcopy(memory)
    assert validate_memory_record(memory,policy,reference)['source_frames']==list(range(8,16))
    memory['future_text_or_generated_outputs_read_by_selector']=True
    with pytest.raises(AssertionError):validate_memory_record(memory,policy,reference)


@pytest.mark.parametrize('key',['archives','decisions','ledger'])
def test_recent_reference_disagreement_blocks_expansion(key):
    memory=record('recent_virtual');reference=deepcopy(memory)
    if key=='ledger':reference[key]['history_H2D_KV_bytes']+=1
    else:reference[key]=[]
    with pytest.raises(AssertionError):validate_memory_record(memory,'recent_virtual',reference)
