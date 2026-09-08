import copy
import pytest
from scripts.collect_event_retrieval import validate


def report(control=None):
    names=['scheduled_Dense','event_contrast'] if control else ['scheduled_Dense','event_cosine','event_contrast']
    rows=[]
    for name in names:
        rows.append(dict(variant=name,status='pass',history_H2D_bytes=123,prefix_before_anchor_bitwise_equal=True,
            event_retrieval=None if name=='scheduled_Dense' else dict(past_episode_keys_only=True,
                workload_role_labels_used=False,predefined_anchor_interval_used=False,first_override_latent=48)))
    return dict(status='pass',mode='event_probe',seed=20260909,latent_frames=120,novel_control=control,
                variants=rows,gpu='test',source_commit='a'*40)


def test_original_and_negative_matrix_scopes():
    assert validate(report(),20260909,None)['cases']==3
    assert validate(report('duck'),20260909,'duck')['cases']==2


@pytest.mark.parametrize('defect',['bytes','prefix','role'])
def test_changed_budget_or_hidden_supervision_rejected(defect):
    p=copy.deepcopy(report());row=p['variants'][-1]
    if defect=='bytes':row['history_H2D_bytes']=124
    elif defect=='prefix':row['prefix_before_anchor_bitwise_equal']=False
    else:row['event_retrieval']['workload_role_labels_used']=True
    with pytest.raises(ValueError):validate(p,20260909,None)
