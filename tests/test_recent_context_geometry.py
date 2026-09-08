from scripts.audit_recent_context_geometry import geometry


def test_real_RAG_eligibility_gap_is_not_all_recent_GPU_context():
    g=geometry(78,chunk=3,local=12,sink=1,memory=6,recent_exclude=5)
    assert g['exact_past']==[76,77]
    assert g['exact_local_including_current']==[76,77,78,79,80]
    assert g['coarse_eligible_before_current_forward'][-1]==61
    assert g['newly_evicted_but_not_in_precomputed_coarse_pool']==[67,68,69]
    assert g['GPU_resident_but_excluded_from_exact']==[70,71,72,73,74,75]
    assert g['guaranteed_excluded_past']==list(range(62,76))
