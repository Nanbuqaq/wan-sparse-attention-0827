from scripts.export_mentor_perfetto import activity_bins,gpu_stage


def test_activity_is_union_fraction_not_sum_or_occupancy():
    assert activity_bins([(0,8),(2,5),(12,15)],0,20,10)==[(0,80.0),(10,30.0)]
    assert activity_bins([(-10,4),(8,30)],0,15,10)==[(0,60.0),(10,100.0)]


def test_attention_core_is_not_all_attention_module_work():
    assert gpu_stage('flash::flash_fwd_kernel','attention.complete_grouped_backend','generation.complete')=='history_and_exact_attention_core'
    assert gpu_stage('ampere_bf16_gemm','self_attention.q','generation.complete')=='self_attention.q'
    assert gpu_stage('flash::flash_fwd_kernel','transformer.cross_attn','generation.complete')=='transformer.cross_attn'
