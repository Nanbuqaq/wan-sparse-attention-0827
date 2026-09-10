from scripts.analyze_native_stage_profile import scope_index,launch_category


def test_nested_launch_scope_wins_over_module_and_uses_launch_not_execution():
    scopes=[(0,100,'native/self_attn/layer0',7),(20,40,'native/self_attention_core',7)]
    idx=scope_index(scopes)
    assert launch_category(idx,7,25,28)=='self_attention_wrapper'
    assert launch_category(idx,7,45,48)=='self_QKV_norm_RoPE_window_output_projection'
    assert launch_category(idx,8,25,28)=='unattributed_or_other_model_work'
    assert launch_category(idx,7,101,102)=='unattributed_or_other_model_work'
