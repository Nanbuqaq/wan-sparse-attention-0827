from scripts.analyze_native_return_context import simulate


def test_first_return_still_has_two_away_blocks_after_global_recall():
    call=simulate('global')['calls'][12]
    assert call['visible_frame_ids']==list(range(40,48))+list(range(48,56))+list(range(88,104))
    assert call['role_latent_counts']=={'reveal':8,'away':16,'return':8}


def test_shot_keeps_original_global_and_halves_away_context():
    call=simulate('shot')['calls'][12]
    assert call['visible_frame_ids']==list(range(8))+list(range(40,48))+list(range(88,104))
    assert call['role_latent_counts']=={'initial':8,'reveal':8,'away':8,'return':8}


def test_expiry_does_not_discard_already_committed_return():
    global_rows=simulate('global')['calls'];ttl=simulate('global_one_chunk')['calls']
    assert global_rows[:13]==ttl[:13]
    assert ttl[13]['visible_frame_ids']==list(range(8))+list(range(88,112))
    assert ttl[15]['visible_frame_ids']==list(range(8))+list(range(96,104))+list(range(112,128))


def test_large_window_retains_all_source_and_away_context():
    call=simulate('window128')['calls'][12]
    assert call['visible_frame_ids']==list(range(104))
    assert call['role_latent_counts']=={'initial':24,'reveal':24,'away':48,'return':8}
