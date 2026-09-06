from scripts.probe_verified_prefetch_routes import admit, byte_count, physical_metrics, lru_replay


def test_partial_frame_block_cost_is_not_padded_64():
    assert byte_count({(0, 1, 24)}) == 24*512
    assert byte_count({(0, 1, 0), (0, 1, 24)}) == 88*512


def test_prefetch_budget_is_causal_per_head_and_completion_keeps_target():
    source = [(h, f, b) for h in range(2) for f in (1, 99) for b in range(25)]
    prediction = admit(source, [1], 2)
    assert all(b[1] == 1 for b in prediction)
    assert byte_count(prediction) <= 2*390*512
    actual = {(0, 1, 0), (1, 1, 24)}
    metrics = physical_metrics(prediction, actual, 88*512)
    assert metrics['final_block_coverage_exact_actual_after_discarding_extras']
    assert metrics['total_prefetch_plus_completion_bytes'] == byte_count(prediction | actual)
    assert metrics['exposed_wait_s'] is None


def test_raw_cache_is_causal_and_respects_variable_tail_size():
    a, b = (0, 1, 0), (0, 1, 24)
    result = lru_replay([(0, {a}), (1, {a, b}), (2, {b})], 64*512)
    assert result['records'][0]['hit_bytes'] == 0
    assert result['records'][1]['hit_bytes'] == 64*512
    assert result['records'][2]['hit_bytes'] == 24*512
    assert result['peak_persistent_bytes'] <= 64*512
