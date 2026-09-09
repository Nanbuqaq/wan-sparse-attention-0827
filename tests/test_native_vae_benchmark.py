from scripts.benchmark_native_vae_layout import paired_order


def test_fixed_thirty_rounds_have_each_mode_once_per_pair():
    order=paired_order()
    assert len(order)==30 and order==paired_order()
    assert all(sorted(row)==['baseline','weights_only'] for row in order)
    assert {tuple(row) for row in order}=={('baseline','weights_only'),('weights_only','baseline')}
