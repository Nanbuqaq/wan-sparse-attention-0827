from scripts.audit_bootstrap_long_calibration import fidelity_checks


def test_long_gate_does_not_average_away_late_regression():
    old = {'lpips_mean': .2, 'late_quarter_lpips_mean': .3, 'latent_error': {'relative_l2': .4}}
    new = {'lpips_mean': .1, 'late_quarter_lpips_mean': .31, 'latent_error': {'relative_l2': .3}}
    assert fidelity_checks(old, new) == {'full_lpips': True, 'late_lpips': False, 'latent_l2': True}
    assert all(fidelity_checks(old, old).values())
