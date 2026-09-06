from pathlib import Path
import pytest
from scripts.package_cross_trajectory_inputs import safe_relative
from scripts.build_aligned_seed_replication import build


def test_bundle_rejects_escape_paths():
    assert safe_relative('lane0/captures/test.pt')==Path('lane0/captures/test.pt')
    for bad in ('/etc/passwd','a/../../b'):
        with pytest.raises(ValueError):safe_relative(bad)


def test_independent_seed_matrix_is_nine_same_card_triplets():
    suite,expected=build('a'*40)
    assert len(expected['cases'])==9
    assert len({r['case_key_sha256'] for r in expected['cases']})==9
    assert {r['seed'] for r in expected['cases']}=={20260911,20260912}
    assert not suite['pooled_cross_category_mean_allowed']
    for lane in range(3):
        rows=[r for r in expected['cases'] if r['lane']==lane]
        assert len(rows)==3 and len({(r['prompt_id'],r['seed']) for r in rows})==1
    text=(Path(__file__).resolve().parents[1]/'scripts/inferhub_aligned_seed_replication_3gpu.sh').read_text()
    assert 'CUDA_VISIBLE_DEVICES=${assigned_gpus[$lane]}' in text
    assert '--shard-count 3' in text


def test_bootstrap_long_calibration_keeps_fresh_seeds_and_same_card_controls():
    suite,expected=build('c'*40,candidate='rope_bootstrap_ablation_history',seed_base=20260916,latent_frames=120)
    assert len(expected['cases'])==9
    assert {r['seed'] for r in expected['cases']}=={20260916,20260917}
    assert all(r['latent_frames']==120 for r in expected['cases'])
    assert suite['method_params']['rope_bootstrap_ablation_history']['bootstrap_layer']==-1
    assert 'rope_aligned_final_history' not in suite['methods']


def test_local_paired_cohort_is_disjoint_from_queued_h_cohort():
    suite,expected=build('d'*40,candidate='rope_bootstrap_ablation_history',seed_base=20260918,latent_frames=120,paired_only=True)
    assert len(expected['cases'])==6 and len(suite['cases'])==2
    assert {r['seed'] for r in expected['cases']}=={20260918}
    for lane in (0,1):
        assert len([r for r in expected['cases'] if r['lane']==lane])==3
