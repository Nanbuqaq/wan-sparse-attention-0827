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
