import subprocess
import sys
from pathlib import Path


def test_bad_metric_weight_path_fails_before_child_or_output_creation(tmp_path):
    root=Path(__file__).resolve().parents[1]
    out=tmp_path/'never_started'
    result=subprocess.run([sys.executable,str(root/'scripts/evaluate_canonical_batch.py'),
        '--states',str(tmp_path/'states.json'),'--expected',str(tmp_path/'expected.json'),
        '--output',str(out),'--linear-weights',str(tmp_path/'missing_linear.pt'),
        '--trunk-weights',str(tmp_path/'missing_trunk.pt'),'--lane','0'],capture_output=True,text=True)
    assert result.returncode != 0
    assert 'missing_linear.pt' in result.stderr
    assert not out.exists()
