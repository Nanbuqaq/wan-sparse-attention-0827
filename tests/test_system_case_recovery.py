import json
import pytest
from scripts.recover_system_case_states import recover


def test_killed_batch_preserves_pass_and_marks_only_unfinished(tmp_path):
    expected={'cases':[{'id':str(i),'case_key_sha256':str(i),'case_key':{'i':i}} for i in range(2)]}
    case=tmp_path/'done'
    case.mkdir()
    (case/'case_state.json').write_text(json.dumps({**expected['cases'][0],'status':'pass'}))
    output=recover(tmp_path,expected,{'job_id':'job','result':{'reason':'infer_gpu_idle'}},[])
    assert [case['status'] for case in output['cases']]==['pass','fail']
    assert output['recovered_case_artifacts']==1
    assert output['missing_terminal_states']==0
    with pytest.raises(ValueError,match='without a terminal'):
        recover(tmp_path,expected,{'job_id':'job'},[])
