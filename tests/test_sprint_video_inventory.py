import json

from scripts.audit_sprint_video_inventory import inspect_cohort


def test_failure_preserved_and_missing_not_renamed_success(tmp_path):
    case=tmp_path/'case';case.mkdir();(case/'terminal.json').write_text(json.dumps({'status':'fail'}))
    report=inspect_cohort(tmp_path,1)
    assert report['status']=='pass' and report['counts']=={'fail':1}
    assert inspect_cohort(tmp_path,2)['status']=='fail'
    assert inspect_cohort(tmp_path,2,allow_active=True)['status']=='running'


def test_success_requires_payload_ownership(tmp_path):
    case=tmp_path/'case';case.mkdir();(case/'terminal.json').write_text(json.dumps({'status':'pass'}))
    assert inspect_cohort(tmp_path,1)['status']=='fail'
    (case/'video.mp4').write_bytes(b'placeholder');(case/'latents.pt').write_bytes(b'placeholder')
    assert inspect_cohort(tmp_path,1,hash_payload=True)['rows'][0]['artifacts']['video']['sha256']
