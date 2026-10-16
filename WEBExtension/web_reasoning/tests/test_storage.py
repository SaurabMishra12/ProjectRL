import json

def test_atomic_write_and_journal(tmp_path):
    from WEBExtension.web_reasoning.storage import atomic_json,Journal,digest
    p=tmp_path/'a.json';atomic_json(p,{'é':2});assert json.loads(p.read_text())=={'é':2}
    j=Journal(tmp_path/'events.jsonl');j.append({'event':'x'});j.append({'event':'y'})
    assert len(j.read())==2 and digest({'a':1,'b':2})==digest({'b':2,'a':1})

def test_unknown_object_cannot_silently_change_hash_identity():
    import pytest
    from WEBExtension.web_reasoning.storage import digest
    with pytest.raises(TypeError):digest({'object':object()})
