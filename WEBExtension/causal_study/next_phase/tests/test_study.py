import json
import pytest
from WEBExtension.causal_study.next_phase.binding_design import make_case,make_manifest
from WEBExtension.causal_study.next_phase.study import MockOnlyBackend,MockTransport,execute,execute_paid,parse_choice


def manifest():
    cases=[]
    for i in range(2):
        cases.append(make_case(str(i),'What city?',
            [{'answer':'Paris','citations':[['France',0]]},{'answer':'Lyon','citations':[['Lyon',0]]},{'answer':'noanswer','citations':[]}],
            [[{'title':'France','sentence_id':0,'text':'Paris is the capital.'}],
             [{'title':'Lyon','sentence_id':0,'text':'Lyon is a city.'}],[]]))
    return make_manifest(cases,n=2)


def config(m):
    return {'model':'frozen','model_revision':'alias','base_url':'https://example.invalid',
        'temperature':0.,'max_tokens':2048,'context_limit':32768,
        'max_completions':4,'max_attempts':6,'min_interval_seconds':0.,
        'max_estimated_tokens':2000000,'mock_only':True,'manifest_hash':m['manifest_sha256']}


def test_paid_action_is_unconditionally_forbidden_before_any_provider_access(tmp_path,monkeypatch):
    import WEBExtension.causal_study.provider as p
    monkeypatch.setattr(p,'azure_transport',lambda _:pytest.fail('credential/provider access forbidden'))
    with pytest.raises(PermissionError):execute_paid()
    assert list(tmp_path.iterdir())==[]


def test_mock_executes_two_arms_then_resume_spends_zero_new_requests(tmp_path):
    m=manifest();b=MockOnlyBackend(tmp_path/'budget',config(m))
    labels={'0':{'answer':'SECRET_GOLD'},'1':{'answer':'SECRET_GOLD'}}
    result=execute(m,labels,b,tmp_path/'checkpoints')
    assert result['status']=='COMPLETE_MOCK_ONLY'
    assert result['mock_http_attempts']==4 and result['real_provider_calls']==0
    assert result['scored_scientific_effects'] is False
    again=execute(m,labels,MockOnlyBackend(tmp_path/'budget',config(m)),tmp_path/'checkpoints')
    assert again['mock_http_attempts']==4
    for p in (tmp_path/'budget/requests').glob('*.json'):
        assert 'SECRET_GOLD' not in json.dumps(json.loads(p.read_text())['request'])


def test_changed_checkpoint_manifest_cannot_resume(tmp_path):
    m=manifest();b=MockOnlyBackend(tmp_path/'budget',config(m));execute(m,{},b,tmp_path/'checkpoints')
    checkpoint=next((tmp_path/'checkpoints').glob('*.json'))
    r=json.loads(checkpoint.read_text());r['manifest_hash']='different';checkpoint.write_text(json.dumps(r))
    with pytest.raises(ValueError):execute(m,{},b,tmp_path/'checkpoints')


@pytest.mark.parametrize('text',['{"choice":"C9"}','{"choice":0}','{"answer":"Paris"}','not JSON','{"choice":"C0","confidence":0.9}'])
def test_selector_rejects_new_answers_and_extra_fields(text):
    with pytest.raises(ValueError):parse_choice({'raw':{'choices':[{'finish_reason':'stop','message':{'content':text}}]}})


def test_truncated_selector_never_becomes_a_scored_answer():
    with pytest.raises(ValueError):parse_choice({'raw':{'choices':[{'finish_reason':'length','message':{'content':'{"choice":"C0"}'}}]}})


def test_mock_backend_does_not_accept_live_configuration(tmp_path):
    m=manifest();c=config(m);c['mock_only']=False
    with pytest.raises(PermissionError):MockOnlyBackend(tmp_path,c)
