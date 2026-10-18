import importlib
import json
import pytest
from .test_e9_design import candidate

MOD='WEBExtension.evidence_value_study.experiments.runtime'

def runtime():
    assert importlib.util.find_spec(MOD) is not None, 'E9 durable mock runtime missing'
    return importlib.import_module(MOD)

def config():
    return {'model':'EXPLICIT_TEST_MODEL','temperature':0.,'max_tokens':384,'max_completions':330,
            'max_attempts':390,'max_input_tokens':300000,'max_output_tokens':140000,
            'context_limit':32768,'max_retries_per_slot':2,'mode':'MOCK'}

def transport(request, attempt):
    return 200, {'x-request-id':attempt}, {'id':attempt,'model':'EXPLICIT_TEST_MODEL',
      'choices':[{'message':{'content':json.dumps({'answer':'','evidence':[],'uncertain':True})},'finish_reason':'stop'}],
      'usage':{'prompt_tokens':12,'completion_tokens':8}}

def test_30_question_dry_run_330_slots_and_resume_no_requests(tmp_path):
    r=runtime(); b=r.DurableMockBackend(tmp_path,config(),transport=transport,token_count=lambda s:len(s)//5)
    for i in range(30):r.run_question(candidate(i),i,b)
    assert b.summary()['successful_completions']==b.summary()['http_attempts']==330
    for i in range(30):r.run_question(candidate(i),i,b)
    assert b.summary()['http_attempts']==330
    assert b.summary()['real_provider_calls']==0

def test_config_mutation_and_payload_identity_rejected(tmp_path):
    r=runtime(); b=r.DurableMockBackend(tmp_path,config(),transport=transport,token_count=len)
    b.complete('slot',[{'role':'user','content':'one'}])
    with pytest.raises(ValueError):b.complete('slot',[{'role':'user','content':'two'}])
    with pytest.raises(ValueError):r.DurableMockBackend(tmp_path,{**config(),'max_tokens':385},transport=transport,token_count=len)

def test_unknown_attempt_halts_across_restart(tmp_path):
    r=runtime()
    def fail(*args):raise TimeoutError('private detail must not log')
    b=r.DurableMockBackend(tmp_path,config(),transport=fail,token_count=len)
    with pytest.raises(r.UncertainRequest):b.complete('s',[{'role':'user','content':'x'}])
    with pytest.raises(r.UncertainRequest):r.DurableMockBackend(tmp_path,config(),transport=transport,token_count=len)
    assert 'private detail' not in (tmp_path/'ledger.jsonl').read_text()

def test_explicit_429_bounded_retry_and_usage(tmp_path):
    r=runtime(); calls=[]
    def reject_once(request, attempt):
        calls.append(attempt)
        return (429,{}, {'error':{'code':'rate_limit_exceeded'}}) if len(calls)==1 else transport(request,attempt)
    b=r.DurableMockBackend(tmp_path,config(),transport=reject_once,token_count=len)
    b.complete('s',[{'role':'user','content':'x'}])
    assert b.summary()['http_attempts']==2 and b.summary()['successful_completions']==1
    assert b.summary()['retries']==1

def test_token_reservations_and_successes_persist(tmp_path):
    r=runtime(); c={**config(),'max_input_tokens':260}
    b=r.DurableMockBackend(tmp_path,c,transport=transport,token_count=len)
    b.complete('s',[{'role':'user','content':'x'}])
    # Each dispatch conservatively reserves129 input tokens, irrespective of synthetic usage.
    b=r.DurableMockBackend(tmp_path,c,transport=transport,token_count=len)
    b.complete('t',[{'role':'user','content':'x'}])
    with pytest.raises(r.BudgetExceeded):b.complete('u',[{'role':'user','content':'x'}])
    assert b.summary()['successful_completions']==2

def test_malformed_success_not_retried_and_unknown_usage_not_zero(tmp_path):
    r=runtime()
    def malformed(request,attempt):return 200,{}, {'choices':[]}
    b=r.DurableMockBackend(tmp_path,config(),transport=malformed,token_count=len)
    b.complete('s',[{'role':'user','content':'x'}]); b.complete('s',[{'role':'user','content':'x'}])
    assert b.summary()['http_attempts']==1
    assert b.summary()['input_tokens'] is None
    assert b.summary()['missing_usage_records']==1

def test_real_provider_path_unavailable_without_annotations():
    r=runtime()
    with pytest.raises(r.ScientificGateClosed,match='independent human'):
        r.execute_paid()

def test_crash_after_success_ledger_before_response_cache_does_not_duplicate(tmp_path):
    r=runtime(); b=r.DurableMockBackend(tmp_path,config(),transport=transport,token_count=len)
    saved=b.complete('s',[{'role':'user','content':'x'}])
    (tmp_path/'responses'/(saved['slot_hash']+'.json')).unlink()
    b=r.DurableMockBackend(tmp_path,config(),transport=transport,token_count=len)
    b.complete('s',[{'role':'user','content':'x'}])
    assert b.summary()['http_attempts']==1

def test_cross_slot_cache_copy_rejected(tmp_path):
    r=runtime(); b=r.DurableMockBackend(tmp_path,config(),transport=transport,token_count=len)
    a=b.complete('a',[{'role':'user','content':'alpha'}]); c=b.complete('b',[{'role':'user','content':'beta'}])
    (tmp_path/'responses'/(a['slot_hash']+'.json')).write_bytes((tmp_path/'responses'/(c['slot_hash']+'.json')).read_bytes())
    with pytest.raises(ValueError,match='identity'):b.complete('a',[{'role':'user','content':'alpha'}])

def test_completion_shaped_malformed_message_records_malformed(tmp_path):
    r=runtime()
    def strange(request,attempt):
        return 200,{}, {'choices':[{'message':'invalid-schema','finish_reason':'stop'}]}
    b=r.DurableMockBackend(tmp_path,config(),transport=strange,token_count=lambda s:len(s)//5)
    result=r.run_question(candidate(),0,b)
    assert all(x['status']=='MALFORMED' for x in result['cells'].values())
    assert result['baseline']['status']=='MALFORMED'
    assert b.summary()['successful_completions']==11

def test_nonempty_sender_packet_roundtrip_is_identical_and_source_owned(tmp_path):
    r=runtime(); b=r.DurableMockBackend(tmp_path,config(),transport=r.mock_transport,token_count=lambda s:len(s)//5)
    result=r.run_question(candidate(),0,b)
    assert result['packet']['answer']=='MOCK_ONLY_NOT_AN_INFERENCE'
    assert result['packet']['evidence']
    sender=result['plan']['sender']
    assert result['packet']['evidence'][0]['document_id'] in result['plan']['placements']['complete'][sender]
    messages={}
    for call in result['calls']:
        if '/update/' in call['slot']:
            req=json.loads((tmp_path/'requests'/(call['slot_hash']+'.json')).read_text())
            messages[call['slot']]=json.loads(req['request']['messages'][1]['content'])
    assert messages['q0/update/complete/peer']['information_packet']==messages['q0/update/missing/peer']['information_packet']
    assert messages['q0/update/complete/private']['information_packet']['answer'] is None

def test_ledger_directory_entry_fsynced_before_dispatch(tmp_path,monkeypatch):
    import os
    import stat
    r=runtime(); synced=[]; original=os.fsync
    def fsync(fd):
        if (tmp_path/'ledger.jsonl').exists() and stat.S_ISDIR(os.fstat(fd).st_mode):synced.append(True)
        return original(fd)
    monkeypatch.setattr(r.os,'fsync',fsync)
    def guarded(request,attempt):
        assert synced, 'new ledger directory entry not durably synced before dispatch'
        return transport(request,attempt)
    b=r.DurableMockBackend(tmp_path,config(),transport=guarded,token_count=len)
    b.complete('s',[{'role':'user','content':'x'}])
