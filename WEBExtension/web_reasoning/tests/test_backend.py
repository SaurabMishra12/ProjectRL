import json,pytest
from .fixtures import backend_config,proposal_body

def request(model='a'):
    from WEBExtension.web_reasoning.backend import Request
    return Request([{'role':'user','content':json.dumps({'visible_sources':[]})}],{'model':model,'revision':'fixed'},__import__('WEBExtension.web_reasoning.tokenization',fromlist=['UTF8FixtureTokenizer']).UTF8FixtureTokenizer().identity,{'dataset':'2wiki','max_tokens':2048,'temperature':0,'schema_version':'v1','replicate_id':'0'})

def transport(calls):
    def send(req,attempt):
        calls.append(attempt)
        return 200,{}, {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(proposal_body(None,'no_evidence'))}}],'usage':{'prompt_tokens':20,'completion_tokens':10}}
    return send

def test_cache_and_identity(tmp_path):
    from WEBExtension.web_reasoning.backend import DurableBackend
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    calls=[];b=DurableBackend(tmp_path,backend_config(),transport(calls),UTF8FixtureTokenizer())
    assert b.complete(request(),{'arm':'a'}).kind=='valid'
    b.complete(request(),{'arm':'b'});assert len(calls)==1 and len(b.references)==2
    b.complete(request('different'),{'arm':'c'});assert len(calls)==2

def test_unknown_attempt_blocks_restart(tmp_path):
    from WEBExtension.web_reasoning.storage import Journal
    from WEBExtension.web_reasoning.backend import DurableBackend,UncertainRequest
    Journal(tmp_path/'calls.jsonl').append({'event':'started','attempt_id':'a1','key':'k1','request_sha256':'r1'})
    with pytest.raises(UncertainRequest):DurableBackend(tmp_path,backend_config(),None,None)

def test_unknown_transport_never_retries(tmp_path):
    from WEBExtension.web_reasoning.backend import DurableBackend,UncertainRequest
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    calls=[]
    def send(req,attempt):calls.append(attempt);raise TimeoutError('uncertain')
    b=DurableBackend(tmp_path,backend_config(),send,UTF8FixtureTokenizer())
    with pytest.raises(UncertainRequest):b.complete(request(),{})
    with pytest.raises(UncertainRequest):b.complete(request(),{})
    assert len(calls)==1

def test_budget_and_malformed_success(tmp_path):
    from WEBExtension.web_reasoning.backend import DurableBackend,BudgetExceeded
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    calls=[];b=DurableBackend(tmp_path,backend_config(max_completions=1,max_attempts=1),transport(calls),UTF8FixtureTokenizer())
    b.complete(request(),{})
    with pytest.raises(BudgetExceeded):b.complete(request('new'),{})
    assert len(calls)==1

def test_raw_response_recovery_and_corrupt_cache(tmp_path):
    from WEBExtension.web_reasoning.backend import DurableBackend,UncertainRequest
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    calls=[];t=UTF8FixtureTokenizer();b=DurableBackend(tmp_path,backend_config(),transport(calls),t)
    b.complete(request(),{});cache=next((tmp_path/'responses').glob('*.json'));cache.unlink()
    c=DurableBackend(tmp_path,backend_config(),transport(calls),t);c.complete(request(),{})
    assert len(calls)==1
    cache.write_text('{}')
    with pytest.raises(UncertainRequest):c.complete(request(),{})

def test_explicit_429_retries_are_bounded(tmp_path):
    from WEBExtension.web_reasoning.backend import DurableBackend,ProviderRejected
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    calls=[]
    def reject(req,attempt):calls.append(attempt);return 429,{}, {'error':{'message':'rate limit'}}
    b=DurableBackend(tmp_path,backend_config(),reject,UTF8FixtureTokenizer())
    with pytest.raises(ProviderRejected):b.complete(request(),{})
    assert len(calls)==3

def test_logical_slot_cannot_override_physical_identity(tmp_path):
    from WEBExtension.web_reasoning.backend import DurableBackend
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    calls=[];b=DurableBackend(tmp_path,backend_config(),transport(calls),UTF8FixtureTokenizer())
    with pytest.raises(ValueError):b.complete(request(),{'key':'fake'})
    assert not calls

def test_crash_after_terminal_event_recovers_without_execution(tmp_path,monkeypatch):
    import WEBExtension.web_reasoning.backend as module
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    calls=[];t=UTF8FixtureTokenizer();b=module.DurableBackend(tmp_path,backend_config(),transport(calls),t)
    original=module.atomic_json
    def broken(path,obj):
        if 'responses' in str(path):raise OSError('simulated crash')
        return original(path,obj)
    monkeypatch.setattr(module,'atomic_json',broken)
    with pytest.raises(OSError):b.complete(request(),{})
    monkeypatch.setattr(module,'atomic_json',original)
    recovered=module.DurableBackend(tmp_path,backend_config(),transport(calls),t)
    recovered.complete(request(),{})
    assert len(calls)==1
