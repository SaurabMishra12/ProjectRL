"""An answer mutation or a changed sink request must fail the offline audit."""
import importlib
import json

import pytest


def audit():
    return importlib.import_module('WEBExtension.pilots.2wiki_150_results_20261017.audit')


def test_audit_recomputes_scores_and_rejects_changed_readout(tmp_path):
    from WEBExtension.web_reasoning.backend import DurableBackend
    from WEBExtension.web_reasoning.config import RunConfig
    from WEBExtension.web_reasoning.contracts import Atom, Source, PublicExample
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    driver=importlib.import_module('WEBExtension.pilots.2wiki_revised_20261016.run')
    cfg=RunConfig(team_sizes=(3,),mode='MOCK',model='fixture',packet_cap=512)
    tok=UTF8FixtureTokenizer()
    def send(body,aid):
        return 200,{}, {'choices':[{'finish_reason':'stop','message':{'content':'{"facts":["Fact"],"missing_entities":[],"answer":"yes"}'}}],
                       'usage':{'prompt_tokens':20,'completion_tokens':10}}
    b=DurableBackend(tmp_path,cfg,send,tok)
    atoms=[Atom(str(i),str(i),0,'text',str(i),'Source '+str(i)) for i in range(3)]
    public=PublicExample('q','2wiki','A question?',tuple(Source(str(i),a.title,(a,)) for i,a in enumerate(atoms)))
    shards=[(a,) for a in atoms];target={'answer':'yes','aliases':[]}
    row=driver.evaluate_row(public,shards,target,b,cfg,tok,'additional')
    result=audit().verify_row(row,public,shards,target,cfg,tok,tmp_path)
    assert result['readouts_checked']==14
    assert result['sink_identity'] is True
    row['arms']['instar']['missing_usage_fields']=1
    with pytest.raises(AssertionError):
        audit().verify_row(row,public,shards,target,cfg,tok,tmp_path)
    row['arms']['instar']['missing_usage_fields']=0
    row['arms']['instar']['hub_request_key']='changed'
    with pytest.raises(AssertionError):
        audit().verify_row(row,public,shards,target,cfg,tok,tmp_path)
    row['arms']['instar']['hub_request_key']=row['arms']['dense']['hub_request_key']
    row['arms']['instar']['readouts']['sink1']['em']=0.
    with pytest.raises(AssertionError):
        audit().verify_row(row,public,shards,target,cfg,tok,tmp_path)


def attempt_fixture(path):
    from WEBExtension.web_reasoning.backend import DurableBackend,Request
    from WEBExtension.web_reasoning.config import RunConfig
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    cfg=RunConfig(mode='MOCK',model='fixture');tok=UTF8FixtureTokenizer();calls=[]
    def send(body,aid):
        calls.append(aid)
        if len(calls)==1:return 429,{}, {'error':{'message':'rate limit'}}
        return 200,{}, {'choices':[{'finish_reason':'stop','message':{'content':'{"answer":"yes"}'}}],
                       'usage':{'prompt_tokens':20,'completion_tokens':10}}
    backend=DurableBackend(path,cfg,send,tok)
    request=Request([{'role':'user','content':'{"visible_sources":[]}'}],{'model':'fixture'},tok.identity,{'dataset':'2wiki','max_tokens':100})
    backend.complete(request,{'qid':'q'})
    return cfg,tok


def test_attempt_audit_rejects_forged_budget_free_rejection(tmp_path):
    from WEBExtension.web_reasoning.storage import digest,atomic_json
    cfg,tok=attempt_fixture(tmp_path)
    result=audit().verify_attempts(tmp_path,cfg,tok)
    assert result['verified_attempts']==2
    assert result['verified_rejections']==1
    path=next(p for p in (tmp_path/'attempts').glob('*.json') if json.loads(p.read_text())['status']==429)
    saved=json.loads(path.read_text());saved['status']=500
    saved['sha256']=digest({k:v for k,v in saved.items() if k!='sha256'})
    atomic_json(path,saved)
    from WEBExtension.web_reasoning.storage import canonical
    ledger=tmp_path/'calls.jsonl';events=[json.loads(l) for l in ledger.read_text().splitlines()]
    for event in events:
        if event['event']=='rejected':event['status']=500
    ledger.write_text(''.join(canonical(e)+'\n' for e in events))
    with pytest.raises(AssertionError):audit().verify_attempts(tmp_path,cfg,tok)


def test_attempt_audit_recomputes_each_dispatch_reservation(tmp_path):
    from WEBExtension.web_reasoning.storage import canonical
    cfg,tok=attempt_fixture(tmp_path)
    path=tmp_path/'calls.jsonl'
    events=[json.loads(l) for l in path.read_text().splitlines()]
    events[0]['reserved_input']=0
    path.write_text(''.join(canonical(e)+'\n' for e in events))
    with pytest.raises(AssertionError):audit().verify_attempts(tmp_path,cfg,tok)
