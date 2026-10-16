"""The human-authorized timeout replacement must retain a hard one-attempt cap."""
import importlib
import json
from dataclasses import replace
import pytest
from WEBExtension.web_reasoning.backend import DurableBackend,Request,ProviderRejected
from WEBExtension.web_reasoning.config import RunConfig
from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer


def test_timeout_reservation_is_removed_from_working_budget_but_not_global():
    m=importlib.import_module('WEBExtension.pilots.2wiki_150_resume_20261017.resume')
    cfg=RunConfig(max_completions=100,max_attempts=120,max_input_tokens=50000,max_output_tokens=60000)
    out=m.reserve_unknown(cfg,{'reserved_input':576,'reserved_output':4096})
    assert out.max_completions==99 and out.max_attempts==119
    assert out.max_input_tokens==49424 and out.max_output_tokens==55904


def test_replacement_key_cannot_be_dispatched_twice_after_confirmed_rejection(tmp_path):
    m=importlib.import_module('WEBExtension.pilots.2wiki_150_resume_20261017.resume')
    cfg=RunConfig(mode='MOCK',model='fixture');tok=UTF8FixtureTokenizer()
    req=Request([{'role':'user','content':'{"visible_sources":[]}'}],{'model':'fixture'},tok.identity,{'dataset':'2wiki','max_tokens':100})
    m.REPLACEMENT_KEY=req.key
    def send(body,aid):return 429,{}, {'error':{'message':'rate limit'}}
    backend=m.OneReplacementBackend(tmp_path,cfg,send,tok)
    with pytest.raises(ProviderRejected,match='one_authorized_replacement_attempt_used'):
        backend.complete(req,{'qid':'q'})
    assert backend.summary()['physical_attempts']==1
    assert backend.summary()['rejected_attempts']==1
    assert backend.summary()['uncertain_attempts']==0


def test_missing_success_usage_keeps_full_reservation_in_combined_report():
    m=importlib.import_module('WEBExtension.pilots.2wiki_150_resume_20261017.resume')
    events=[{'event':'started','attempt_id':'a','reserved_input':100,'reserved_output':4096},
            {'event':'success','attempt_id':'a','usage':{'prompt_tokens':None,'completion_tokens':12}}]
    assert m.charged_usage(events)==(100,12)
    events.append({'event':'started','attempt_id':'b','reserved_input':576,'reserved_output':4096})
    events.append({'event':'transport_error','attempt_id':'b','error_type':'APITimeoutError'})
    assert m.charged_usage(events)==(676,4108)
