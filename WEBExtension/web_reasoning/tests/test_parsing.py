import json,pytest
from .fixtures import wiki_row,proposal_body

def raw(body,finish='stop'):
    return {'choices':[{'finish_reason':finish,'message':{'content':json.dumps(body)}}],'usage':{'prompt_tokens':10,'completion_tokens':5}}

def atom():
    from WEBExtension.web_reasoning.data import from_2wiki
    return from_2wiki(wiki_row(),{}).public.sources[0].atoms[0]

@pytest.mark.parametrize('bad',['status','answer','quote','locator','length','unknown','counts'])
def test_invalid_output_preserved(bad):
    from WEBExtension.web_reasoning.parsing import parse_outcome
    a=atom();b=proposal_body(a);finish='stop'
    if bad=='status':b['status']='certain'
    if bad=='answer':b['answer']='guess'
    if bad=='quote':b['facts'][0]['quote']='invented'
    if bad=='locator':b['facts'][0]['sentence_id']=2
    if bad=='length':finish='length'
    if bad=='unknown':b['confidence']=1.0
    if bad=='counts':b['entity_tags']=['a']*9
    o=parse_outcome(raw(b,finish),(a,),'2wiki')
    assert o.kind=='invalid' and o.proposal is None and o.raw_sha256 and o.usage

def test_partial_evidence_retained_and_supported_requires_fact():
    from WEBExtension.web_reasoning.parsing import parse_outcome
    a=atom();o=parse_outcome(raw(proposal_body(a)),(a,),'2wiki')
    assert o.kind=='valid' and o.proposal.answer is None and len(o.proposal.facts)==1
    assert parse_outcome(raw(proposal_body(None,'supported','Delta')),(),'2wiki').kind=='invalid'

def test_refusal_duplicate_json_and_unknown_transport():
    from WEBExtension.web_reasoning.parsing import parse_outcome
    assert parse_outcome({'choices':[{'message':{'refusal':'refused'},'finish_reason':'stop'}]},(),'2wiki').kind=='refusal'
    assert parse_outcome({'choices':[{'message':{'content':'{"status":"partial","status":"supported"}'}}]},(),'2wiki').kind=='invalid'
    assert parse_outcome({},(),'2wiki').kind=='invalid'
