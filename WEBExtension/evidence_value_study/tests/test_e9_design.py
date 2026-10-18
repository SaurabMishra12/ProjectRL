import importlib
import json
from collections import Counter

import pytest

MODULE='WEBExtension.evidence_value_study.experiments.design'

def design():
    assert importlib.util.find_spec(MODULE) is not None, 'E9 pure design not implemented'
    return importlib.import_module(MODULE)

def candidate(i=0):
    return {'id':f'q{i}','question':f'Which city hosts object {i}?',
            'documents':[{'id':f'd{j}','title':f'Original title {j}', 'text':f'Visible source {j}.'} for j in range(6)],
            'support_ids':['d0','d1'],'gold_answer':'SEALED_GOLD_CANARY'}

def test_accessibility_changes_receiver_only_and_union_is_identical():
    d=design(); p=d.assign(candidate(),0)
    r,s=p['receiver'],p['sender']; c,m=p['placements']['complete'],p['placements']['missing']
    assert len(c[r])==len(m[r])==2
    assert set(c[r])==set(candidate()['support_ids'])
    assert len(set(m[r])&set(candidate()['support_ids']))==1
    assert c[s]==m[s] and p['missing_document'] in c[s]
    assert set().union(*map(set,c))==set().union(*map(set,m))=={f'd{i}' for i in range(6)}
    assert all(c[i]==m[i] for i in range(3) if i!=r)

def test_receiver_and_support_counterbalance_for_30_questions():
    d=design(); plans=[d.assign(candidate(i),i) for i in range(30)]
    assert Counter(p['receiver'] for p in plans)=={0:10,1:10,2:10}
    assert Counter(p['missing_document'] for p in plans)=={'d0':15,'d1':15}
    assert Counter(p['update_order'][0] for p in plans)=={'private':15,'peer':15}

def test_payload_allowlists_and_state_reuse():
    d=design(); q=candidate(); p=d.assign(q,0)
    initial=d.initial_messages(q,p,'complete',p['receiver'])
    assert 'SEALED_GOLD_CANARY' not in json.dumps(initial)
    own={'answer':'Possibly X','evidence':[],'uncertain':True,'raw':'INITIAL_STATE_CANARY'}
    peer={'answer':'Y','evidence':[{'document_id':'d1','quote':'Visible source 1.'}], 'status':'VALID'}
    a=d.update_messages(q,p,'complete',own,None)
    b=d.update_messages(q,p,'complete',own,peer)
    assert a[0]==b[0]
    ua,ub=(json.loads(x[1]['content']) for x in (a,b))
    assert ua['initial_state']==ub['initial_state']
    assert ua['documents']==ub['documents']
    assert ua['information_packet']!=ub['information_packet']
    assert 'SEALED_GOLD_CANARY' not in json.dumps([a,b])
    assert len(json.loads(d.baseline_messages(q)[1]['content'])['documents'])==6

def test_sender_quotes_are_verbatim_and_ownership_checked():
    d=design(); q=candidate(); p=d.assign(q,0); allowed=p['placements']['complete'][p['sender']]
    bad={'answer':'X','evidence':[{'document_id':'d99','quote':'Hidden'}],'uncertain':False}
    parsed=d.parse_response(json.dumps(bad),q,allowed)
    assert parsed['status']=='MALFORMED' and parsed['answer'] is None
    bad['evidence']=[{'document_id':allowed[0],'quote':'Invented statement'}]
    assert d.parse_response(json.dumps(bad),q,allowed)['status']=='MALFORMED'

def test_malformed_refused_uncertain_and_empty_outputs():
    d=design(); q=candidate(); allowed=['d0']
    for text in ['not json','{}','{"answer": 23}','{"answer":"x","evidence":[],"uncertain":false,"gold":"x"}']:
        assert d.parse_response(text,q,allowed)['status']=='MALFORMED'
    assert d.parse_response('',q,allowed,refusal='refused')['status']=='REFUSED'
    assert d.parse_response('{"answer":"","evidence":[],"uncertain":true}',q,allowed)['status']=='ABSTAINED'
    assert d.parse_response('{"answer":"x","evidence":[],"uncertain":true}',q,allowed)['status']=='UNCERTAIN'
    assert d.parse_response('{"answer":"x","evidence":[],"uncertain":false}',q,allowed,finish_reason='length')['status']=='MALFORMED'

def test_packet_identical_across_accessibility_and_role_balance():
    d=design(); q=candidate(); p=d.assign(q,0)
    own={'raw':'frozen state','answer':'X','evidence':[],'uncertain':False,'status':'VALID'}
    packet={'answer':'Z','evidence':[],'status':'VALID'}
    complete,missing=(json.loads(d.update_messages(q,p,s,own,packet)[1]['content']) for s in ['complete','missing'])
    assert complete['information_packet']==missing['information_packet']
    assert complete['documents']!=missing['documents']
    counts=Counter(d.assign(candidate(i),i)['roles'][d.assign(candidate(i),i)['receiver']] for i in range(30))
    assert counts=={role:10 for role in d.ROLES}

def test_quote_cannot_be_invented_by_joining_separate_received_spans():
    d=design()
    q={'documents':[{'id':'d1','text':'first\nsecond','visible_spans':['first','second']}]}
    raw=json.dumps({'answer':'X','uncertain':False,'evidence':[{'document_id':'d1','quote':'first\nsecond'}]})
    assert d.parse_response(raw,q,['d1'])['status']=='MALFORMED'
