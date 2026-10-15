import copy
import json
from collections import Counter

import pytest

from WEBExtension.causal_study.next_phase.binding_design import make_case, make_manifest, messages, validate_case, validate_manifest


def fixture(qid='q'):
    responses=[{'answer':'Paris','citations':[['France',0]]},
               {'answer':'Lyon','citations':[['Lyon',0]]},
               {'answer':'noanswer','citations':[]}]
    facts=[[{'title':'France','sentence_id':0,'text':'Paris is the capital of France.'}],
           [{'title':'Lyon','sentence_id':0,'text':'Lyon is a city in France.'}],[]]
    return make_case(qid,'What is the capital of France?',responses,facts)


def test_swap_changes_only_two_references_and_preserves_all_degrees():
    case=fixture();validate_case(case)
    assert case['eligible']
    a,b=[json.loads(messages(case,arm)[1]['content']) for arm in ['recorded','exchanged']]
    assert a['sources']==b['sources'] and a['question']==b['question']
    assert [c['answer'] for c in a['candidates']]==['Paris','Lyon','noanswer']
    assert [c['answer'] for c in a['candidates']]==[c['answer'] for c in b['candidates']]
    assert [len(c['source_ids']) for c in a['candidates']]==[1,1,0]
    assert Counter(x for c in a['candidates'] for x in c['source_ids'])==Counter(x for c in b['candidates'] for x in c['source_ids'])
    assert sum(x!=y for l,r in zip(a['candidates'],b['candidates']) for x,y in zip(l['source_ids'],r['source_ids']))==2


def test_only_abstention_disagreement_does_not_qualify():
    r=[{'answer':'Paris','citations':[['France',0]]},{'answer':'noanswer','citations':[]},{'answer':'noanswer','citations':[]}]
    case=make_case('q','Question',r,[[{'title':'France','sentence_id':0,'text':'Source'}],[],[]])
    assert not case['eligible'] and case['reason']=='FEWER_THAN_TWO_SUBSTANTIVE_ANSWERS'


def test_invalid_sender_invisible_citation_never_enters_bank():
    r=[{'answer':'A','citations':[['Hidden',0]]},{'answer':'B','citations':[['Shown',0]]},{'answer':'noanswer','citations':[]}]
    case=make_case('q','Question',r,[[],[{'title':'Shown','sentence_id':0,'text':'Visible'}],[]])
    assert not case['eligible']
    assert case['reason']=='NO_ADMISSIBLE_TARGET_EXCHANGE'


def test_duplicate_citations_after_exchange_are_rejected():
    facts=[{'title':'A','sentence_id':0,'text':'First'}, {'title':'B','sentence_id':0,'text':'Second'}]
    r=[{'answer':'x','citations':[['A',0],['B',0]]},{'answer':'y','citations':[['A',0],['B',0]]},{'answer':'noanswer','citations':[]}]
    assert not make_case('q','Question',r,[facts,facts,[]])['eligible']


def test_source_and_answer_corruption_fail_validation():
    case=fixture();case['source_bank'][0]['text']='rewritten'
    with pytest.raises(ValueError): validate_case(case)
    case=fixture();case['candidates'][0]['answer']='changed'
    with pytest.raises(ValueError): validate_case(case)


def test_gold_and_arm_metadata_are_not_model_visible():
    case=fixture();case['gold_answer']='SECRET_GOLD';case['reference']='SECRET_REF'
    content=json.dumps(messages(case,'recorded'))
    assert 'SECRET_GOLD' not in content and 'SECRET_REF' not in content
    assert set(json.loads(messages(case,'recorded')[1]['content']))=={'question','candidates','sources'}
    assert 'exchanged' not in content and 'recorded' not in content


def test_twelve_question_schedule_balances_candidate_and_arm_orders():
    cases=[fixture(str(i)) for i in range(20)]
    m=make_manifest(cases,n=12)
    validate_manifest(m)
    assert len(m['questions'])==12 and m['projected_finalizer_calls']==24
    assert sorted(Counter(tuple(q['candidate_order']) for q in m['questions']).values())==[2]*6
    assert sorted(Counter(tuple(q['arm_order']) for q in m['questions']).values())==[6,6]
    assert set(Counter((tuple(q['candidate_order']),tuple(q['arm_order'])) for q in m['questions']).values())=={1}
    assert {q['question_id'] for q in m['questions']}=={q['question_id'] for q in make_manifest(list(reversed(cases)),n=12)['questions']}


def test_duplicate_question_ids_and_changed_manifest_are_rejected():
    with pytest.raises(ValueError):make_manifest([fixture(),fixture()],n=2)
    m=make_manifest([fixture()],n=1);m['questions'][0]['question']='edited'
    with pytest.raises(ValueError):validate_manifest(m)


def test_prompt_token_equality_is_measured_not_assumed():
    case=fixture();m=make_manifest([case],n=1,token_count=lambda s:len(s.encode()))
    q=m['questions'][0]
    assert q['prompt_tokens']['recorded']==q['prompt_tokens']['exchanged']
    assert q['prompt_tokens']['recorded']>0
