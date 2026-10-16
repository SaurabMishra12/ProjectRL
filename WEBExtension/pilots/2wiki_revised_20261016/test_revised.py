"""Tests protect relaxed scoring, complementary coverage, and expansion gates."""
import importlib
import json

import pytest


def impl():return importlib.import_module('WEBExtension.pilots.2wiki_revised_20261016.core')


def raw(content,finish='stop'):
    return {'choices':[{'finish_reason':finish,'message':{'content':content}}],'usage':{'prompt_tokens':10,'completion_tokens':20}}


def test_relaxed_parser_keeps_answer_despite_invisible_source_and_long_lists():
    value={'answer':'Paris','facts':[{'source':'not visible','claim':'Paris'}]*30,'missing_entities':['entity']*12}
    parsed=impl().parse(raw(json.dumps(value)))
    assert parsed['kind']=='valid' and parsed['answer']=='Paris'
    assert len(parsed['facts'])==30 and len(parsed['missing_entities'])==12


@pytest.mark.parametrize('content',['broken','[]','{"answer":42}'])
def test_unusable_json_and_answer_types_remain_failures(content):
    assert impl().parse(raw(content))['kind']=='invalid'


def test_refusal_and_truncation_are_not_masked_as_valid_abstentions():
    assert impl().parse(raw('{}','content_filter'))['kind']=='refusal'
    assert impl().parse(raw('{"answer":"Paris"}','length'))['kind']=='invalid'


def test_usable_answer_survives_missing_optional_collections_and_fences():
    parsed=impl().parse(raw('```json\n{"answer":"Paris"}\n```'))
    assert parsed['answer']=='Paris' and parsed['kind']=='valid' and parsed['warnings']


def test_complementary_router_selects_two_distinct_missing_entities():
    states=[{'answer':None,'missing_entities':[]},{'answer':None,'missing_entities':['Alice birth date','Bob birth date']},{'answer':None,'missing_entities':[]}]
    assert impl().choose('complementary',states,[['Alice'],['Other'],['Bob']],'Who was born later, Alice or Bob?',123)==(0,2)


def test_complementary_router_can_stop_or_use_one_sender():
    states=[{'answer':None,'missing_entities':[]},{'answer':'Paris','missing_entities':[]},{'answer':None,'missing_entities':[]}]
    assert impl().choose('complementary',states,[['Alice'],['France'],['Bob']],'Question',123)==()
    states[1]={'answer':None,'missing_entities':['Alice birth date']}
    assert impl().choose('complementary',states,[['Alice'],['Other'],['Bob']],'Question',123)==(0,)


def test_unmatched_unresolved_need_conservatively_contacts_both_senders():
    states=[{}, {'answer':None,'missing_entities':['unknown bridge']}, {}]
    assert impl().choose('complementary',states,[['Alice'],['Other'],['Bob']],'Question',123)==(0,2)


def test_abstention_aware_vote_does_not_let_two_abstentions_win():
    assert impl().vote([{'answer':None},{'answer':'Paris'},{'answer':'noanswer'}])=='Paris'
    assert impl().vote([{'answer':'noanswer'}]*3)=='noanswer'


def gate_fixture():
    rows=[]
    for i in range(20):
        row={'arms':{}}
        for arm in impl().ARMS:
            correct=i<(2 if arm=='nocomm' else 3 if arm in ('random','heuristic') else 6)
            score={'em':100. if correct else 0.,'f1':100. if correct else 0.}
            row['arms'][arm]={'readouts':{'sink1':score,'abstention_aware':score},'hub_request_key':'same' if arm in ('instar','dense') else arm}
        rows.append(row)
    return rows


def test_gate_allows_only_the_declared_success_conditions():
    rows=gate_fixture()
    assert impl().gate(rows,{'physical_completions':180,'contract_failures':1})['pass']
    assert not impl().gate(rows,{'physical_completions':100,'contract_failures':2})['pass']
    assert not impl().gate(rows[:19],{'physical_completions':180,'contract_failures':0})['pass']


def test_gate_does_not_select_low_nocomm_readout_or_ignore_losing_complementary_arm():
    rows=gate_fixture()
    for row in rows[:4]:row['arms']['nocomm']['readouts']['abstention_aware']={'em':100.,'f1':100.}
    assert not impl().gate(rows,{'physical_completions':180,'contract_failures':0})['pass']
    rows=gate_fixture();rows[5]['arms']['complementary']['readouts']['sink1']={'em':0.,'f1':0.}
    assert not impl().gate(rows,{'physical_completions':180,'contract_failures':0})['pass']


def test_real_durable_cache_reuses_instar_dense_sink_and_survives_restart(tmp_path):
    from WEBExtension.web_reasoning.backend import DurableBackend
    from WEBExtension.web_reasoning.config import RunConfig
    from WEBExtension.web_reasoning.contracts import Atom,Source,PublicExample
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    driver=importlib.import_module('WEBExtension.pilots.2wiki_revised_20261016.run')
    config=RunConfig(team_sizes=(3,),mode='MOCK',model='fixture',packet_cap=512)
    tokenizer=UTF8FixtureTokenizer()
    def transport(request,attempt_id):
        return 200,{},raw('{"facts":["A fact"],"missing_entities":[],"answer":"yes"}')
    backend=DurableBackend(tmp_path,config,transport,tokenizer)
    atoms=[Atom(str(i),str(i),0,'text',str(i),'Source '+str(i)) for i in range(3)]
    public=PublicExample('q','2wiki','Is this a fixture?',tuple(Source(str(i),a.title,(a,)) for i,a in enumerate(atoms)))
    target={'answer':'yes','aliases':[]}
    row=driver.evaluate_row(public,[(a,) for a in atoms],target,backend,config,tokenizer,'pilot')
    assert row['arms']['instar']['hub_request_key']==row['arms']['dense']['hub_request_key']
    assert row['arms']['instar']['readouts']['sink1']['em']==100
    before=backend.summary()['physical_completions']
    assert before<len(backend.references)
    restarted=DurableBackend(tmp_path,config,transport,tokenizer)
    driver.evaluate_row(public,[(a,) for a in atoms],target,restarted,config,tokenizer,'pilot')
    assert restarted.summary()['physical_completions']==before


def test_preparation_excludes_missing_annotated_support_before_sampling():
    prepare=importlib.import_module('WEBExtension.pilots.2wiki_revised_20261016.prepare')
    assert not prepare.complete_support({'context':[['Alice',['a sentence']]],'supporting_facts':[['Alice',1]]})
    assert prepare.complete_support({'context':[['Alice',['a sentence']]],'supporting_facts':[['Alice',0]]})
