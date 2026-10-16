import json,pytest
from .fixtures import w1_case,backend_config

def test_variants_are_common_task_configs_and_prefix_is_explicit(tmp_path):
    from WEBExtension.web_reasoning.ablations import ablation_variants
    from WEBExtension.web_reasoning.runner import run_w1
    variants=ablation_variants(backend_config())
    assert any(c.packet_mode=='prefix' for c in variants)
    p,a,s,b,c=w1_case(3,tmp_path)
    from dataclasses import replace
    prefix=replace(c,packet_mode='prefix',packet_cap=192)
    result=run_w1(p,a,s,b,prefix)
    assert all(x.tokens<=192 for x in result.arms['instar'].transmissions if x.kind!='control')

def test_legacy_adapter_does_not_invent_facts(tmp_path):
    from WEBExtension.web_reasoning.parsing import parse_outcome
    from WEBExtension.web_reasoning.data import from_2wiki
    from .fixtures import wiki_row
    a=from_2wiki(wiki_row(),{}).public.sources[0].atoms
    raw={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'answer':'noanswer','citations':[['Mira',0]]})}}]}
    o=parse_outcome(raw,a,'2wiki',variant='legacy')
    assert o.kind=='valid' and o.proposal.answer is None and len(o.proposal.facts)==1

def test_legacy_style_runs_all_arms_under_same_contract(tmp_path):
    from dataclasses import replace
    from WEBExtension.web_reasoning.runner import run_w1
    p,a,s,b,c=w1_case(3,tmp_path)
    run=run_w1(p,a,s,b,replace(c,prompt_variant='legacy'))
    assert all(arm.outcome.kind=='valid' for arm in run.arms.values())
