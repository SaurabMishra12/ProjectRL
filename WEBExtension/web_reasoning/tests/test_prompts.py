import json
from .fixtures import wiki_row

def test_common_schema_and_no_private_identity():
    from WEBExtension.web_reasoning.prompts import build_messages
    from WEBExtension.web_reasoning.data import from_2wiki
    e=from_2wiki(wiki_row(),{});a=e.public.sources[0].atoms
    m=build_messages(e.public.question,a,None,(),None,'2wiki','initial')
    payload=json.loads(m[1]['content'])
    assert 'fixture-1' not in str(m) and 'supporting_facts' not in str(m)
    assert payload['output_schema']['additionalProperties'] is False
    assert payload['visible_sources'][0]['text']==a[0].text
    assert m==build_messages(e.public.question,a,None,(),None,'2wiki','initial')

def test_previous_proposal_roundtrip_is_a_model_visible_contract():
    from WEBExtension.web_reasoning.prompts import build_messages
    from WEBExtension.web_reasoning.contracts import Proposal,Fact
    from WEBExtension.web_reasoning.data import from_2wiki
    a=from_2wiki(wiki_row(),{}).public.sources[0].atoms[0]
    p=Proposal('partial',None,(Fact(a.atom_id,a.text,'Observed fact'),))
    data=json.loads(build_messages('Where?',(a,),p,(),None,'2wiki','synthesis')[1]['content'])
    assert data['own_previous_state']['facts'][0]['fact_id']==a.atom_id
    assert 'program' not in data['own_previous_state']
