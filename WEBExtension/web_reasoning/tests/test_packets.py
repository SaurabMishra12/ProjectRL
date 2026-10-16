import json,pytest
from .fixtures import wiki_row

def state():
    from WEBExtension.web_reasoning.contracts import AgentState,Fact,Outcome,Proposal
    from WEBExtension.web_reasoning.data import from_2wiki
    a=from_2wiki(wiki_row(),{}).public.sources[0].atoms[0]
    f=Fact(a.atom_id,a.text,'Birthplace Delta')
    return AgentState(0,(a,),(f,),Outcome('valid',Proposal('partial',None,(f,))))

@pytest.mark.parametrize('cap',[100,192,1000])
def test_packets_are_whole_and_exactly_counted(cap):
    from WEBExtension.web_reasoning.packets import pack_evidence,apply_delivery
    from WEBExtension.web_reasoning.contracts import AgentState,Outcome
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    t=UTF8FixtureTokenizer();s=state();p=pack_evidence(s,1,None,cap,t)
    assert p.tokens==t.count(p.text)<=cap
    receiver=AgentState(1,(),(),Outcome('unresolved'))
    updated=apply_delivery(receiver,p)
    assert len(updated.visible_atoms)==len(p.facts)
    for a in updated.visible_atoms:assert a.text==s.visible_atoms[0].text and a.atom_id in p.text
    if p.facts:assert p.facts[0].quote==s.ledger[0].quote

def test_too_small_cap_fails_before_dispatch():
    from WEBExtension.web_reasoning.packets import pack_evidence
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    with pytest.raises(ValueError):pack_evidence(state(),1,None,1,UTF8FixtureTokenizer())

def test_control_and_unicode_caps():
    from WEBExtension.web_reasoning.packets import pack_control
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    t=UTF8FixtureTokenizer()
    assert t.count(pack_control(0,{'status':'partial','count':1,'tags':['Δ'*100]},96,t))<=96
