import pytest
from .fixtures import w1_case

def test_stop_requires_supported_complete_visible_evidence():
    from WEBExtension.web_reasoning.adaptive import stop_fields
    assert not stop_fields('partial',None,2,0,0)
    assert not stop_fields('supported','Delta',1,1,0)
    assert not stop_fields('supported','Delta',1,0,1)
    assert stop_fields('supported','Delta',1,0,0)

@pytest.mark.parametrize('n',[2,3,4,5])
def test_adaptive_budget_and_unknown_status(n,tmp_path):
    from WEBExtension.web_reasoning.runner import initialize
    from WEBExtension.web_reasoning.adaptive import run_adaptive,validate_transmission_budget
    p,a,s,b,c=w1_case(n,tmp_path)
    initial=initialize(p,a,b,c);r=run_adaptive(p,initial,s,b,c,'selected')
    validate_transmission_budget(r.transmissions,n,c.packet_cap,c.metadata_cap)
    assert len([x for x in r.transmissions if x.kind!='control'])<=3*(n-1)
    assert r.stop_reason!='supported_stop'
    assert all(f.atom_id in {x.atom_id for x in r.states[s].visible_atoms} for f in r.states[s].ledger)

def test_excess_budget_rejected():
    from WEBExtension.web_reasoning.adaptive import validate_transmission_budget
    from WEBExtension.web_reasoning.packets import Packet
    with pytest.raises(ValueError):validate_transmission_budget([Packet(0,1,'x',1)]*4,2,192,96)

def test_followup_really_executes_and_dense_comparator_runs(tmp_path):
    from WEBExtension.web_reasoning.runner import initialize
    from WEBExtension.web_reasoning.adaptive import run_adaptive
    p,a,s,b,c=w1_case(4,tmp_path,with_needs=True);initial=initialize(p,a,b,c)
    result=run_adaptive(p,initial,s,b,c,'selected')
    assert len([x for x in result.transmissions if x.kind=='request'])==2
    assert len(result.references)==4
    dense=run_adaptive(p,initial,s,b,c,'dense',always_run=True)
    assert len(dense.references)==12
    assert len([x for x in result.transmissions if x.kind=='control'])>4

@pytest.mark.parametrize('n',[2,3,4,5])
@pytest.mark.parametrize('phases',[1,2,3])
def test_fixed_controls_ignore_initial_supported_answer(n,phases,tmp_path):
    from dataclasses import replace
    from WEBExtension.web_reasoning.runner import initialize
    from WEBExtension.web_reasoning.adaptive import run_adaptive
    from WEBExtension.web_reasoning.contracts import Outcome,Proposal,Fact
    p,a,s,b,c=w1_case(n,tmp_path);c=replace(c,max_phases=phases)
    initial=list(initialize(p,a,b,c));atom=initial[s].visible_atoms[0]
    initial[s]=replace(initial[s],outcome=Outcome('valid',Proposal('supported','Delta',(Fact(atom.atom_id,atom.text,'Observed'),))))
    private=run_adaptive(p,tuple(initial),s,b,c,'private')
    assert len(private.references)==sum((1,max(1,n//2),1)[:phases])
    dense=run_adaptive(p,tuple(initial),s,b,c,'dense')
    assert len(dense.references)==n*phases
