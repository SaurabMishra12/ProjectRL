import pytest

def states():
    from WEBExtension.web_reasoning.contracts import AgentState,Outcome,Proposal,Fact
    good=Outcome('valid',Proposal('supported','Delta',(Fact('d:0','q','claim'),)))
    empty=Outcome('valid',Proposal('no_evidence',None))
    return (AgentState(0,(),(),empty),AgentState(1,(),(),good),AgentState(2,(),(),empty))

def test_abstentions_do_not_outvote_substantive_answer():
    from WEBExtension.web_reasoning.readout import abstention_vote
    from dataclasses import replace
    from WEBExtension.web_reasoning.contracts import Outcome
    assert abstention_vote(states(),(0,1,2))=='Delta'
    with pytest.raises(ValueError):abstention_vote((replace(states()[0],outcome=Outcome('invalid')),), (0,))

def test_original_vote_is_retained_as_diagnostic():
    from WEBExtension.web_reasoning.readout import original_vote,abstention_vote
    assert original_vote(states(),(0,1,2))=='noanswer'
    assert abstention_vote(states(),(0,1,2))=='Delta'
