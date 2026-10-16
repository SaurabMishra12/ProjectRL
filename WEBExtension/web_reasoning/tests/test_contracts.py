import dataclasses
import pytest

def test_public_serializer_has_only_allowed_fields():
    from WEBExtension.web_reasoning.contracts import Atom,Source,PublicExample,runtime_object
    a=Atom("d0:0","d0",0,"text","Mira was born in Delta.","Mira")
    p=PublicExample("private-id","2wiki","Where?",(Source("d0","Mira",(a,)),))
    assert set(runtime_object(p))=={"question","visible_sources"}
    assert "private-id" not in str(runtime_object(p))
    with pytest.raises(dataclasses.FrozenInstanceError):p.question="changed"
    with pytest.raises(TypeError):PublicExample("x","2wiki","q",(),answer="secret")

def test_config_rejects_unknown_and_inconsistent_fields():
    from WEBExtension.web_reasoning.config import validate_config
    with pytest.raises(ValueError):validate_config({"gold":"secret"})
    with pytest.raises(ValueError):validate_config({"max_completions":10,"max_attempts":2})
    with pytest.raises(ValueError):validate_config({"team_sizes":[1]})
    c=validate_config({})
    assert c.packet_cap==192 and c.metadata_cap==96 and c.max_control_frames==6
    assert c.mode=="MOCK" and c.max_output_per_call==2048

def test_outcome_status_and_nested_sequences_are_immutable():
    from WEBExtension.web_reasoning.contracts import Fact,Proposal,Outcome
    f=Fact("d0:0","q","c")
    p=Proposal("partial",None,(f,))
    assert p.facts==(f,)
    with pytest.raises(ValueError):Proposal("partial","guess",(f,))
    with pytest.raises(ValueError):Outcome("valid",None)
