import pytest

def test_fact_merge_preserves_conflicting_sources():
    from WEBExtension.web_reasoning.contracts import Fact
    from WEBExtension.web_reasoning.evidence import merge_ledger
    a=Fact('d0:0','Delta','birthplace');b=Fact('d1:0','Epsilon','birthplace')
    assert merge_ledger((a,),(a,b))==(a,b)
    with pytest.raises(ValueError):merge_ledger((a,),(Fact('d0:0','changed','x'),))
