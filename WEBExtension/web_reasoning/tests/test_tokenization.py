import pytest

def test_fixture_tokenizer_cannot_claim_scientific_tokens():
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer,PinnedTokenizer
    t=UTF8FixtureTokenizer();assert t.count('Δ')==2 and t.identity['kind']=='mock'
    with pytest.raises(ValueError):PinnedTokenizer('/missing','main','0'*64)
    with pytest.raises((OSError,ValueError,FileNotFoundError)):PinnedTokenizer('/missing','pinned','0'*64)
