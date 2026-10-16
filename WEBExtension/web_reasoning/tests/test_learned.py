import pytest

def test_scorer_is_equivariant_and_parameter_count_independent_of_n():
    import torch
    from WEBExtension.web_reasoning.learned import PooledSenderScorer
    torch.manual_seed(7);model=PooledSenderScorer().eval();count=sum(p.numel() for p in model.parameters())
    for n in (2,3,4,5):
        x=torch.randn(n-1,32);sink=torch.randn(32);perm=torch.randperm(n-1)
        assert torch.allclose(model(x[perm],sink),model(x,sink)[perm],atol=1e-6)
        assert model(x,sink).shape==(n-1,) and sum(p.numel() for p in model.parameters())==count

def test_features_are_only_observable_bounded_metadata():
    from WEBExtension.web_reasoning.learned import make_features
    from WEBExtension.web_reasoning.policies import Metadata
    m=tuple(Metadata(i,'partial',2,('Mira',),('birthplace',),()) for i in range(5))
    x,sink,nodes=make_features(m,0,'Where was Mira born?',set(),7)
    assert x.shape==(4,32) and sink.shape==(32,) and nodes==(1,2,3,4)
