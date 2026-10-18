import pytest
from WEBExtension.causal_study.external_validation.power_budget import n_normal,simulated_power


def test_interaction_power_scales_with_cluster_variance_not_agent_count():
    assert abs(n_normal(.05,1)/n_normal(.05,.5)-4)<.01
    assert n_normal(.05,2)>2366
    assert n_normal(.05,.5)<2366


def test_bounded_interaction_generator_rejects_impossible_variance_and_uses_question_n():
    with pytest.raises(ValueError):simulated_power(100,.1,.0625)
    p=simulated_power(n_normal(.05,.5),.05,.25,draws=10000)
    assert .76<p['simulated_power']<.85
