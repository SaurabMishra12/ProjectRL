import pytest
from scipy.stats import binom
from WEBExtension.causal_study.next_phase.power_refinement import exact_power


def test_exact_power_matches_all_gains_hand_case():
    # At delta=q, every discordance is a gain. Two-sided p<=.05 needs >=6.
    assert exact_power(50, .1, .1) == pytest.approx(binom.sf(5, 50, .1), abs=1e-12)


def test_exact_power_has_null_size_below_alpha_and_increases_at_large_effect():
    assert exact_power(200, 0, .25) <= .05
    assert exact_power(400, .1, .25) > .95


def test_exact_power_rejects_impossible_joint_rates():
    with pytest.raises(ValueError):
        exact_power(50, .2, .1)
