import pytest
from WEBExtension.causal_study.next_phase.planning import exact_mcnemar,paired_analysis,power_grid


def test_exact_mcnemar_matches_hand_enumerated_binomial():
    assert exact_mcnemar(8,0)==pytest.approx(2/256)
    assert exact_mcnemar(0,8)==pytest.approx(2/256)
    assert exact_mcnemar(3,3)==1
    assert exact_mcnemar(0,0)==1


def test_paired_effect_uses_questions_and_rejects_duplicate_units():
    rows=[{'question_id':'q1','recorded_em':100,'exchanged_em':0,'recorded_f1':100,'exchanged_f1':0},
          {'question_id':'q2','recorded_em':0,'exchanged_em':100,'recorded_f1':50,'exchanged_f1':100}]
    r=paired_analysis(rows,draws=100)
    assert r['n_questions']==2 and r['em_delta_pp']==0 and r['f1_delta_pp']==25
    assert r['gains']==1 and r['losses']==1 and r['exact_mcnemar_p']==1
    with pytest.raises(ValueError):paired_analysis(rows+[rows[0]])


def test_missing_cell_is_not_silently_zero():
    with pytest.raises(ValueError):paired_analysis([{'question_id':'q','recorded_em':100}])


def test_power_simulates_paired_discordance_and_reports_no_outcome_estimate():
    rows=power_grid(effects=(.1,),discordances=(.25,),sample_sizes=(50,400),simulations=2000)
    assert len(rows)==2
    assert rows[1]['estimated_power']>rows[0]['estimated_power']
    assert rows[1]['estimated_power']>.9
    assert rows[1]['method']=='exact_conditional_mcnemar_monte_carlo'


def test_impossible_effect_discordance_is_rejected():
    with pytest.raises(ValueError):power_grid(effects=(.2,),discordances=(.1,),sample_sizes=(50,),simulations=10)
