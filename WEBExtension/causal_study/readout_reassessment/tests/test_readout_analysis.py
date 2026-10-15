import inspect
import itertools
import pytest
from WEBExtension.causal_study.readout_reassessment.analysis import (
    normalize, score_answer, is_abstention, aggregate, paired, interaction, validate_rows)


def responses(*answers):
    return [{'answer': a, 'provider_refusal': a == ''} for a in answers]


def test_abstentions_are_not_votes_against_a_substantive_answer():
    rs = responses('noanswer', 'Paris', 'noanswer')
    assert aggregate(rs, 'original') == 'noanswer'
    assert aggregate(rs, 'abstention_aware') == 'Paris'
    assert aggregate(rs, 'abstention_lexical') == 'Paris'


def test_plurality_is_not_strict_majority_and_node_order_breaks_ties():
    rs = responses('Paris', 'Lyon', 'Marseille')
    assert aggregate(rs, 'original') == 'Paris'
    assert aggregate(list(reversed(rs)), 'original') == 'Marseille'
    assert aggregate(rs, 'abstention_lexical') == 'Lyon'
    assert aggregate(list(reversed(rs)), 'abstention_lexical') == 'Lyon'


def test_one_abstention_can_win_a_three_way_tie_by_node_order():
    rs = responses('noanswer', 'Paris', 'Lyon')
    assert aggregate(rs, 'original') == 'noanswer'
    assert aggregate(rs, 'abstention_aware') == 'Paris'


def test_normalized_aliases_pool_without_gold_labels():
    assert normalize('The PARIS!') == 'paris'
    assert aggregate(responses('The Paris!', 'Paris', 'Lyon'), 'original') == 'The Paris!'
    assert list(inspect.signature(aggregate).parameters) == ['responses', 'rule']


def test_valid_letter_a_yes_no_unknown_are_not_abstentions():
    for answer in ['A', 'yes', 'no', 'unknown', 'not enough information']:
        assert not is_abstention({'answer': answer})
    assert normalize('A') == ''
    assert aggregate(responses('A', 'noanswer', 'noanswer'), 'abstention_aware') == 'A'


def test_explicit_refusal_and_all_abstaining_behavior():
    assert is_abstention({'answer': '', 'provider_refusal': True})
    assert aggregate(responses('', 'noanswer', 'noanswer'), 'abstention_aware') == 'noanswer'
    with pytest.raises(ValueError): is_abstention({'answer': ''})
    with pytest.raises(ValueError): is_abstention({'answer': '  '})
    with pytest.raises(ValueError): is_abstention({'answer': 'Paris', 'provider_refusal': True})


def test_empty_or_invalid_sets_do_not_become_predictions():
    with pytest.raises(ValueError): aggregate([], 'original')
    with pytest.raises(ValueError): aggregate(responses('Paris'), 'fitted_oracle')


def test_official_answer_scoring_special_cases():
    assert score_answer('The Paris!', 'Paris') == {'em': 100., 'f1': 100.}
    assert score_answer('yes indeed', 'yes')['f1'] == 0
    assert score_answer('New York City', 'New York')['f1'] == pytest.approx(80.)
    assert score_answer('', 'A') == {'em': 100., 'f1': 0.}


def test_paired_binary_exact_counts_and_no_pseudoreplication():
    a = {'q1': 100., 'q2': 0., 'q3': 100.}
    b = {'q1': 0., 'q2': 100., 'q3': 100.}
    r = paired(a, b, binary=True, draws=100)
    assert (r['wins'], r['losses'], r['ties']) == (1, 1, 1)
    assert r['exact_mcnemar_p'] == 1
    with pytest.raises(ValueError): paired(a, {'q1': 0.}, binary=True)


def test_eight_discordant_wins_have_exact_binomial_pvalue():
    a = {str(i): 100. for i in range(8)}
    b = {str(i): 0. for i in range(8)}
    assert paired(a, b, binary=True, draws=100)['exact_mcnemar_p'] == pytest.approx(2/256)


def test_readout_interaction_preserves_four_cell_question_pairing():
    # New communication effect 10pp; old communication effect 30pp.
    r = interaction({'q': 70.}, {'q': 60.}, {'q': 50.}, {'q': 20.}, draws=100)
    assert r['delta_pp'] == -20
    assert r['ci95_pp'] == [-20., -20.]
    with pytest.raises(ValueError): interaction({'a': 1}, {'b': 1}, {'a': 1}, {'a': 1})


def test_missing_or_duplicate_rows_and_arms_are_rejected():
    row = {'id': 'q', 'conditions': {'P_NoComm': {'responses': responses('a', 'b', 'c')}}}
    with pytest.raises(ValueError): validate_rows([row, row], {'q': {}}, ['P_NoComm'])
    with pytest.raises(ValueError): validate_rows([row], {'q': {}}, ['P_NoComm', 'P_Broadcast'])


def test_excluding_zero_score_abstentions_preserves_substantive_winners():
    for answers in itertools.product(['noanswer', '', 'yes', 'no', 'Paris', 'Lyon'], repeat=3):
        rs = responses(*answers)
        old = aggregate(rs, 'original'); new = aggregate(rs, 'abstention_aware')
        if old and normalize(old) != 'noanswer':
            assert old == new
        for reference in ['Paris', 'Lyon', 'yes', 'no']:
            assert score_answer(new, reference)['em'] >= score_answer(old, reference)['em']


def test_abstention_aware_rule_is_not_valid_as_universal_unanswerability_policy():
    rs = responses('noanswer', 'Paris', 'noanswer')
    assert score_answer(aggregate(rs, 'original'), 'noanswer')['em'] == 100
    assert score_answer(aggregate(rs, 'abstention_aware'), 'noanswer')['em'] == 0
