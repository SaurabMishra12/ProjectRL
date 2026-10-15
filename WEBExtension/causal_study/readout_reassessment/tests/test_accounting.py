from WEBExtension.causal_study.readout_reassessment.accounting import summarize_keys


def test_shared_requests_and_retries_are_counted_once_per_union():
    attempts = {
        '1': {'key': 'a', 'dispatched': True, 'success': True, 'usage': {'prompt_tokens': 10, 'completion_tokens': 2}},
        '2': {'key': 'b', 'dispatched': True, 'success': False, 'usage': {}, 'uncertain': True},
        '3': {'key': 'b', 'dispatched': True, 'success': True, 'usage': {'prompt_tokens': 20, 'completion_tokens': 3}},
        '4': {'key': 'c', 'dispatched': True, 'success': True, 'usage': {'prompt_tokens': 30, 'completion_tokens': 4}},
    }
    ab = summarize_keys({'a', 'b'}, attempts)
    bc = summarize_keys({'b', 'c'}, attempts)
    union = summarize_keys({'a', 'b', 'c'}, attempts)
    assert ab['documented_attempts'] == 3 and bc['documented_attempts'] == 3
    assert union['documented_attempts'] == 4
    assert union['observed_input_tokens_lower_bound'] == 60
    assert union['uncertain_execution_attempts'] == 1


def test_missing_usage_is_unknown_and_not_fabricated_as_complete_zero():
    a = {'x': {'key': 'a', 'dispatched': True, 'success': True,
               'usage': {'prompt_tokens': 10, 'completion_tokens': None}}}
    r = summarize_keys({'a'}, a)
    assert r['observed_output_tokens_lower_bound'] == 0
    assert r['successful_attempts_missing_output_usage'] == 1
    assert r['usage_complete'] is False
