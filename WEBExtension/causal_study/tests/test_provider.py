"""External HTTP is mocked; budgeting, files, hashes and resume logic are real."""
import json
import tempfile
from pathlib import Path

import pytest

from WEBExtension.causal_study.provider import BudgetBackend, BudgetExceeded, UncertainRequest, ProviderRejected


CONFIG = {'model': 'frozen-test', 'model_revision': 'test-v1', 'base_url': 'https://example.invalid/openai/v1',
          'temperature': 0.0, 'max_tokens': 100, 'context_limit': 32768,
          'max_completions': 150, 'max_attempts': 180, 'min_interval_seconds': 0,
          'max_retries_per_request': 2, 'max_estimated_tokens': 2000000}
MESSAGES = [{'role': 'user', 'content': 'A gold-free question'}]


def response(answer='Paris', finish_reason='stop'):
    return {'id': 'provider-1', 'model': 'frozen-test', 'choices': [{'finish_reason': finish_reason,
            'message': {'content': json.dumps({'answer': answer, 'citations': []})}}],
            'usage': {'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18}}


class Transport:
    def __init__(self, results): self.results = iter(results); self.requests = []
    def __call__(self, request, request_id):
        self.requests.append((request, request_id))
        result = next(self.results)
        if isinstance(result, Exception): raise result
        return result


def backend(path, transport, **overrides):
    return BudgetBackend(path, {**CONFIG, **overrides}, transport=transport, sleep=lambda _: None)


def test_success_cache_resume_and_usage_are_physical(tmp_path):
    transport = Transport([(200, {'x-request-id': 'request-1'}, response())])
    b = backend(tmp_path, transport)
    result = b.complete(MESSAGES, {'question_id': 'q1'})
    assert result['raw']['usage']['prompt_tokens'] == 11
    resumed = backend(tmp_path, Transport([]))
    assert resumed.complete(MESSAGES, {'question_id': 'q1'})['cache_hit']
    assert resumed.summary()['successful_completions'] == 1
    assert resumed.summary()['http_attempts'] == 1
    assert resumed.summary()['input_tokens'] == 11
    assert len(list((tmp_path/'requests').glob('*.json'))) == 1


def test_success_ceiling_blocks_next_distinct_request(tmp_path):
    t = Transport([(200, {}, response())])
    b = backend(tmp_path, t, max_completions=1)
    b.complete(MESSAGES)
    with pytest.raises(BudgetExceeded): b.complete([{'role':'user','content':'another'}])
    assert len(t.requests) == 1
    assert b.summary()['http_attempts'] == 1


def test_attempt_ceiling_includes_429_and_honors_delay(tmp_path):
    t = Transport([(429, {'retry-after': '4'}, {'error': {'code': '429'}}), (200, {}, response())])
    sleeps = []
    b = BudgetBackend(tmp_path, {**CONFIG, 'max_attempts': 1}, transport=t, sleep=sleeps.append)
    with pytest.raises(BudgetExceeded): b.complete(MESSAGES)
    assert b.summary()['http_attempts'] == 1
    assert b.summary()['successful_completions'] == 0
    assert 4 in sleeps


def test_retry_reuses_prompt_but_has_distinct_persistent_attempt_id(tmp_path):
    t = Transport([(429, {'retry-after-ms': '1000'}, {'error': {'code': '429'}}), (200, {}, response())])
    b = backend(tmp_path, t)
    b.complete(MESSAGES)
    assert b.summary()['http_attempts'] == 2
    assert b.summary()['retries'] == 1
    assert t.requests[0][0] == t.requests[1][0]
    assert t.requests[0][1] != t.requests[1][1]


def test_timeout_blocks_all_further_calls_and_resume(tmp_path):
    t = Transport([TimeoutError('execution status unknown')])
    b = backend(tmp_path, t)
    with pytest.raises(UncertainRequest): b.complete(MESSAGES)
    with pytest.raises(UncertainRequest): b.complete(MESSAGES)
    with pytest.raises(UncertainRequest): backend(tmp_path, Transport([]))
    assert len(t.requests) == 1


def test_http_500_is_uncertain_and_not_retried(tmp_path):
    b = backend(tmp_path, Transport([(500, {}, {'error': {'code': 'server'}})]))
    with pytest.raises(UncertainRequest): b.complete(MESSAGES)
    assert b.summary()['http_attempts'] == 1
    assert b.summary()['uncertain_requests'] == 1


def test_malformed_success_still_spends_completion_and_is_not_regenerated(tmp_path):
    raw = response(); raw['choices'][0]['message']['content'] = 'not JSON'
    b = backend(tmp_path, Transport([(200, {}, raw)]))
    assert b.complete(MESSAGES)['raw'] == raw
    assert b.summary()['successful_completions'] == 1
    assert backend(tmp_path, Transport([])).complete(MESSAGES)['cache_hit']


def test_refusal_completion_body_on_400_counted_conservatively(tmp_path):
    raw = response('', 'content_filter')
    b = backend(tmp_path, Transport([(400, {}, raw)]))
    assert b.complete(MESSAGES)['raw']['choices'][0]['finish_reason'] == 'content_filter'
    assert b.summary()['successful_completions'] == 1


def test_rejected_auth_never_retried(tmp_path):
    b = backend(tmp_path, Transport([(401, {}, {'error': {'code': 'Unauthorized'}})]))
    with pytest.raises(ProviderRejected): b.complete(MESSAGES)
    assert b.summary()['http_attempts'] == 1


def test_approved_caps_cannot_be_raised_by_config(tmp_path):
    for overrides in [{'max_completions':151}, {'max_attempts':181}]:
        with pytest.raises(ValueError): backend(tmp_path, Transport([]), **overrides)


def test_changed_frozen_config_and_tampered_cache_rejected(tmp_path):
    b = backend(tmp_path, Transport([(200, {}, response())]))
    r = b.complete(MESSAGES)
    with pytest.raises(ValueError): backend(tmp_path, Transport([]), model='other')
    p = tmp_path/'responses'/(r['request_hash']+'.json')
    x = json.loads(p.read_text()); x['raw']['model'] = 'tampered'; p.write_text(json.dumps(x))
    with pytest.raises(ValueError): b.complete(MESSAGES)


def test_crash_after_response_save_recovers_without_http(tmp_path):
    b = backend(tmp_path, Transport([(200, {}, response())]))
    r = b.complete(MESSAGES)
    p = tmp_path/'ledger.jsonl'
    lines = p.read_text().splitlines(); p.write_text(lines[0]+'\n')
    restored = backend(tmp_path, Transport([]))
    assert restored.complete(MESSAGES)['cache_hit']
    assert restored.summary()['successful_completions'] == 1


def test_context_and_token_safety_gate_runs_before_dispatch(tmp_path):
    t = Transport([])
    b = backend(tmp_path,t,context_limit=10)
    with pytest.raises(BudgetExceeded): b.complete(MESSAGES)
    assert b.summary()['http_attempts'] == 0


def test_assignment_namespace_does_not_change_model_prompt(tmp_path):
    t=Transport([(200,{},response()),(200,{},response())])
    b=backend(tmp_path,t)
    a=b.complete(MESSAGES,{'placement':'colocated'})
    z=b.complete(MESSAGES,{'placement':'distributed'})
    assert a['request_hash'] != z['request_hash']
    assert t.requests[0][0] == t.requests[1][0]
    assert b.summary()['successful_completions'] == 2


def test_actual_150_completion_boundary_persists_after_restart(tmp_path):
    b=backend(tmp_path,Transport([(200,{},response())]*150))
    for i in range(150): b.complete([{'role':'user','content':str(i)}])
    restored=backend(tmp_path,Transport([]))
    with pytest.raises(BudgetExceeded): restored.complete(MESSAGES)
    assert restored.summary()['successful_completions']==150
    assert restored.summary()['http_attempts']==150


def test_actual_180_attempt_boundary_includes_every_rejection(tmp_path):
    b=backend(tmp_path,Transport([(429,{}, {'error':{'code':'429'}})]*180))
    for i in range(60):
        with pytest.raises(ProviderRejected): b.complete([{'role':'user','content':str(i)}])
    with pytest.raises(BudgetExceeded): backend(tmp_path,Transport([])).complete(MESSAGES)
    assert b.summary()['http_attempts']==180
    assert b.summary()['successful_completions']==0
