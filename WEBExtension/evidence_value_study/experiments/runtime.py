"""Durable offline E9 harness. No credentials, HTTP client, or live switch.

Reuse the existing study's canonical hashing, directory-fsynced atomic writes,
and exceptions. Do not inherit its E6 authorization or mutate its ledger.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import uuid
from pathlib import Path
from filelock import FileLock
from WEBExtension.causal_study.provider import atomic_json, canonical, digest, BudgetExceeded, UncertainRequest
from .design import assign, initial_messages, update_messages, baseline_messages, parse_response, peer_packet


class ScientificGateClosed(RuntimeError):
    pass


def execute_paid():
    raise ScientificGateClosed('Paid inference unavailable: independent human necessity qualification and release review are required')


class DurableMockBackend:
    """One immutable mock namespace; persistent conservative exposure reservations.

    This class is intentionally mock-only. Injected transports are testing code;
    the CLI only supplies the deterministic local mock_transport below.
    """
    def __init__(self, directory, config, *, transport, token_count):
        if config.get('mode') != 'MOCK':
            raise ScientificGateClosed('This implementation only dispatches MOCK requests')
        for key, ceiling in [('max_completions', 330), ('max_attempts', 390),
                             ('max_input_tokens', 300000), ('max_output_tokens', 140000)]:
            if type(config.get(key)) is not int or not 0 < config[key] <= ceiling:
                raise ValueError('Authorized ceiling: ' + key)
        if type(config.get('max_tokens')) is not int or not 0 < config['max_tokens'] <= 8192:
            raise ValueError('Invalid output cap')
        if not 0 <= config.get('max_retries_per_slot', 2) <= 2:
            raise ValueError('At most two explicit429 retries per slot')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config, self.transport, self.token_count = dict(config), transport, token_count
        self.ledger = self.directory / 'ledger.jsonl'
        self.lock = FileLock(str(self.directory / 'runtime.lock'), timeout=0)
        with self.lock:
            p = self.directory / 'frozen_config.json'
            if p.exists() and json.loads(p.read_text()) != self.config:
                raise ValueError('Immutable configuration changed; do not create a replacement ledger')
            if not p.exists():
                atomic_json(p, self.config)
            self._recover()

    def _records(self):
        try:
            return [json.loads(s) for s in self.ledger.read_text().splitlines()] if self.ledger.exists() else []
        except ValueError:
            raise UncertainRequest('Damaged ledger requires reconciliation') from None

    def _append(self, event):
        with self.ledger.open('a') as f:
            f.write(canonical({**event, 'utc': dt.datetime.now(dt.timezone.utc).isoformat()}) + '\n')
            f.flush()
            os.fsync(f.fileno())
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _recover(self):
        rows = self._records()
        closed = {r['attempt_id'] for r in rows if r['event'] in ('success', 'rejected')}
        # A crash can follow the fsynced success event but precede cache creation.
        # Restore from the raw attempt; a missing raw body is uncertain, never a rerun.
        for row in rows:
            if row['event'] != 'success':
                continue
            cache = self.directory / 'responses' / (row['slot_hash'] + '.json')
            if not cache.exists():
                path = self.directory / 'attempts' / (row['attempt_id'] + '.json')
                if not path.exists():
                    raise UncertainRequest('Successful ledger entry lacks raw response')
                saved = json.loads(path.read_text())
                self._validate(saved)
                if saved['slot_hash'] != row['slot_hash']:
                    raise UncertainRequest('Successful response identity mismatch')
                atomic_json(cache, saved)
        for row in rows:
            if row['event'] != 'started' or row['attempt_id'] in closed:
                continue
            p = self.directory / 'attempts' / (row['attempt_id'] + '.json')
            if not p.exists():
                raise UncertainRequest('Unresolved execution; halt across restart')
            saved = json.loads(p.read_text())
            self._validate(saved)
            if saved['slot_hash'] != row['slot_hash']:
                raise UncertainRequest('Recovered attempt identity mismatch')
            self._finish(saved)

    def _validate(self, saved):
        if saved.get('sha256') != digest({k: v for k, v in saved.items() if k != 'sha256'}):
            raise ValueError('Response checksum mismatch')

    def _finish(self, saved):
        status, raw = saved['http_status'], saved['raw']
        # Count any completion-shaped response conservatively, even malformed/refused.
        if status in (200, 201) or (isinstance(raw, dict) and raw.get('choices')):
            self._append({'event': 'success', 'attempt_id': saved['attempt_id'],
                          'slot_hash': saved['slot_hash'], 'usage': raw.get('usage') if isinstance(raw, dict) else None})
            atomic_json(self.directory / 'responses' / (saved['slot_hash'] + '.json'), saved)
            return 'success'
        if status == 429 and isinstance(raw, dict) and isinstance(raw.get('error'), dict) and not raw.get('choices'):
            self._append({'event': 'rejected', 'attempt_id': saved['attempt_id'],
                          'slot_hash': saved['slot_hash'], 'http_status': status})
            return 'rejected'
        raise UncertainRequest('Unconfirmed generation status; no automatic retry')

    def summary(self):
        rows = self._records()
        starts = [r for r in rows if r['event'] == 'started']
        successes = [r for r in rows if r['event'] == 'success']
        rejected = [r for r in rows if r['event'] == 'rejected']
        closed = {r['attempt_id'] for r in successes + rejected}
        usage = [r.get('usage') or {} for r in successes]
        valid_in = [u.get('prompt_tokens') for u in usage if type(u.get('prompt_tokens')) is int]
        valid_out = [u.get('completion_tokens') for u in usage if type(u.get('completion_tokens')) is int]
        return {'mode': 'MOCK', 'real_provider_calls': 0, 'successful_completions': len(successes),
                'http_attempts': len(starts), 'retries': len(starts) - len({r['slot_hash'] for r in starts}),
                'rejected_attempts': len(rejected), 'uncertain_requests': sum(r['attempt_id'] not in closed for r in starts),
                'input_tokens': sum(valid_in) if len(valid_in) == len(successes) else None,
                'output_tokens': sum(valid_out) if len(valid_out) == len(successes) else None,
                'missing_usage_records': sum(not all(type(u.get(k)) is int for k in ('prompt_tokens', 'completion_tokens')) for u in usage),
                'reserved_input_tokens': sum(r['input_reserve'] for r in starts),
                'reserved_output_tokens': sum(r['output_reserve'] for r in starts),
                'monetary_cost': None, 'usage_status': 'SYNTHETIC_MOCK_USAGE_ONLY'}

    def complete(self, slot, messages):
        with self.lock:
            self._recover()
            key = digest({'slot': slot, 'config': self.config})
            request = {k: self.config[k] for k in ('model', 'temperature', 'max_tokens')}
            request['messages'] = messages
            p = self.directory / 'requests' / (key + '.json')
            record = {'slot': slot, 'request': request, 'slot_hash': key, 'prompt_sha256': digest(messages)}
            if p.exists() and json.loads(p.read_text()) != record:
                raise ValueError('Slot reused with different model-visible payload')
            if not p.exists():
                atomic_json(p, record)
            cached = self.directory / 'responses' / (key + '.json')
            if cached.exists():
                saved = json.loads(cached.read_text())
                self._validate(saved)
                if (saved.get('slot') != slot or saved.get('slot_hash') != key or
                    saved.get('request') != request or saved.get('prompt_sha256') != digest(messages)):
                    raise ValueError('Cached response identity mismatch')
                events = self._records()
                if not any(r['event'] == 'success' and r['attempt_id'] == saved['attempt_id'] and
                           r['slot_hash'] == key for r in events):
                    raise ValueError('Cached response identity lacks success event')
                return saved
            reserve_in = sum(self.token_count(m['content']) for m in messages) + 128
            reserve_out = self.config['max_tokens']
            if reserve_in + reserve_out > self.config['context_limit']:
                raise BudgetExceeded('Context bound; no silent source truncation')
            while True:
                state = self.summary()
                if state['successful_completions'] >= self.config['max_completions'] or state['http_attempts'] >= self.config['max_attempts']:
                    raise BudgetExceeded('Persistent call ceiling')
                if (state['reserved_input_tokens'] + reserve_in > self.config['max_input_tokens'] or
                    state['reserved_output_tokens'] + reserve_out > self.config['max_output_tokens']):
                    raise BudgetExceeded('Persistent token reservation ceiling')
                previous = sum(r['event'] == 'started' and r['slot_hash'] == key for r in self._records())
                if previous > self.config.get('max_retries_per_slot', 2):
                    raise BudgetExceeded('Per-slot retry ceiling persists across restart')
                attempt = uuid.uuid4().hex
                self._append({'event': 'started', 'attempt_id': attempt, 'slot_hash': key,
                              'slot': slot, 'input_reserve': reserve_in, 'output_reserve': reserve_out})
                start = time.perf_counter()
                try:
                    status, headers, raw = self.transport(request, attempt)
                except Exception as exc:
                    self._append({'event': 'uncertain', 'attempt_id': attempt, 'error_type': type(exc).__name__})
                    raise UncertainRequest('Unknown execution; exception details suppressed') from None
                saved = {'mode': 'MOCK', 'slot': slot, 'slot_hash': key, 'attempt_id': attempt,
                         'http_status': status, 'raw': raw, 'headers': headers,
                         'latency_seconds': time.perf_counter() - start, 'request': request,
                         'prompt_sha256': digest(messages), 'response_sha256': digest(raw)}
                saved['sha256'] = digest(saved)
                atomic_json(self.directory / 'attempts' / (attempt + '.json'), saved)
                if self._finish(saved) == 'success':
                    return saved


def _parsed(saved, q, allowed):
    raw = saved['raw']
    choices = raw.get('choices') if isinstance(raw, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get('message')
    if not isinstance(message, dict):
        message = {}
    content = message.get('content')
    if not isinstance(content, str):
        content = ''
    return parse_response(content, q, allowed,
                          refusal=message.get('refusal'), finish_reason=choice.get('finish_reason'))


def run_question(q, index, backend):
    plan = assign(q, index)
    initial, cells, calls = {}, {}, []
    for placement in plan['placement_order']:
        initial[placement] = []
        for node in range(3):
            slot = f'{q["id"]}/initial/{placement}/{node}'
            saved = backend.complete(slot, initial_messages(q, plan, placement, node))
            initial[placement].append(_parsed(saved, q, plan['placements'][placement][node]))
            calls.append({'slot': slot, 'slot_hash': saved['slot_hash'], 'response_sha256': saved['response_sha256']})
    packet = peer_packet(initial['complete'][plan['sender']])
    for placement in plan['placement_order']:
        own = initial[placement][plan['receiver']]
        for update in plan['update_order']:
            slot = f'{q["id"]}/update/{placement}/{update}'
            messages = update_messages(q, plan, placement, own, packet if update == 'peer' else None)
            saved = backend.complete(slot, messages)
            # Receiver may cite local text or quotes actually supplied by the packet,
            # never arbitrary unseen portions of the sender's full document.
            visible = {d['id']: dict(d) for d in q['documents'] if d['id'] in plan['placements'][placement][plan['receiver']]}
            if update == 'peer':
                for fact in packet['evidence']:
                    if fact['document_id'] not in visible:
                        visible[fact['document_id']] = {'id': fact['document_id'], 'text': '', 'visible_spans': []}
                    if fact['document_id'] not in plan['placements'][placement][plan['receiver']]:
                        visible[fact['document_id']]['visible_spans'].append(fact['quote'])
            cells[placement + '_' + update] = _parsed(saved, {'documents': list(visible.values())}, list(visible))
            calls.append({'slot': slot, 'slot_hash': saved['slot_hash'], 'response_sha256': saved['response_sha256']})
    saved = backend.complete(f'{q["id"]}/baseline', baseline_messages(q))
    baseline = _parsed(saved, q, [d['id'] for d in q['documents']])
    calls.append({'slot': f'{q["id"]}/baseline', 'slot_hash': saved['slot_hash'], 'response_sha256': saved['response_sha256']})
    result = {'mode': 'MOCK', 'question_id': q['id'], 'plan': plan, 'initial': initial,
              'packet': packet, 'cells': cells, 'baseline': baseline, 'calls': calls,
              'semantic_missing_fact_delivery': 'NOT_ANNOTATED', 'scientific_outcomes': 'NOT_RUN'}
    result['sha256'] = digest(result)
    atomic_json(backend.directory / 'questions' / (digest(q['id']) + '.json'), result)
    return result


def mock_transport(request, attempt):
    """Local answer/quote fixture; no synthetic accuracy presented as science."""
    payload = json.loads(request['messages'][1]['content'])
    source = payload['documents'][0]
    reply = {'answer': 'MOCK_ONLY_NOT_AN_INFERENCE',
             'evidence': [{'document_id': source['id'], 'quote': source['text'][:80]}], 'uncertain': True}
    return 200, {'x-request-id': 'mock-' + attempt}, {
        'id': 'mock-' + attempt, 'model': request['model'],
        'choices': [{'message': {'content': canonical(reply)}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'status': 'SYNTHETIC_NOT_BILLED'}}
