"""Reproduce E9 offline validation. No CLI action can invoke a real provider."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from WEBExtension.causal_study.provider import atomic_json, digest, canonical
from WEBExtension.causal_study.run import tokenizer
from .design import VERSION, assign, initial_messages, update_messages, baseline_messages
from .runtime import DurableMockBackend, mock_transport, run_question, execute_paid
from ..analysis.statistics import power_grid, PUBLISHER_METRIC

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Frozen file differs; preserve original and review amendment: ' + str(path))
    else:
        atomic_json(path, value)


def freeze(qs, tok):
    old = json.loads((ROOT.parent / 'causal_study/frozen/config.json').read_text())
    config = {k: old[k] for k in ('provider', 'base_url', 'model', 'model_revision', 'api_version', 'temperature', 'context_limit')}
    config.update({'mode': 'MOCK', 'max_tokens': 384, 'max_completions': 330, 'max_attempts': 390,
                   'max_input_tokens': 300000, 'max_output_tokens': 140000, 'max_retries_per_slot': 2,
                   'tokenizer_identity': tok.identity, 'candidate_records_sha256': digest(qs),
                   'paid_authorized': False, 'protocol': VERSION})
    dependencies = [ROOT / 'experiments/design.py', ROOT / 'experiments/runtime.py',
                    ROOT / 'experiments/run.py', ROOT / 'analysis/statistics.py',
                    ROOT.parent / 'causal_study/provider.py', ROOT.parent / 'causal_study/statistics.py',
                    ROOT.parent / 'causal_study/run.py', ROOT.parent / 'protocol.py', PUBLISHER_METRIC,
                    ROOT / 'frozen/private_candidates.json']
    hashes = {str(p.relative_to(REPO)): file_hash(p) for p in dependencies}
    assignments = [assign(q, i) for i, q in enumerate(qs)]
    write_once(ROOT / 'configs/e9_mock.json', config)
    write_once(ROOT / 'frozen/e9_assignments.json', assignments)
    write_once(ROOT / 'frozen/execution_source_hashes.json', hashes)
    write_once(ROOT / 'frozen/e9_manifest.json', {
        'status': 'CANDIDATE_MOCK_FREEZE_NOT_QUALIFIED_PILOT', 'protocol': VERSION,
        'candidate_ids': [q['id'] for q in qs], 'candidate_records_sha256': digest(qs),
        'assignments_sha256': digest(assignments), 'config_sha256': digest(config),
        'source_hashes_sha256': digest(hashes), 'questions': 30, 'completion_slots': 330,
        'successful_completion_ceiling': 330, 'http_attempt_ceiling': 390,
        'paid_release_status': 'NOT_AVAILABLE_INDEPENDENT_ANNOTATION_REQUIRED'})
    return config


def _payload_audit(q, result, directory, tok):
    plans = result['plan']
    receiver, sender = plans['receiver'], plans['sender']
    full = {d['id'] for d in q['documents']}
    assert len(result['calls']) == 11
    assert set(result['cells']) == {'complete_private', 'complete_peer', 'missing_private', 'missing_peer'}
    for placement in ('complete', 'missing'):
        assert set().union(*(set(x) for x in plans['placements'][placement])) == full
    assert plans['placements']['complete'][sender] == plans['placements']['missing'][sender]
    payloads, token_sizes = {}, {}
    for call in result['calls']:
        req = json.loads((directory / 'requests' / (call['slot_hash'] + '.json')).read_text())
        payload = json.loads(req['request']['messages'][1]['content'])
        assert set(payload) <= {'question', 'documents', 'initial_state', 'information_packet'}
        assert payload['question'] == q['question']
        assert all(set(d) == {'id', 'title', 'text'} for d in payload['documents'])
        payloads[call['slot']] = payload
        token_sizes[call['slot']] = sum(tok.count(m['content']) for m in req['request']['messages'])
        if '/initial/' in call['slot']:
            parts = call['slot'].split('/')
            place, node = parts[-2], int(parts[-1])
            assert [d['id'] for d in payload['documents']] == plans['placements'][place][node]
        if call['slot'].endswith('/baseline'):
            assert {d['id'] for d in payload['documents']} == full
    for place in ('complete', 'missing'):
        private = payloads[f'{q["id"]}/update/{place}/private']
        peer = payloads[f'{q["id"]}/update/{place}/peer']
        assert private['initial_state'] == peer['initial_state']
        assert private['documents'] == peer['documents']
        assert private['information_packet']['evidence'] == []
        assert private['information_packet']['answer'] is None
        assert {d['id'] for d in private['documents']} == set(plans['placements'][place][receiver])
    assert payloads[f'{q["id"]}/update/complete/peer']['information_packet'] == payloads[f'{q["id"]}/update/missing/peer']['information_packet']
    return token_sizes


def dry_run():
    all_candidates = json.loads((ROOT / 'frozen/private_candidates.json').read_text())
    if len(all_candidates) < 40:
        raise ValueError('Thirty priority candidates plus ten fixed reserve candidates required')
    qs = all_candidates[:30]
    if len({q['id'] for q in qs}) != 30:
        raise ValueError('Question pairing fails')
    tok = tokenizer()
    config = freeze(qs, tok)
    directory = ROOT / 'dry_run/runtime'
    backend = DurableMockBackend(directory, config, transport=mock_transport, token_count=tok.count)
    initial_attempts = backend.summary()['http_attempts']
    started = time.perf_counter()
    sizes = {}
    for i, q in enumerate(qs):
        result = run_question(q, i, backend)
        sizes.update(_payload_audit(q, result, directory, tok))
    before = backend.summary()
    for i, q in enumerate(qs):
        run_question(q, i, backend)
    after = backend.summary()
    assert before['http_attempts'] == after['http_attempts'] == 330
    assert before['successful_completions'] == 330 and len(sizes) == 330
    # Conservative planning envelope, not a guarantee of server framing or JSON
    # reserialization behavior: four own states counted twice plus two peer packets.
    empty_content = 0
    for i, q in enumerate(qs):
        plan = assign(q, i)
        for placement in ('complete', 'missing'):
            for node in range(3):
                empty_content += sum(tok.count(m['content']) for m in initial_messages(q, plan, placement, node))
            for peer in (None, {'answer': None, 'evidence': [], 'status': 'ABSTAINED'}):
                empty_content += sum(tok.count(m['content']) for m in update_messages(q, plan, placement, {'raw': ''}, peer))
        empty_content += sum(tok.count(m['content']) for m in baseline_messages(q))
    envelope = empty_content + 330 * 128 + 30 * (4 * 2 + 2) * config['max_tokens']
    projection = {'status': 'OFFLINE_PROJECTION_NOT_USAGE', 'prompt_content_tokens_mock_states': sum(sizes.values()),
                  'prompt_content_tokens_empty_states': empty_content,
                  'framing_allowance_per_request': 128,
                  'input_envelope_with_max_state_and_packet_tokens': envelope,
                  'input_ceiling': config['max_input_tokens'], 'output_ceiling': config['max_output_tokens'],
                  'successful_completion_output_cap_exposure': 330 * config['max_tokens'],
                  'all390_attempt_output_reservation': 390 * config['max_tokens'],
                  'envelope_fits_initial_input_ceiling': envelope <= config['max_input_tokens'],
                  'max_prompt_content_tokens_mock_state': max(sizes.values()),
                  'content_token_counts': sizes,
                  'limits': 'Envelope assumes<=2x serialized own output and <=1x packet output; serving framing/escaping not verified. Actual prompts must be reserved before dispatch. Retries reserve again and may stop before390 attempts.',
                  'monetary_tariff': 'NOT_AVAILABLE', 'projected_currency_cost': None}
    validation = {'status': 'PASS_MOCK_ONLY', 'candidate_questions': 30, 'qualified_questions': 0,
                  'mocked_completion_slots': 330, 'mocked_http_attempts': after['http_attempts'],
                  'new_mock_attempts_this_execution': after['http_attempts'] - initial_attempts,
                  'resume_added_requests': after['http_attempts'] - before['http_attempts'],
                  'real_provider_calls': 0, 'paid_completions': 0, 'scored_scientific_effects': False,
                  'same_packet_across_accessibility': True, 'same_state_within_accessibility': True,
                  'source_union_and_ownership_checked': True, 'payload_allowlists_checked': True,
                  'source_necessity_status': 'NOT_QUALIFIED', 'paid_execution': 'NOT_RUN',
                  'runtime_seconds_this_execution': time.perf_counter() - started,
                  'mock_usage_summary': after}
    atomic_json(ROOT / 'dry_run/validation.json', validation)
    atomic_json(ROOT / 'analysis/prompt_exposure.json', projection)
    atomic_json(ROOT / 'analysis/power_sensitivity.json', power_grid())
    atomic_json(ROOT / 'analysis/pilot_outcomes.json', {'status': 'NOT_RUN', 'assigned_qualified_questions': 0,
                'four_cell_F1_EM': None, 'primary_interaction': None, 'confidence_interval': None,
                'p_value': None, 'mechanism_diagnostics': 'NOT_AVAILABLE', 'full_evidence_RAG': 'NOT_RUN'})
    atomic_json(ROOT / 'paid_pilot/status.json', {'status': 'NOT_RUN', 'reason': 'Independent qualified human annotations and adjudication absent',
                'successful_completions': 0, 'http_attempts': 0, 'input_tokens': 0, 'output_tokens': 0,
                'retries': 0, 'uncertain_requests': 0, 'provider_refusals': 0,
                'monetary_cost': 0, 'monetary_cost_basis': 'No new inference requests dispatched',
                'model_deployment_planned': config['model'], 'immutable_model_snapshot': 'NOT_AVAILABLE',
                'smoke_test_55': 'NOT_RUN', 'remaining25': 'NOT_RUN'})
    atomic_json(ROOT / 'source_validation/runtime_environment.json', {'python': platform.python_version(),
                'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'transformers', 'tokenizers', 'filelock', 'pytest']},
                'tokenizer': tok.identity})
    print(json.dumps({k: v for k, v in validation.items() if k != 'mock_usage_summary'}, indent=2))


def verify_preservation():
    baseline = json.loads((ROOT / 'source_validation/historical_baseline.json').read_text())
    changed, missing = [], []
    for name, expected in baseline['files'].items():
        p = REPO / name
        if not p.is_file():
            missing.append(name)
        elif file_hash(p) != expected['sha256']:
            changed.append(name)
    result = {'status': 'PASS' if not changed and not missing else 'FAIL',
              'historical_files_checked': len(baseline['files']), 'changed': changed, 'missing': missing,
              'baseline_sha256': file_hash(ROOT / 'source_validation/historical_baseline.json'),
              'scope': 'All baseline regular files; excluded directories and symlinks listed in baseline'}
    atomic_json(ROOT / 'source_validation/historical_preservation_audit.json', result)
    print(json.dumps(result, indent=2))
    if changed or missing:
        raise ValueError('Historical preservation failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['dry-run', 'verify-preservation', 'paid'])
    action = parser.parse_args().action
    if action == 'dry-run':
        dry_run()
    elif action == 'verify-preservation':
        verify_preservation()
    else:
        execute_paid()
