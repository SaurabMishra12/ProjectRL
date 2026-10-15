"""Historical physical request references, never a sum of logical arm costs."""
from __future__ import annotations
import json
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
from .analysis import ROOT, WEB, ARMS, load_inputs, save, table, sha


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def summarize_keys(keys, attempts):
    chosen = [a for a in attempts.values() if a['key'] in keys and a.get('dispatched')]
    successes = [a for a in chosen if a.get('success')]
    missing_input = sum(a.get('usage', {}).get('prompt_tokens') is None for a in successes)
    missing_output = sum(a.get('usage', {}).get('completion_tokens') is None for a in successes)
    return {'referenced_unique_request_keys': len(keys), 'documented_attempts': len(chosen),
        'documented_successful_attempts': len(successes),
        'uncertain_execution_attempts': sum(a.get('uncertain', False) for a in chosen),
        'observed_input_tokens_lower_bound': sum(a.get('usage', {}).get('prompt_tokens') or 0 for a in successes),
        'observed_output_tokens_lower_bound': sum(a.get('usage', {}).get('completion_tokens') or 0 for a in successes),
        'successful_attempts_missing_input_usage': missing_input,
        'successful_attempts_missing_output_usage': missing_output,
        'usage_complete': not (missing_input or missing_output or any(a.get('uncertain') for a in chosen))}


def main():
    rows, gold, raw, _ = load_inputs()
    arm_keys = {arm: {key for row in rows for key in row['conditions'][arm]['cost']['call_keys']} for arm in ARMS}
    all_keys = set().union(*arm_keys.values()); cache = raw.parent / 'inference/responses'
    attempts = {}; hashes = {}; cached = {}; source_labels = Counter()
    def attempt(aid, key):
        if aid in attempts and attempts[aid]['key'] != key: raise ValueError('Attempt ID maps to multiple request keys')
        return attempts.setdefault(aid, {'key': key, 'dispatched': False, 'success': False, 'usage': {}, 'sources': []})
    provenance = json.loads((ROOT.parent / 'offline/provenance.json').read_text())
    ledgers = [Path(p) for p in provenance['source_artifacts'] if p.endswith('/calls.jsonl')]
    for path in ledgers:
        if sha(path) != provenance['source_artifacts'][str(path)]: raise ValueError('Historical ledger changed')
        hashes[str(path)] = sha(path)
        for line in path.read_text().splitlines():
            event = json.loads(line); key = event.get('key'); aid = event.get('attempt_id')
            if key not in all_keys or not aid: continue
            a = attempt(aid, key); a['sources'].append({'path': str(path), 'event': event['event']})
            if event['event'] == 'started': a['dispatched'] = True
            if event['event'] == 'success': a['success'] = True; a['usage'].update(event.get('usage') or {})
            if event['event'] == 'uncertain': a['uncertain'] = True
            if event['event'] == 'rejected': a['rejected'] = True
        for kind in ['dispatch', 'receipt']:
            for meta in (path.parent / 'network_attempts').glob('*.' + kind + '.json'):
                event = json.loads(meta.read_text()); key = event.get('key')
                if key not in all_keys: continue
                hashes[str(meta)] = sha(meta); a = attempt(event['attempt_id'], key)
                a['sources'].append({'path': str(meta), 'event': kind})
                if kind == 'dispatch': a['dispatched'] = True
                elif event['http_status'] == 200: a['success'] = True
    for key in sorted(all_keys):
        path = cache / (key + '.json'); saved = json.loads(path.read_text()); hashes[str(path)] = sha(path)
        if saved['key'] != key or saved['content_hash'] != digest({k: v for k, v in saved.items() if k != 'content_hash'}):
            raise ValueError('Invalid frozen cache content')
        cached[key] = saved
        a = attempt(saved['attempt_id'], key)
        a['cache_success_evidence'] = True
        a['success'] = True
        a['usage'].update(saved['response'].get('usage') or {})
        a['sources'].append({'path': str(path), 'event': 'cached_provider_response'})
    for row in rows:
        for arm, c in row['conditions'].items():
            if len(c['cost']['call_keys']) != c['cost']['logical_calls']: raise ValueError('Logical call record inconsistent')
            if not set(r['call_key'] for r in c['responses']) <= set(c['cost']['call_keys']): raise ValueError('Final responses missing from condition cost')
            for node, r in enumerate(c['responses']):
                saved = cached[r['call_key']]
                if saved['metadata']['node'] != node: raise ValueError('Agent ordering differs from request node')
                # Refusal parser substitutes an explicit empty response; raw text
                # remains the provider content (or empty if content is null).
                content = saved['response']['choices'][0]['message'].get('content') or ''
                if r['raw_text'] != content: raise ValueError('Frozen trace differs from cached provider text')
                if not r.get('provider_refusal'):
                    cleaned = content.strip()
                    if cleaned.startswith('```'):
                        import re
                        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned)
                    if json.loads(cleaned)['answer'].strip() != r['answer']: raise ValueError('Parsed answer differs from original provider answer')
                source_labels['final_response_references_checked'] += 1
    costs = []
    for arm, keys in arm_keys.items():
        s = summarize_keys(keys, attempts)
        costs.append({'arm': arm, **s,
            'logical_condition_calls_not_physical_spend': sum(r['conditions'][arm]['cost']['logical_calls'] for r in rows),
            'logical_input_tokens_complete_or_null': sum(r['conditions'][arm]['cost']['input_tokens'] for r in rows) if all(r['conditions'][arm]['cost']['input_tokens'] is not None for r in rows) else None,
            'logical_output_tokens_complete_or_null': sum(r['conditions'][arm]['cost']['output_tokens'] for r in rows) if all(r['conditions'][arm]['cost']['output_tokens'] is not None for r in rows) else None,
            'delivered_message_tokens': sum(r['conditions'][arm]['payload_tokens'] for r in rows),
            'cached_successes_without_documented_dispatch': sum(not attempts[cached[k]['attempt_id']]['dispatched'] for k in keys),
            'nonadditive_shared_request_references': True})
    matched = []
    for regime in ['P', 'R']:
        for seed in [7, 19, 31]:
            arm, baseline = f'{regime}_Learned_seed{seed}', f'{regime}_MatchedRandom_seed{seed}'
            same_edges = same_quota = same_recipients = same_calls = 0
            for row in rows:
                a, b = row['conditions'][arm], row['conditions'][baseline]
                same_edges += len(a['action']['edges']) == len(b['action']['edges'])
                same_quota += a['action']['quota'] == b['action']['quota']
                same_recipients += len({t for s, t in a['action']['edges']}) == len({t for s, t in b['action']['edges']})
                same_calls += a['cost']['logical_calls'] == b['cost']['logical_calls']
            matched.append({'arm': arm, 'baseline': baseline, 'n': len(rows), 'same_edge_count': same_edges,
                'same_quota': same_quota, 'same_recipient_count': same_recipients, 'same_standalone_calls': same_calls})
    summary = {'scope': 'Nonadditive historical physical attempts associated with referenced keys; not exclusive per-arm spend',
        'union': summarize_keys(all_keys, attempts), 'arms': costs, 'matched_resource_checks': matched,
        'new_requests': 0, 'new_paid_completions': 0, 'raw_trace_response_checks': dict(source_labels),
        'source_hashes': hashes,
        'cached_successes_without_dispatch_in_union': sum(not attempts[cached[k]['attempt_id']]['dispatched'] for k in all_keys),
        'unknown_execution_cost_is_not_imputed': True,
        'training_and_unreferenced_policy_search_cost_not_allocated_to_test_arms': True,
        'readout_changes_add_no_requests_or_tokens': True}
    save(ROOT / 'results/physical_accounting.json', summary); table(ROOT / 'results/physical_accounting.csv', costs)
    save(ROOT / 'results/attempt_inventory.json', attempts)
    save(ROOT / 'results/referenced_request_keys.json', {a: sorted(k) for a, k in arm_keys.items()})
    print(json.dumps({'union': summary['union'], 'checks': dict(source_labels), 'unmatched_successes': summary['cached_successes_without_dispatch_in_union'], 'matched': matched}, indent=2))


if __name__ == '__main__': main()
