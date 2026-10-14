"""E1–E4 real-inference runners. Gold labels enter only after states are generated."""
from __future__ import annotations
import datetime
import importlib.metadata
import shutil
import json
import os
import platform
import subprocess
from pathlib import Path
from protocol import (ROOT, CYCLES, load_official, select_sample, retrieve, assign_evidence,
                      role_order, evidence_lines, build_prompt, exact_slice, message_text,
                      delivered_facts, majority, digest, canonical, file_hash, write_json, score)

def load_config(path=None):
    cfg = json.loads(Path(path or ROOT / 'config.json').read_text())
    for key in ['n_questions', 'top_k', 'payload_budget', 'max_tokens', 'context_limit', 'stop_depth']:
        if type(cfg[key]) is not int or cfg[key] <= 0: raise ValueError('Invalid positive integer: ' + key)
    if cfg['top_k'] < 3 or cfg['top_k'] > 10: raise ValueError('top_k must be between 3 and 10')
    if not isinstance(cfg['replicate_id'], str) or not cfg['replicate_id']: raise ValueError('replicate_id required')
    return cfg

def freeze(config):
    rows = select_sample(load_official(), config['n_questions'], config['sample_seed'])
    sources = {p.name: file_hash(p) for p in ROOT.glob('*.py') if p.name != 'build_notebooks.py'}
    dataset = json.loads((ROOT / 'data/hotpot_dev_fullwiki_v1.provenance.json').read_text())
    plan = {'config': config, 'dataset_sha256': dataset['sha256'], 'sources': sources,
            'question_ids': [r['id'] for r in rows], 'primary_estimand': 'mean[(CW-NoComm)_P - (CW-NoComm)_R]',
            'secondary': ['CW-CCW_R', 'CW-SelfUpdate interaction', 'E3 prefix-crop', 'E4 stopped-final'],
            'scope': 'HotpotQA fullwiki validation candidate contexts with fixed BM25 retrieval',
            'gold_usage': 'scoring and post-hoc diagnostics only'}
    run_id = digest(plan)[:16]
    directory = ROOT / 'artifacts' / ('run_' + run_id)
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / 'frozen_manifest.json'
    if frozen.exists() and json.loads(frozen.read_text())['plan'] != plan: raise ValueError('Frozen manifest differs')
    if not frozen.exists():
        try: commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        except subprocess.CalledProcessError: commit = None
        write_json(frozen, {'run_id': run_id, 'plan': plan, 'git_commit': commit,
                            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            'python': platform.python_version(), 'status': 'FROZEN_NOT_EXECUTED'})
        snapshot = directory / 'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for filename in sources:
            shutil.copyfile(ROOT / filename, snapshot / filename)
        versions = {}
        for package in ['numpy','pandas','matplotlib','openai','transformers','pyarrow','nbformat','nbclient','ipykernel']:
            try: versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: versions[package] = None
        write_json(directory / 'environment_versions.json', versions)
        write_json(directory / 'sample_gold.json', rows)
        # Separate gold-free input; runner uses this whitelist to construct model input.
        write_json(directory / 'sample_inputs.json', [{k: r[k] for k in ['id', 'question', 'documents']} for r in rows])
    return directory, rows

def preview(config):
    directory, rows = freeze(config)
    controls = config.get('self_refinement_control', True)
    n = len(rows)
    return {'run_dir': str(directory), 'n_questions': n,
            'E1_max_new_calls': n * (24 if controls else 18),
            'E2_standalone_max_calls': n * 9, 'E2_after_E1_new_calls': 0,
            'E3_standalone_max_calls': n * 5, 'E3_after_E1_max_calls': n * 2,
            'E4_standalone_max_calls': n * (3 + 3 * config['stop_depth']),
            'note': 'Upper bounds; E1/E2 identical prompts reuse exact cache. No inference performed.'}

def visible_documents(own, peers, pool):
    allowed = {d['title']: set(range(len(d['sentences']))) for d in own}
    for text in peers:
        for f in delivered_facts(text): allowed.setdefault(f['title'], set()).add(f['sentence_id'])
    return [{**d, 'visible_indices': sorted(allowed[d['title']])} for d in pool if d['title'] in allowed]

def initialize(row, regime, config, backend):
    # Explicit whitelist: no answer or supporting labels in this data path.
    question, qid = row['question'], row['id']
    pool = retrieve(question, row['documents'], config['top_k'])
    assignment = assign_evidence(pool, regime, qid, config['partition_seed'])
    roles = role_order(qid, config['role_seed'])
    responses = [backend.complete(build_prompt(question, roles[i], assignment[i], None, []), assignment[i],
                                 {'qid': qid, 'regime': regime, 'stage': 'initial', 'node': i}) for i in range(3)]
    return {'pool': pool, 'assignment': assignment, 'roles': roles, 'responses': responses}

def payloads(state, config, tokenizer, mode='prefix'):
    return [exact_slice(message_text(r, state['pool']), config['payload_budget'], tokenizer, mode,
                        config['crop_seed']) for r in state['responses']]

def update(row, state, edges, selected, backend, tag):
    recipients = sorted({t for s, t in edges})
    revised = list(state['responses'])
    deliveries = []
    for i in recipients:
        incoming = [selected[s]['text'] for s, t in edges if t == i]
        revised[i] = backend.complete(build_prompt(row['question'], state['roles'][i], state['assignment'][i],
                                                   state['responses'][i]['answer'], incoming),
                                      visible_documents(state['assignment'][i], incoming, state['pool']),
                                      {'qid': row['id'], 'stage': tag, 'node': i})
    for s, t in edges:
        deliveries.append({'sender': s, 'recipient': t, **selected[s], 'facts': delivered_facts(selected[s]['text'])})
    return revised, deliveries

def no_peer_update(row, state, backend):
    return [backend.complete(build_prompt(row['question'], state['roles'][i], state['assignment'][i],
                                        state['responses'][i]['answer'], []), state['assignment'][i],
                             {'qid': row['id'], 'stage': 'self_update', 'node': i}) for i in range(3)]

def measured_cost(responses):
    # Logical condition cost: includes shared initial inference. Physical billing is in calls.jsonl.
    inputs, outputs = [r['input_tokens'] for r in responses], [r['output_tokens'] for r in responses]
    return {'input_tokens': sum(inputs) if all(x is not None for x in inputs) else None,
            'output_tokens': sum(outputs) if all(x is not None for x in outputs) else None,
            'usage_complete': all(x is not None for x in inputs + outputs), 'logical_calls': len(responses),
            'cache_reads_this_execution': sum(r['cache_hit'] for r in responses),
            'call_keys': [r['call_key'] for r in responses]}

def cumulative_deliveries(rounds):
    return [{**d,'round':i+1} for i, deliveries in enumerate(rounds) for d in deliveries]

def recipient_fact_sets(deliveries):
    facts = [set() for _ in range(3)]
    for d in deliveries:
        facts[d['recipient']].update((f['title'],f['sentence_id']) for f in d['facts'])
    return facts

def condition(row, state, revised, deliveries, include_initial=True):
    answer = majority(revised)
    gold = set(map(tuple, row['supporting_facts']))
    available = {(d['title'], i) for d in state['pool'] for i in range(len(d['sentences']))}
    own = [{(d['title'], i) for d in docs for i in range(len(d['sentences']))} for docs in state['assignment']]
    transferred = recipient_fact_sets(deliveries)
    support = set(tuple(c) for r in revised if normalize_equal(r['answer'], answer) for c in r['citations'])
    initial = state['responses'] if include_initial else []
    return {'answer': answer, **score(answer, row['answer']), 'responses': revised, 'deliveries': deliveries,
            'payload_tokens': sum(d['tokens'] for d in deliveries),
            'support_available_recall': len(gold & available) / max(1, len(gold)),
            'all_support_available': gold <= available,
            'split_support_across_nodes': (gold <= available and not any(gold <= o for o in own)),
            'initial_support_recall_by_node': [len(gold & o) / max(1, len(gold)) for o in own],
            'new_support_facts_delivered_by_node': [len((gold & tr) - o) for tr, o in zip(transferred, own)],
            'answer_support_recall': len(gold & support) / max(1, len(gold)),
            'answer_support_precision': len(gold & support) / max(1, len(support)),
            'invalid_citation_count': sum(len(r['invalid_citations']) for r in revised),
            'cost': measured_cost(initial + revised),
            'citation_entailment': 'not measured'}

def normalize_equal(a, b):
    from protocol import normalize
    return normalize(a) == normalize(b)

def run_question(row, experiment, config, tokenizer, backend):
    record = {'id': row['id'], 'type': row['type'], 'experiment': experiment, 'conditions': {}}
    if experiment in ['E1', 'E2']:
        for regime in (['R', 'P'] if experiment == 'E1' else ['R']):
            state = initialize(row, regime, config, backend)
            base = condition(row, state, state['responses'], [], include_initial=False)
            record['conditions'][regime + '_NoComm'] = base
            if experiment == 'E1' and config.get('self_refinement_control', True):
                refined = no_peer_update(row, state, backend)
                record['conditions'][regime + '_SelfUpdate'] = condition(row, state, refined, [])
            selected = payloads(state, config, tokenizer)
            for label, edges in CYCLES.items():
                revised, delivered = update(row, state, edges, selected, backend, label)
                record['conditions'][regime + '_' + label] = condition(row, state, revised, delivered)
            cw = record['conditions'][regime + '_CW']
            ccw = record['conditions'][regime + '_CCW']
            if cw['payload_tokens'] != ccw['payload_tokens']: raise AssertionError('Cycle budgets differ')
            if [d['text'] for d in cw['deliveries']] != [d['text'] for d in ccw['deliveries']]:
                raise AssertionError('Frozen source messages differ')
            record.setdefault('states', {})[regime] = state
    elif experiment == 'E3':
        state = initialize(row, 'R', config, backend)
        texts = message_text(state['responses'][0], state['pool'])
        for mode in ['prefix', 'crop']:
            chosen = exact_slice(texts, config['payload_budget'], tokenizer, mode, config['crop_seed'])
            revised, delivered = update(row, state, [(0, 1)], [chosen], backend, mode)
            # E3 endpoint is fixed RECEIVER answer, not ensemble that dilutes one-node changes.
            out = condition(row, state, revised, delivered)
            out['ensemble_answer'] = out['answer']
            out['answer'] = revised[1]['answer']
            out.update(score(out['answer'], row['answer']))
            gold = set(map(tuple, row['supporting_facts']))
            support = set(map(tuple, revised[1]['citations']))
            out['answer_support_recall'] = len(gold & support) / max(1,len(gold))
            out['answer_support_precision'] = len(gold & support) / max(1,len(support))
            out['invalid_citation_count'] = len(revised[1]['invalid_citations'])
            out['cost'] = measured_cost(state['responses'] + [revised[1]])
            record['conditions']['R_' + mode] = out
        if record['conditions']['R_prefix']['payload_tokens'] != record['conditions']['R_crop']['payload_tokens']:
            raise AssertionError('Content payload budgets differ')
        record['crop_equals_prefix'] = record['conditions']['R_prefix']['deliveries'][0]['text'] == record['conditions']['R_crop']['deliveries'][0]['text']
    elif experiment == 'E4':
        state = initialize(row, 'P', config, backend)
        snapshots = [condition(row, state, state['responses'], [], include_initial=False)]
        all_responses = list(state['responses'])
        total_payload = 0
        delivery_rounds = []
        for depth in range(1, config['stop_depth'] + 1):
            selected = payloads(state, config, tokenizer)
            revised, delivered = update(row, state, CYCLES['CW'], selected, backend, 'trajectory_' + str(depth))
            all_responses += revised
            total_payload += sum(d['tokens'] for d in delivered)
            delivery_rounds.append(delivered)
            snap = condition(row, state, revised, cumulative_deliveries(delivery_rounds))
            snap['round_deliveries'] = delivered
            snap['cost'] = measured_cost(all_responses)
            snap['payload_tokens'] = total_payload
            snapshots.append(snap)
            state = {**state, 'responses': revised}
        from protocol import normalize
        chosen = len(snapshots) - 1
        for depth, snap in enumerate(snapshots):
            answers = {normalize(r['answer']) for r in snap['responses']}
            if len(answers) == 1 and next(iter(answers)) not in ['', 'noanswer']:
                chosen = depth
                break
        record['conditions'] = {'P_stopped': snapshots[chosen], 'P_final': snapshots[-1]}
        record['trajectory'] = snapshots
        record['stop_depth'] = chosen
        record['hypothetical_calls_avoided'] = 3 * (config['stop_depth'] - chosen)
        record['wrong_unanimity'] = chosen < len(snapshots) - 1 and snapshots[chosen]['em'] < 100
        record['continuation_corrected_wrong_stop'] = record['wrong_unanimity'] and snapshots[-1]['em'] == 100
    else: raise ValueError(experiment)
    return record

def run(experiment, config, execute=False):
    if not execute: return preview(config)
    directory, rows = freeze(config)
    from filelock import FileLock
    # E1/E2 reuse one response cache: simultaneous notebooks must never duplicate billable requests.
    with FileLock(str(directory / 'inference.lock'), timeout=0):
        return _execute(experiment, config, directory, rows)

def _execute(experiment, config, directory, rows):
    from protocol import ExactTokenizer
    from backend import RealBackend
    tokenizer = ExactTokenizer(config['tokenizer'], config['tokenizer_revision'])
    backend = RealBackend(config, tokenizer, directory)
    output = directory / (experiment + '.jsonl')
    completed = {}
    if output.exists():
        for line in output.read_text().splitlines():
            obj = json.loads(line)
            if obj['id'] in completed: raise ValueError('Duplicate question checkpoint')
            completed[obj['id']] = obj
    if not set(completed) <= {r['id'] for r in rows}: raise ValueError('Unexpected checkpoint question')
    status = directory / (experiment + '_status.json')
    write_json(status, {'status': 'RUNNING', 'completed': len(completed), 'target': len(rows)})
    try:
        for row in rows:
            if row['id'] in completed: continue
            result = run_question(row, experiment, config, tokenizer, backend)
            with open(output, 'a') as f: f.write(canonical(result) + '\n'); f.flush(); os.fsync(f.fileno())
            completed[row['id']] = result
            print(f"{experiment}: {len(completed)}/{len(rows)}", flush=True)
        write_json(status, {'status': 'COMPLETE', 'completed': len(completed), 'target': len(rows),
                            'sha256': file_hash(output), 'tokenizer': tokenizer.identity})
    except Exception as exc:
        write_json(status, {'status': 'INCOMPLETE', 'completed': len(completed), 'target': len(rows),
                            'error_type': type(exc).__name__})
        raise
    return str(directory)
