"""Independent, offline E5 matched readouts. No provider or old analysis imports."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parent
WEB = ROOT.parents[1]
RULES = ('original', 'abstention_aware', 'abstention_lexical')
METHODS = ['NoComm', 'SelfUpdate', 'Static', 'Broadcast'] + [f'{m}_seed{s}' for m in ['Learned', 'MatchedRandom'] for s in [7, 19, 31]]
ARMS = [r + '_' + m for r in ['P', 'R'] for m in METHODS]
SEED = 20261016


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))


def table(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
        for r in rows: writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def normalize(answer):
    if not isinstance(answer, str): raise ValueError('Answer must be a string')
    no_punctuation = answer.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(?:a|an|the)\b', ' ', no_punctuation).split())


def score_answer(answer, reference):
    a, b = normalize(answer), normalize(reference)
    common = sum((Counter(a.split()) & Counter(b.split())).values())
    f1 = 0.
    if common and not (a != b and ({a, b} & {'yes', 'no', 'noanswer'})):
        f1 = 200 * common / (len(a.split()) + len(b.split()))
    return {'em': 100. * (a == b), 'f1': f1}


def is_abstention(response):
    answer = response['answer']; normalize(answer)
    flag = response.get('provider_refusal', False)
    if not isinstance(flag, bool): raise ValueError('Refusal flag must be boolean')
    if not answer.strip() and not flag: raise ValueError('Unmarked empty answer')
    if answer.strip() and flag: raise ValueError('Refusal and substantive response conflict')
    return normalize(answer) == 'noanswer' or (not answer.strip() and flag)


def aggregate(responses, rule):
    if not responses or rule not in RULES: raise ValueError('Nonempty responses and known readout required')
    abstentions = [is_abstention(r) for r in responses]
    pool = list(responses) if rule == 'original' else [r for r, a in zip(responses, abstentions) if not a]
    if not pool: return 'noanswer'
    counts = Counter(normalize(r['answer']) for r in pool)
    top = max(counts.values()); ties = {answer for answer, n in counts.items() if n == top}
    winner = min(ties) if rule == 'abstention_lexical' else next(normalize(r['answer']) for r in pool if normalize(r['answer']) in ties)
    return next(r['answer'] for r in pool if normalize(r['answer']) == winner)


def bootstrap(values, draws=10000):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all(): raise ValueError('Finite paired differences required')
    rng = np.random.default_rng(SEED)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return np.quantile(samples, [.025, .975]).tolist()


def paired(a, b, binary=False, draws=10000):
    if not a or set(a) != set(b): raise ValueError('Exact complete question pairing required')
    ids = sorted(a); diffs = np.array([a[q] - b[q] for q in ids], dtype=float)
    if not np.isfinite(diffs).all(): raise ValueError('Missing/nonfinite outcome')
    out = {'n_questions': len(ids), 'delta_pp': float(diffs.mean()), 'ci95_pp': bootstrap(diffs, draws),
           'wins': int(sum(diffs > 1e-9)), 'losses': int(sum(diffs < -1e-9)), 'ties': int(sum(abs(diffs) <= 1e-9))}
    if binary:
        if any(a[q] not in (0, 100) or b[q] not in (0, 100) for q in ids): raise ValueError('Binary EM must be0or100')
        d = out['wins'] + out['losses']
        out['exact_mcnemar_p'] = min(1., 2 * float(binom.cdf(min(out['wins'], out['losses']), d, .5))) if d else 1.
    return out


def interaction(new_comm, new_base, old_comm, old_base, draws=10000):
    if not new_comm or any(set(x) != set(new_comm) for x in [new_base, old_comm, old_base]):
        raise ValueError('Four-cell question pairing required')
    values = [(new_comm[q] - new_base[q]) - (old_comm[q] - old_base[q]) for q in sorted(new_comm)]
    return {'n_questions': len(values), 'delta_pp': float(np.mean(values)), 'ci95_pp': bootstrap(values, draws),
            'interpretation': 'exploratory_readout_by_communication_contrast_not_isolated_mediation'}


def validate_rows(rows, gold, arms=ARMS):
    ids = [r['id'] for r in rows]
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(gold): raise ValueError('Question set incomplete or duplicated')
    for row in rows:
        if set(row['conditions']) != set(arms): raise ValueError('Condition set incomplete or changed')
        for c in row['conditions'].values():
            if len(c['responses']) != 3: raise ValueError('Exactly three ordered agent responses required')
            for response in c['responses']: is_abstention(response)


def load_inputs():
    provenance = json.loads((ROOT.parent / 'offline/provenance.json').read_text())
    found = {Path(k).name: Path(k) for k in provenance['source_artifacts'] if k.endswith(('E5.jsonl', 'sample_gold.json'))}
    for path in found.values():
        if sha(path) != provenance['source_artifacts'][str(path)]: raise ValueError('Historical raw hash changed')
    raw = found['E5.jsonl']; status = json.loads((raw.parent / 'E5_status.json').read_text())
    if status['status'] != 'COMPLETE' or status['sha256'] != sha(raw): raise ValueError('E5 is not frozen complete')
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    gold = {r['id']: r for r in json.loads(found['sample_gold.json'].read_text())}
    validate_rows(rows, gold)
    return rows, gold, raw, found


def official_evaluator():
    path = WEB / 'references/hotpot_evaluate_v1.py'
    # Execute only these three inspected pure definitions; omit CLI and I/O.
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'normalize_answer', 'f1_score', 'exact_match_score'}]
    if len(selected) != 3: raise ValueError('Official evaluator layout changed')
    env = {'re': re, 'string': string, 'Counter': Counter}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), env)
    return env, path


def states(responses):
    abstentions = [is_abstention(r) for r in responses]
    counts = Counter(normalize(r['answer']) for r in responses)
    substantive = Counter(normalize(r['answer']) for r, a in zip(responses, abstentions) if not a)
    top = max(counts.values())
    return {'n_abstentions': sum(abstentions), 'n_substantive_strings': len(substantive),
            'all_abstaining': all(abstentions), 'unanimous': len(counts) == 1,
            'node_order_tie': sum(n == top for n in counts.values()) > 1,
            'substantive_tie': bool(substantive) and sum(n == max(substantive.values()) for n in substantive.values()) > 1}


def holm(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]['em']['exact_mcnemar_p'])
    running = 0.
    for j, i in enumerate(order):
        running = max(running, min(1., (len(rows) - j) * rows[i]['em']['exact_mcnemar_p']))
        rows[i]['em']['holm_p_within_declared_family'] = running
        rows[i]['em']['multiplicity_family_size'] = len(rows)


def main():
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    rows, gold, raw, inputs = load_inputs(); official, evaluator_path = official_evaluator()
    records = []; by_group = defaultdict(dict); verification = Counter(); reaggregation = []
    for line, row in enumerate(rows, 1):
        qid = row['id']; reference = gold[qid]['answer']
        for arm in ARMS:
            c = row['conditions'][arm]; rs = c['responses']; regime = arm[0]
            initial = row['conditions'][regime + '_NoComm']['responses']
            if c['all_support_available'] != row['conditions'][regime + '_NoComm']['all_support_available']:
                raise ValueError('Annotated-support stratum changed after communication')
            info = states(rs); initial_info = states(initial)
            coverage = any(score_answer(r['answer'], reference)['em'] == 100 for r in rs)
            initial_coverage = any(score_answer(r['answer'], reference)['em'] == 100 for r in initial)
            original = aggregate(rs, 'original'); original_score = score_answer(original, reference)
            if original != c['answer'] or any(abs(original_score[m] - c[m]) > 1e-8 for m in ['em', 'f1']):
                raise ValueError('Independent original reconstruction differs from frozen arm')
            for response in rs:
                verification['response_references'] += 1
                own = score_answer(response['answer'], reference)
                assert own['em'] == 100 * official['exact_match_score'](response['answer'], reference)
                assert abs(own['f1'] - 100 * official['f1_score'](response['answer'], reference)[0]) < 1e-8
            for rule in RULES:
                answer = aggregate(rs, rule); score = score_answer(answer, reference)
                assert score['em'] == 100 * official['exact_match_score'](answer, reference)
                assert abs(score['f1'] - 100 * official['f1_score'](answer, reference)[0]) < 1e-8
                if score['em'] == 100 and not coverage: raise ValueError('Selection readout exceeded proposal ceiling')
                norm_counts = Counter(normalize(r['answer']) for r in rs)
                key = normalize(reference); correct_count = norm_counts[key]
                correct_tied_top = correct_count > 0 and correct_count == max(norm_counts.values())
                rec = {'question_id': qid, 'type': row['type'], 'regime': regime, 'arm': arm, 'readout': rule,
                    'answer': answer, 'normalized_answer': normalize(answer), **score, **info,
                    'final_abstention': normalize(answer) == 'noanswer' or not answer.strip(),
                    'initial_n_abstentions': initial_info['n_abstentions'],
                    'initial_n_substantive_strings': initial_info['n_substantive_strings'],
                    'initial_unanimous': initial_info['unanimous'],
                    'initial_any_correct': initial_coverage, 'final_any_correct': coverage,
                    'correct_proposal_count': correct_count,
                    'original_answer': original, 'original_em': original_score['em'],
                    'changed_from_original': normalize(answer) != normalize(original),
                    'original_abstention_override': (normalize(original) == 'noanswer' or not original.strip()) and info['n_abstentions'] < 3,
                    'correct_minority_rejected': coverage and score['em'] == 0 and correct_count < max(norm_counts.values()),
                    'correct_tied_top_rejected': coverage and score['em'] == 0 and correct_tied_top,
                    'missing_annotated_support': not c['all_support_available'],
                    'support_available_recall': c['support_available_recall'],
                    'citation_entailment': 'NOT_INDEPENDENTLY_ASSESSED',
                    'initial_answers': [r['answer'] for r in initial], 'final_answers': [r['answer'] for r in rs],
                    'initial_refusals': [r.get('provider_refusal', False) for r in initial],
                    'final_refusals': [r.get('provider_refusal', False) for r in rs],
                    'call_keys': [r['call_key'] for r in rs],
                    'raw_trace': str(raw) + ':' + str(line)}
                records.append(rec); by_group[(arm, rule)][qid] = rec
        verification['questions'] += 1
    summary = []
    for (arm, rule), group in by_group.items():
        values = list(group.values()); n = len(values)
        correct = sum(r['em'] == 100 for r in values); available = sum(r['final_any_correct'] for r in values)
        summary.append({'arm': arm, 'regime': arm[0], 'readout': rule, 'n_questions': n,
            'correct': correct, 'em': 100 * correct / n, 'f1': float(np.mean([r['f1'] for r in values])),
            'final_abstentions': sum(r['final_abstention'] for r in values),
            'final_abstention_percent': 100 * sum(r['final_abstention'] for r in values) / n,
            'agent_abstentions': sum(r['n_abstentions'] for r in values),
            'any_correct_proposals': available, 'proposal_ceiling_em': 100 * available / n,
            'gap_to_proposal_ceiling_pp': 100 * (available - correct) / n,
            'correct_proposal_selection_rate_percent': 100 * correct / available if available else None,
            'changed_answers_vs_original': sum(r['changed_from_original'] for r in values)})
    comparisons = []; interactions = []; transitions = []; joint_tables = []; subgroup = []
    for regime in ['P', 'R']:
        for rule in RULES:
            pairs = [(regime + '_' + m, regime + '_NoComm') for m in METHODS if m != 'NoComm']
            pairs += [(f'{regime}_Learned_seed{s}', regime + '_' + b) for s in [7, 19, 31] for b in ['Static', 'Broadcast', f'MatchedRandom_seed{s}']]
            for a, b in pairs:
                left, right = by_group[(a, rule)], by_group[(b, rule)]
                effect = {'arm': a, 'baseline': b, 'readout': rule, 'regime': regime}
                for metric in ['em', 'f1']:
                    effect[metric] = paired({q: r[metric] for q, r in left.items()}, {q: r[metric] for q, r in right.items()}, binary=metric == 'em')
                comparisons.append(effect)
            for method in METHODS:
                arm = regime + '_' + method; group = by_group[(arm, rule)]; base = by_group[(regime + '_NoComm', rule)]
                if method == 'NoComm': continue
                joint = defaultdict(list)
                for q, after in group.items():
                    before = base[q]
                    coverage_key = str(int(before['final_any_correct'])) + str(int(after['final_any_correct']))
                    em_key = str(int(before['em'] == 100)) + str(int(after['em'] == 100))
                    joint[coverage_key + '/' + em_key].append(q)
                    counts0, counts1 = Counter(map(normalize, before['final_answers'])), Counter(map(normalize, after['final_answers']))
                    substantive0 = {normalize(a) for a, flag in zip(before['final_answers'], before['final_refusals'])
                                    if not is_abstention({'answer': a, 'provider_refusal': flag})}
                    substantive1 = {normalize(a) for a, flag in zip(after['final_answers'], after['final_refusals'])
                                    if not is_abstention({'answer': a, 'provider_refusal': flag})}
                    correct0 = {i for i, a in enumerate(before['final_answers']) if score_answer(a, gold[q]['answer'])['em'] == 100}
                    correct1 = {i for i, a in enumerate(after['final_answers']) if score_answer(a, gold[q]['answer'])['em'] == 100}
                    transitions.append({'question_id': q, 'arm': arm, 'readout': rule, 'regime': regime,
                        'coverage_transition': coverage_key, 'correctness_transition': em_key,
                        'em_delta': after['em'] - before['em'], 'f1_delta': after['f1'] - before['f1'],
                        'new_correct_proposal': coverage_key == '01', 'correct_proposal_disappeared': coverage_key == '10',
                        'candidate_set_unchanged': set(counts0) == set(counts1),
                        'substantive_candidate_set_unchanged': substantive0 == substantive1,
                        'candidate_multiset_unchanged': counts0 == counts1,
                        'ordered_proposals_unchanged': list(map(normalize, before['final_answers'])) == list(map(normalize, after['final_answers'])),
                        'proposal_multiplicity_changed': counts0 != counts1,
                        'correct_proposal_multiplicity_change': len(correct1) - len(correct0),
                        'correct_nodes_before': sorted(correct0), 'correct_nodes_after': sorted(correct1),
                        'new_correct_nodes': sorted(correct1 - correct0), 'lost_correct_nodes': sorted(correct0 - correct1),
                        'selected_differently_with_same_set': set(counts0) == set(counts1) and before['normalized_answer'] != after['normalized_answer'],
                        'before_answer': before['answer'], 'after_answer': after['answer'],
                        'before_answers': before['final_answers'], 'after_answers': after['final_answers'],
                        'before_original_abstention_override': before['original_abstention_override'],
                        'after_original_abstention_override': after['original_abstention_override'],
                        'before_selected_abstention_override': before['final_abstention'] and before['n_abstentions'] < 3,
                        'after_selected_abstention_override': after['final_abstention'] and after['n_abstentions'] < 3,
                        'after_substantive_tie': after['substantive_tie'],
                        'after_correct_minority_rejected': after['correct_minority_rejected'],
                        'after_correct_tied_top_rejected': after['correct_tied_top_rejected'],
                        'missing_annotated_support': after['missing_annotated_support'],
                        'semantic_source_support': 'NOT_INDEPENDENTLY_ASSESSED', 'raw_trace': after['raw_trace']})
                joint_tables.append({'arm': arm, 'readout': rule, 'cells': {k: {'n': len(v), 'ids': v} for k, v in sorted(joint.items())}})
                for field in ['missing_annotated_support', 'initial_unanimous', 'initial_n_substantive_strings', 'type']:
                    for value in sorted({r[field] for r in group.values()}, key=str):
                        ids = [q for q, r in group.items() if r[field] == value]
                        subgroup.append({'arm': arm, 'readout': rule, 'field': field, 'value': value, 'n': len(ids),
                            'em': paired({q: group[q]['em'] for q in ids}, {q: base[q]['em'] for q in ids}, binary=True, draws=2000),
                            'f1': paired({q: group[q]['f1'] for q in ids}, {q: base[q]['f1'] for q in ids}, draws=2000)})
        for method in METHODS:
            arm = regime + '_' + method
            for rule in RULES[1:]:
                left, right = by_group[(arm, rule)], by_group[(arm, 'original')]
                reaggregation.append({'arm': arm, 'readout': rule,
                    **{metric: paired({q: r[metric] for q, r in left.items()}, {q: r[metric] for q, r in right.items()}, binary=metric == 'em') for metric in ['em', 'f1']}})
            if method == 'NoComm': continue
            new_a, new_b = by_group[(arm, 'abstention_aware')], by_group[(regime + '_NoComm', 'abstention_aware')]
            old_a, old_b = by_group[(arm, 'original')], by_group[(regime + '_NoComm', 'original')]
            interactions.append({'arm': arm, 'baseline': regime + '_NoComm',
                **{metric: interaction(*[{q: r[metric] for q, r in g.items()} for g in [new_a, new_b, old_a, old_b]]) for metric in ['em', 'f1']}})
    holm(comparisons); holm(reaggregation)
    jsonl(out / 'question_readouts.jsonl', records); jsonl(out / 'communication_transitions.jsonl', transitions)
    jsonl(out / 'readout_failures.jsonl', [r for r in records if r['em'] != r['original_em']])
    save(out / 'matched_summary.json', summary); table(out / 'matched_summary.csv', summary)
    save(out / 'communication_effects.json', comparisons); save(out / 'readout_effects.json', reaggregation)
    save(out / 'interaction_effects.json', interactions); save(out / 'joint_transitions.json', joint_tables)
    save(out / 'all_pre_treatment_subgroups.json', subgroup)
    ref_audit = {'selected_references': len(gold), 'normalized_noanswer_references': sum(normalize(r['answer']) == 'noanswer' for r in gold.values()),
        'raw_empty_references': sum(not r['answer'].strip() for r in gold.values()),
        'normalized_empty_but_nonempty_references': [q for q, r in gold.items() if r['answer'].strip() and not normalize(r['answer'])],
        'counts': dict(verification), 'readout_rows': len(records), 'communication_transition_rows': len(transitions),
        'official_evaluator_path': str(evaluator_path), 'official_evaluator_sha256': sha(evaluator_path),
        'all_original_answers_and_scores_match': True, 'all_readout_scores_match_official_evaluator': True,
        'aggregation_inputs': ['ordered_answer_strings', 'explicit_provider_refusal_flags'], 'gold_in_aggregation': False,
        'new_provider_requests': 0, 'new_paid_completions': 0,
        'source_hashes': {str(p): sha(p) for p in list(inputs.values()) + [Path(__file__), evaluator_path, WEB / 'protocol.py', WEB / 'experiments.py', WEB / 'controller_study.py']},
        'graph': {'project': 'projectRLMAIN', 'generation': '2026-10-13T17:28:45Z', 'coverage': 'not_tracked_direct_source_fallback'},
        'exploratory_post_hoc': True, 'bootstrap_draws_main': 10000, 'bootstrap_draws_subgroups': 2000, 'bootstrap_seed': SEED}
    save(out / 'verification.json', ref_audit)
    print(json.dumps([r for r in summary if r['arm'] in ['P_NoComm', 'P_Broadcast', 'R_NoComm', 'R_Broadcast']], indent=2))


if __name__ == '__main__': main()
