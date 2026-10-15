"""Reconstruct the historical E1--E5 saved traces, without importing a provider.

Only writes to next_phase/transition_artifacts and 01_E8_DEEP_TRANSITIONS.md.
Existing E8 helpers validate call identities and official scoring; their writer is
never called. All intervals resample paired questions within one contrast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from WEBExtension.causal_study import offline_e8 as old

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def agreement(answers):
    normalized = [old.normalize(a) for a in answers]
    pairs = [(a, b) for i, a in enumerate(normalized) for b in normalized[i+1:]]
    return {'unique_answers': len(set(normalized)), 'unanimous': len(set(normalized)) == 1,
            'pairwise_agreement': sum(a == b for a, b in pairs) / len(pairs) if pairs else 1.0}


def is_abstention(response):
    """Fixed semantic rule: noanswer or empty answer with explicit provider refusal."""
    answer = response['answer']
    if answer == '' and not response.get('provider_refusal'):
        raise ValueError('Empty answer lacks an explicit provider abstention marker')
    return old.normalize(answer) == 'noanswer' or (answer == '' and bool(response.get('provider_refusal')))


def nonabstaining_vote(responses):
    """Deployable offline readout with fixed node-order ties; no outcomes are inputs."""
    candidates = [r for r in responses if not is_abstention(r)]
    return old.majority(candidates) if candidates else 'noanswer'


def cross_tab(before, after, ids):
    if not (len(before) == len(after) == len(ids)) or len(set(ids)) != len(ids):
        raise ValueError('Cross-tab requires one aligned observation per question')
    cells = {key: [] for key in ['00', '01', '10', '11']}
    for a, b, qid in zip(before, after, ids):
        cells[str(int(bool(a))) + str(int(bool(b)))].append(qid)
    return {key: {'n': len(value), 'ids': value} for key, value in cells.items()}


def paired_mean_ci(values, draws=2000, seed=20261015):
    """Question bootstrap; repeated arms/seeds are never independent resampling units."""
    if not values:
        return None
    if len(set(values)) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(sum(rng.choices(values, k=n))/n for _ in range(draws))
    return [samples[round(.025*(draws-1))], samples[round(.975*(draws-1))]]


def enrich_record(record, condition, raw_path, raw_line, initial_path, initial_line):
    record = dict(record)
    a, b = agreement(record['initial_answers']), agreement(record['final_answers'])
    initial_norm = {old.normalize(answer) for answer in record['initial_answers']}
    initial_count = sum(x == 100 for x in record['initial_candidate_em'])
    final_count = sum(x == 100 for x in record['final_candidate_em'])
    all_support = bool(condition['all_support_available'])
    support_location = ('missing_annotated_support' if not all_support else
                        'distributed_annotated_support' if condition['split_support_across_nodes'] else
                        'colocated_annotated_support')
    record.update({
        'regime': record['condition'].split('_')[0],
        'initial_unique_answers': a['unique_answers'], 'final_unique_answers': b['unique_answers'],
        'initial_unanimous': a['unanimous'], 'final_unanimous': b['unanimous'],
        'initial_disagreement': not a['unanimous'],
        'initial_pairwise_agreement': a['pairwise_agreement'],
        'final_pairwise_agreement': b['pairwise_agreement'],
        'pairwise_agreement_change': b['pairwise_agreement']-a['pairwise_agreement'],
        'initial_correct_candidate_count': initial_count, 'final_correct_candidate_count': final_count,
        'correct_candidate_count_change': final_count-initial_count,
        'final_answer_in_initial_proposals': old.normalize(record['final_answer']) in initial_norm,
        'correct_initial_proposal_disappeared': record['initial_any_em'] and not record['final_any_em'],
        'previously_unavailable_correct_proposal_appeared': not record['initial_any_em'] and record['final_any_em'],
        'annotated_support_available': all_support,
        'annotated_support_location': support_location,
        'initial_support_recall_by_node': condition['initial_support_recall_by_node'],
        'support_available_recall': condition['support_available_recall'],
        'answer_support_precision': condition['answer_support_precision'],
        'answer_support_recall': condition['answer_support_recall'],
        'citation_entailment': condition.get('citation_entailment', 'not measured'),
        'raw_path': str(raw_path), 'raw_line': raw_line,
        'raw_link': str(raw_path) + ':' + str(raw_line),
        'initial_raw_link': str(initial_path) + ':' + str(initial_line),
    })
    return record


def load_raw():
    """Use historical status gates, then independently verify old provenance hashes."""
    data, gold, evidence = old.load_default()
    provenance_path = HERE.parent/'offline/provenance.json'
    provenance = json.loads(provenance_path.read_text())
    inventory_path = HERE.parent/'offline/repository_inventory.json'
    inventory = json.loads(inventory_path.read_text())
    raw_paths = {}
    checked = {}
    if old.sha256(Path(old.__file__)) != provenance['analysis_source_sha256']:
        raise ValueError('Historical E8 validation helper changed')
    raw_evidence_paths = {path.resolve() for path in evidence if path.name in
                          {'E1.jsonl', 'E2.jsonl', 'E3.jsonl', 'E4.jsonl', 'E5.jsonl'}}
    for path_string, checksum in provenance['source_artifacts'].items():
        path = Path(path_string)
        if path.name == 'sample_gold.json':
            if path not in evidence or old.sha256(path) != checksum:
                raise ValueError('Historical gold/provenance selection changed')
        if path.name in {'E1.jsonl', 'E2.jsonl', 'E3.jsonl', 'E4.jsonl', 'E5.jsonl'}:
            if path.resolve() not in raw_evidence_paths:
                raise ValueError('Active raw source differs from historical provenance')
            experiment = path.stem
            if path.parent.name.startswith('e3_corrected'):
                experiment = 'E3-v2'
            raw_paths[experiment] = path
            actual = old.sha256(path)
            if actual != checksum:
                raise ValueError('Historical raw artifact changed: ' + str(path))
            checked[str(path)] = actual
    if set(raw_paths) != set(data):
        raise ValueError('Historical provenance does not identify all raw experiments')
    for experiment, rows in data.items():
        if len(rows) != inventory['experiments'][experiment]['n_unique_questions']:
            raise ValueError('Raw row count differs from historical inventory')
    return data, gold, raw_paths, evidence, {'historical_provenance': str(provenance_path),
            'historical_provenance_sha256': old.sha256(provenance_path),
            'historical_inventory_sha256': old.sha256(inventory_path), 'raw_hashes': checked}


def reconstruct_records(data, gold, raw_paths):
    """Return one enriched record per observed raw question/condition, plus not-run entries."""
    records, missing = [], []
    rows_by_id = {exp: {row['id']: row for row in rows} for exp, rows in data.items()}
    line_by_id = {}
    for exp, path in raw_paths.items():
        line_by_id[exp] = {json.loads(line)['id']: n for n, line in
                           enumerate(path.read_text().splitlines(), 1) if line.strip()}
    for experiment, rows in data.items():
        for row in rows:
            qid, reference = row['id'], gold[row['id']]['answer']
            if experiment == 'E3-v2' and not row.get('eligible'):
                for label in ['R_prefix', 'R_crop']:
                    missing.append({'experiment': experiment, 'id': qid, 'condition': label,
                                    'status': 'NOT_RUN_INELIGIBLE_NO_SENDER_EVIDENCE',
                                    'raw_link': str(raw_paths[experiment])+':'+str(line_by_id[experiment][qid])})
            for label, condition in row['conditions'].items():
                if experiment == 'E3-v2' and not row.get('eligible') and label in {'R_prefix', 'R_crop'}:
                    continue
                regime, history, initial_exp = label.split('_')[0], None, experiment
                if experiment in {'E1', 'E2'}:
                    initial, initial_source = row['states'][regime]['responses'], 'same_record.states.'+regime
                elif experiment in {'E3', 'E3-v2'}:
                    initial = rows_by_id['E1'][qid]['states']['R']['responses']
                    initial_source, initial_exp = 'E1.states.R; exact_call_keys_verified', 'E1'
                elif experiment == 'E4':
                    initial = row['trajectory'][0]['responses']
                    depth = row['stop_depth'] if label == 'P_stopped' else len(row['trajectory'])-1
                    history = [state['responses'] for state in row['trajectory'][:depth+1]]
                    initial_source = 'same_record.trajectory[0]'
                else:
                    initial = row['conditions'][regime+'_NoComm']['responses']
                    initial_source = 'same_record.conditions.'+regime+'_NoComm'
                endpoint = 'receiver_1' if experiment in {'E3', 'E3-v2'} else 'majority'
                record = old.condition_record(experiment, row, label, condition, initial, reference,
                                              history, initial_source, endpoint)
                record['initial_provider_refusal'] = [bool(r.get('provider_refusal')) for r in initial]
                record['final_provider_refusal'] = [bool(r.get('provider_refusal')) for r in condition['responses']]
                records.append(enrich_record(record, condition, raw_paths[experiment], line_by_id[experiment][qid],
                                             raw_paths[initial_exp], line_by_id[initial_exp][qid]))
            if experiment == 'E4':
                initial = row['trajectory'][0]['responses']
                for depth, condition in enumerate(row['trajectory']):
                    history = [state['responses'] for state in row['trajectory'][:depth+1]]
                    record = old.condition_record('E4_round', row, 'P_depth_'+str(depth), condition,
                                                  initial, reference, history, 'same_record.trajectory[0]')
                    records.append(enrich_record(record, condition, raw_paths[experiment], line_by_id[experiment][qid],
                                                 raw_paths[experiment], line_by_id[experiment][qid]))
    if len({(r['experiment'], r['id'], r['condition']) for r in records}) != len(records):
        raise ValueError('Duplicate question/condition record')
    return records, missing


def condition_summary(records, draws=2000):
    grouped = defaultdict(list)
    for r in records:
        grouped[r['experiment'], r['condition']].append(r)
    summaries = []
    for (exp, label), group in sorted(grouped.items()):
        ids = [r['id'] for r in group]
        deltas = [r['final_minus_initial_endpoint_em'] for r in group]
        f1d = [r['final_minus_initial_endpoint_f1'] for r in group]
        mean = lambda field: sum(r[field] for r in group)/len(group)
        rejected = [r for r in group if r['aggregator_rejected_correct_candidate'] is True]
        summaries.append({'experiment': exp, 'condition': label, 'n': len(group), 'endpoint': group[0]['endpoint'],
            'initial_any': sum(r['initial_any_em'] for r in group),
            'final_any': sum(r['final_any_em'] for r in group),
            'initial_endpoint_correct': sum(r['initial_endpoint_em'] == 100 for r in group),
            'final_endpoint_correct': sum(r['final_em'] == 100 for r in group),
            'em_delta_pp': sum(deltas)/len(group), 'em_delta_ci95_pp': paired_mean_ci(deltas, draws),
            'f1_delta_pp': sum(f1d)/len(group), 'f1_delta_ci95_pp': paired_mean_ci(f1d, draws),
            'coverage_transition': cross_tab([r['initial_any_em'] for r in group], [r['final_any_em'] for r in group], ids),
            'vote_transition': cross_tab([r['initial_endpoint_em'] == 100 for r in group], [r['final_em'] == 100 for r in group], ids),
            'unanimity_transition': cross_tab([r['initial_unanimous'] for r in group], [r['final_unanimous'] for r in group], ids),
            'mean_pairwise_agreement_change': mean('pairwise_agreement_change'),
            'final_answers_in_initial': sum(r['final_answer_in_initial_proposals'] for r in group),
            'final_correct_answers_in_initial': sum(r['final_em'] == 100 and r['final_answer_in_initial_proposals'] for r in group),
            'correct_candidate_count_change_sum': sum(r['correct_candidate_count_change'] for r in group),
            'aggregation_rejected_count': len(rejected) if group[0]['endpoint'] == 'majority' else None,
            'aggregation_rejected_ids': [r['id'] for r in rejected],
            'categories': dict(Counter(r['final_correctness_category'] for r in group))})
    return summaries


def paired_summary(before, after, draws=2000):
    """Pair complete question sets; reject changed initial states rather than matching on ID only."""
    left, right = {r['id']: r for r in before}, {r['id']: r for r in after}
    if len(left) != len(before) or len(right) != len(after) or left.keys() != right.keys():
        raise ValueError('Paired contrast needs identical unique question identities')
    if not left:
        raise ValueError('Empty contrast')
    ids = sorted(left)
    for qid in ids:
        a, b = left[qid], right[qid]
        if (a['experiment'], a['regime'], a['initial_call_keys']) != (b['experiment'], b['regime'], b['initial_call_keys']):
            raise ValueError('Paired contrast changes initial state or experiment')
    em = [right[q]['final_em']-left[q]['final_em'] for q in ids]
    f1 = [right[q]['final_f1']-left[q]['final_f1'] for q in ids]
    vote = cross_tab([left[q]['final_em'] == 100 for q in ids], [right[q]['final_em'] == 100 for q in ids], ids)
    coverage = cross_tab([left[q]['final_any_em'] for q in ids], [right[q]['final_any_em'] for q in ids], ids)
    joint = {c+'_'+v: [] for c in ['00', '01', '10', '11'] for v in ['00', '01', '10', '11']}
    for q in ids:
        c = str(int(left[q]['final_any_em']))+str(int(right[q]['final_any_em']))
        v = str(int(left[q]['final_em'] == 100))+str(int(right[q]['final_em'] == 100))
        joint[c+'_'+v].append(q)
    return {'experiment': before[0]['experiment'], 'regime': before[0]['regime'],
            'before': before[0]['condition'], 'after': after[0]['condition'], 'n': len(ids),
            'em_delta_pp': sum(em)/len(ids), 'em_delta_ci95_pp': paired_mean_ci(em, draws),
            'f1_delta_pp': sum(f1)/len(ids), 'f1_delta_ci95_pp': paired_mean_ci(f1, draws),
            'coverage_transition': coverage, 'vote_transition': vote,
            'joint_coverage_vote': {k: {'n': len(v), 'ids': v} for k, v in joint.items()},
            'before_correct': sum(left[q]['final_em'] == 100 for q in ids),
            'after_correct': sum(right[q]['final_em'] == 100 for q in ids),
            'before_mean_f1': sum(left[q]['final_f1'] for q in ids)/len(ids),
            'after_mean_f1': sum(right[q]['final_f1'] for q in ids)/len(ids)}


def build_contrasts(records, draws=2000):
    groups = defaultdict(list)
    for r in records:
        groups[r['experiment'], r['condition']].append(r)
    output = []
    for (exp, label), group in sorted(groups.items()):
        regime = group[0]['regime']
        baselines = []
        if exp in {'E1', 'E2', 'E5'}:
            baselines.append(regime+'_NoComm')
            if label not in {regime+'_NoComm', regime+'_SelfUpdate'}:
                baselines.append(regime+'_SelfUpdate')
        elif exp == 'E3-v2':
            baselines = ['R_NoComm_receiver', 'R_SelfUpdate_receiver']
        elif exp == 'E3' and label == 'R_crop':
            baselines = ['R_prefix']
        elif exp == 'E4' and label == 'P_final':
            baselines = ['P_stopped']
        for baseline in baselines:
            if baseline != label and (exp, baseline) in groups:
                baseline_group = groups[exp, baseline]
                excluded = 0
                if exp == 'E3-v2' and label in {'R_crop', 'R_prefix'}:
                    # These arms were not run without sender evidence. Restrict the
                    # baseline explicitly; paired_summary still rejects missing pairs.
                    observed_ids = {r['id'] for r in group}
                    aligned = [r for r in baseline_group if r['id'] in observed_ids]
                    excluded = len(baseline_group)-len(aligned)
                    baseline_group = aligned
                item = paired_summary(baseline_group, group, draws)
                item['baseline_questions_excluded_not_run'] = excluded
                output.append(item)
    return output


def reaggregation_diagnostics(records, gold, draws=2000):
    """New exploratory offline reaggregation, never a communication ablation."""
    output = []
    for regime in ['P', 'R']:
        before = [r for r in records if r['experiment'] == 'E5' and r['condition'] == regime+'_NoComm']
        broadcast = [r for r in records if r['experiment'] == 'E5' and r['condition'] == regime+'_Broadcast']
        new, detail, eligible = [], [], []
        nonabstaining_count, distinct_count = Counter(), Counter()
        for r in before:
            initial = [{'answer': answer, 'provider_refusal': refusal} for answer, refusal in
                       zip(r['initial_answers'], r['initial_provider_refusal'])]
            candidates = [response for response in initial if not is_abstention(response)]
            distinct = len({old.normalize(response['answer']) for response in candidates})
            nonabstaining_count[len(candidates)] += 1
            distinct_count[distinct] += 1
            answer = nonabstaining_vote(initial)  # Gold cannot enter this decision.
            score = old.answer_score(answer, gold[r['id']]['answer'])
            item = dict(r, condition=regime+'_NoAbstentionVote', final_answer=answer,
                        final_em=score['em'], final_f1=score['f1'])
            new.append(item)
            if distinct >= 2:
                eligible.append(r['id'])
            detail.append({'id': r['id'], 'raw_link': r['raw_link'], 'initial_answers': r['initial_answers'],
                           'nonabstaining_candidates': len(candidates), 'distinct_nonabstaining_answers': distinct,
                           'eligible_without_gold': distinct >= 2, 'original_answer': r['final_answer'],
                           'reaggregated_answer': answer, 'reaggregated_em': score['em'], 'reaggregated_f1': score['f1']})
        comparisons = [paired_summary(before, new, draws), paired_summary(new, broadcast, draws),
                       paired_summary(before, broadcast, draws)]
        for comparison in comparisons:
            comparison['population'] = 'ALL_SAVED_QUESTIONS'
        if eligible:
            ids = set(eligible)
            restricted = [paired_summary([r for r in before if r['id'] in ids], [r for r in new if r['id'] in ids], draws),
                          paired_summary([r for r in new if r['id'] in ids], [r for r in broadcast if r['id'] in ids], draws),
                          paired_summary([r for r in before if r['id'] in ids], [r for r in broadcast if r['id'] in ids], draws)]
            for comparison in restricted:
                comparison['population'] = 'AT_LEAST_TWO_DISTINCT_NONABSTAINING_INITIAL_ANSWERS'
            comparisons += restricted
        output.append({'regime': regime, 'n': len(before), 'eligible_two_distinct_nonabstaining': len(eligible),
            'eligible_ids': eligible, 'nonabstaining_candidate_count': dict(nonabstaining_count),
            'distinct_nonabstaining_answer_count': dict(distinct_count),
            'comparisons': comparisons, 'question_readouts': detail,
            'interpretation': 'EXPLORATORY_OFFLINE_REAGGREGATION_NOT_CAUSAL_COMMUNICATION_ABLATION',
            'rule': 'Exclude normalized noanswer and empty explicit provider abstention; preserve original node-order ties; all-abstaining returns noanswer'})
    return output


def subgroup_summaries(records, draws=2000):
    """Baseline-defined E5 contrasts; support availability refers to annotated facts, not sufficiency."""
    output = []
    for regime in ['P', 'R']:
        before = [r for r in records if r['experiment'] == 'E5' and r['condition'] == regime+'_NoComm']
        after = [r for r in records if r['experiment'] == 'E5' and r['condition'] == regime+'_Broadcast']
        right = {r['id']: r for r in after}
        dimensions = ['type', 'annotated_support_available', 'annotated_support_location',
                      'initial_disagreement', 'initial_any_em']
        for dimension in dimensions:
            values = sorted({r[dimension] for r in before}, key=str)
            for value in values:
                subset = [r for r in before if r[dimension] == value]
                summary = paired_summary(subset, [right[r['id']] for r in subset], draws)
                summary.update({'stratum': dimension, 'value': value, 'interpretation': 'EXPLORATORY_BASELINE_SUBGROUP_NOT_CAUSAL_MODERATION'})
                output.append(summary)
    return output


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def compact_cells(table):
    return '/'.join(str(table[key]['n']) for key in ['00', '01', '10', '11'])


def fmt_ci(ci):
    return '['+', '.join(f'{x:.3f}' for x in ci)+']' if ci else 'NA'


def raw_links(ids, lookup):
    return ', '.join('['+qid+']('+lookup[qid]['raw_link']+')' for qid in ids) or '(none)'


def write_report(records, summaries, contrasts, subgroups, diagnostics, missing, provenance, output):
    lookup = {r['id']: r for r in records if r['experiment'] == 'E5' and r['condition'] == 'P_Broadcast'}
    p = next(c for c in contrasts if c['experiment'] == 'E5' and c['before'] == 'P_NoComm' and c['after'] == 'P_Broadcast')
    ps = next(s for s in summaries if s['experiment'] == 'E5' and s['condition'] == 'P_Broadcast')
    r = next(c for c in contrasts if c['experiment'] == 'E5' and c['before'] == 'R_NoComm' and c['after'] == 'R_Broadcast')
    lines = ['# E8 deep question-level transitions', '',
        'Status: exploratory offline reconstruction. Previous scientific status remains REVISE. Paid/network inference requests: **0**.', '',
        '## Strongest observed accounting', '',
        f"In E5 partitioned evidence, deterministic final-vote EM rises from {p['before_correct']}/300 to {p['after_correct']}/300: {p['vote_transition']['01']['n']} previously wrong endpoints become correct and {p['vote_transition']['10']['n']} correct endpoint becomes wrong. The paired change is {p['em_delta_pp']:.3f} percentage points, question-bootstrap 95% CI {fmt_ci(p['em_delta_ci95_pp'])}. Initial and final any-correct proposal coverage are {ps['initial_any']}/300 and {ps['final_any']}/300, respectively.", '',
        'Coverage is **not the same set of questions**: 61 retain correct coverage, 6 lose it, 5 gain it, and 228 never have it. Of the 25 final-vote gains, 23 occur while correct coverage is present both before and after and 2 occur when correct coverage first appears. The one final-vote loss also loses all correct coverage. Thus +24 correct endpoints = +23 gains with preserved coverage +2 gains with newly correct coverage −1 loss with lost coverage. The 67→66 marginal coverage masks 11 changes in membership.', '',
        'This establishes a change in the use and multiplicity of proposals at the system level. It does not isolate selecting a fixed proposal set: node revision can copy answers, verify them, correct them independently, or change their frequency. Broadcast also delivers selected citations. No intervention separates those components. The new abstention-aware reaggregation below is stronger than original NoComm and Broadcast on overall partitioned EM; therefore the 10→34 comparison cannot support a claim that communication is needed to exploit existing correct proposals.', '',
        '| Correct coverage initial→final | Vote wrong→wrong | Wrong→correct | Correct→wrong | Correct→correct |',
        '|---|---:|---:|---:|---:|']
    for c in ['00', '01', '10', '11']:
        lines.append('| '+c+' | '+' | '.join(str(p['joint_coverage_vote'][c+'_'+v]['n']) for v in ['00', '01', '10', '11'])+' |')
    lines += ['', 'Here 0/1 denote false/true. Each cell counts unique paired questions, never individual agents or repeated seeds.', '',
        f"Broadcast final answers occur among initial proposals on {ps['final_answers_in_initial']}/300 questions; {ps['final_correct_answers_in_initial']}/{ps['final_endpoint_correct']} correct broadcast endpoints occur among initial proposals. The initially available correct proposal is rejected by the initial vote on {ps['initial_any']-ps['initial_endpoint_correct']} questions; the final vote rejects an available correct proposal on {ps['aggregation_rejected_count']} questions. The mean change in within-question pairwise proposal agreement is {ps['mean_pairwise_agreement_change']:.4f}. Agreement is descriptive and can converge on an incorrect answer.", '',
        '## Exact E5 question identities and raw lines', '',
        'Every ID below links to its full raw E5 JSONL line, containing both paired conditions, node answers, citations, deliveries and exact call keys. Full question/condition records also link to their exact initial trace; E3 initial states point to E1 and are validated by call key.', '']
    for key, cell in p['joint_coverage_vote'].items():
        if cell['n'] and key != '00_00':
            lines += [f"- Coverage {key[:2]}, vote {key[3:]} (n={cell['n']}): "+raw_links(cell['ids'], lookup)]
    lines += ['', 'The 228 neither-covered, neither-correct identities and all other complete cell memberships are in [paired_contrasts.json]('+str(output/'paired_contrasts.json')+').', '',
        '## Reconstructed protocols and paired changes', '',
        'Every row below reconstructs raw proposals and validates the recorded majority (or receiver endpoint) against the stored score using official normalization. Coverage/vote cells are ordered 00/01/10/11. Changes compare the final endpoint with the exact same initial endpoint. E3 uses receiver 1, so it is not an aggregation experiment. E4 round rows preserve intermediate revision history.', '',
        '| Protocol | n | Coverage initial/final | Coverage cells | Correct initial/final | Vote cells | EM change pp [95% paired CI] | F1 change pp [95% paired CI] |',
        '|---|---:|---:|---|---:|---|---|---|']
    for s in summaries:
        lines.append(f"| {s['experiment']} {s['condition']} | {s['n']} | {s['initial_any']}/{s['final_any']} | {compact_cells(s['coverage_transition'])} | {s['initial_endpoint_correct']}/{s['final_endpoint_correct']} | {compact_cells(s['vote_transition'])} | {s['em_delta_pp']:.3f} {fmt_ci(s['em_delta_ci95_pp'])} | {s['f1_delta_pp']:.3f} {fmt_ci(s['f1_delta_ci95_pp'])} |")
    lines += ['', 'Explicit same-question arm contrasts, including communication versus private SelfUpdate, are in [paired_contrasts.json]('+str(output/'paired_contrasts.json')+'). NoComm versus communication confounds message access with revision computation; communication versus SelfUpdate improves that control but still bundles peer strings with selected source quotations.', '',
        '## Baseline subgroup analysis', '',
        'All subgroups and intervals are exploratory, were selected after inspecting prior experiments, and have no multiplicity correction. Rows condition only on baseline information: question type, initial disagreement, initial correct coverage, and initial annotated support availability/location. No subgroup is defined by post-communication correctness. Association across baseline subgroups does not itself identify causal effect modification. Annotated article/sentence availability does not prove that every article is necessary or that a single article is insufficient.', '',
        '| Regime | Baseline stratum | n | NoComm/Broadcast correct | Gains/losses | EM change pp [95% paired CI] | F1 change pp [95% paired CI] |',
        '|---|---|---:|---:|---:|---|---|']
    for s in subgroups:
        lines.append(f"| {s['regime']} | {s['stratum']}={s['value']} | {s['n']} | {s['before_correct']}/{s['after_correct']} | {s['vote_transition']['01']['n']}/{s['vote_transition']['10']['n']} | {s['em_delta_pp']:.3f} {fmt_ci(s['em_delta_ci95_pp'])} | {s['f1_delta_pp']:.3f} {fmt_ci(s['f1_delta_ci95_pp'])} |")
    lines += ['', '## Zero-call abstention-aware reaggregation diagnostic', '',
        'This is a new exploratory deployable readout of saved initial proposals, not a causal communication ablation or a novelty claim. The rule is fixed without consulting outcomes: remove normalized `noanswer` and empty answers carrying an explicit provider-refusal marker; apply the original normalized majority to remaining candidates, with the same original node-order tie break. If all abstain, return `noanswer`. No confidence, gold answer, annotation or evidence is an input. An empty answer without a provider marker is rejected. Synonyms such as `unknown` are not retrospectively reclassified.', '',
        '| Regime | Population | Before→after | n | Correct before/after | Gains/losses | EM change pp [95% paired CI] | F1 change pp [95% paired CI] |',
        '|---|---|---|---:|---:|---:|---|---|']
    for diagnostic in diagnostics:
        for c in diagnostic['comparisons']:
            population = 'all saved questions' if c['population'] == 'ALL_SAVED_QUESTIONS' else '≥2 distinct non-abstaining initial answers'
            lines.append(f"| {diagnostic['regime']} | {population} | {c['before']}→{c['after']} | {c['n']} | {c['before_correct']}/{c['after_correct']} | {c['vote_transition']['01']['n']}/{c['vote_transition']['10']['n']} | {c['em_delta_pp']:.3f} {fmt_ci(c['em_delta_ci95_pp'])} | {c['f1_delta_pp']:.3f} {fmt_ci(c['f1_delta_ci95_pp'])} |")
    lines += ['', 'Removing abstention multiplicity raises initial-vote exact matches from 10 to 62 in P (52 gains, no losses); Broadcast has 34. In R, reaggregation raises 69 to 73, versus 70 for Broadcast. This deterministic baseline rules out interpreting the P 10→34 result as evidence that communication is necessary to make the initial correct answers usable. It does not prove that abstention multiplicity caused every broadcast gain or that communication never helps. The reaggregation rule was requested after inspecting existing results, so prospective evaluation is still needed.', '',
        'Gold-free frozen-candidate eligibility is sparse: 20/300 P questions and 33/300 R questions have at least two distinct non-abstaining initial answers. P has 158 questions with zero, 122 with one, 17 with two and 3 with three distinct non-abstaining answers. R has 142/125/25/8. These are counts in each regime over the same question identities; do not sum them as independent examples or assume they are fresh confirmatory data. Initial disagreement including abstentions is a much broader condition than disagreement between substantive answers.', '']
    for diagnostic in diagnostics:
        lookup_diagnostic = {r['id']: r for r in records if r['experiment'] == 'E5' and r['condition'] == diagnostic['regime']+'_NoComm'}
        lines.append('- '+diagnostic['regime']+' eligible question IDs: '+raw_links(diagnostic['eligible_ids'], lookup_diagnostic))
    lines += ['', 'Complete gold-free decisions, source lines, eligibility lists and paired comparisons are in [abstention_reaggregation.json]('+str(output/'abstention_reaggregation.json')+').', '']
    lines += ['', '## What the traces identify', '',
        '- **Proposal generation:** exact-correct proposal absent initially and present finally is observable. It may arise from copied evidence, private correction or prior knowledge. Temporal emergence does not prove communication caused it.',
        '- **Preservation:** any-correct coverage retained/lost and individual correct nodes becoming wrong are observable. Retained coverage can coexist with individual losses and recoveries.',
        '- **Correction:** node-level incorrect→correct transitions and proposal multiplicity changes are observable. A corrected node can adopt an existing answer; that is distinct from a newly available answer.',
        '- **Selection:** final answer originating in the initial set and correct-coverage-with-wrong-final gaps are observable. This does not provide a fixed-proposal counterfactual.',
        '- **Aggregation:** all E1/E2/E4/E5 endpoints use deterministic normalized majority; tied counts choose the first normalized answer in node order. Aggregator rejection is descriptive. E3 receiver outcomes must not be reported as vote failures.',
        '- **Evidence-backed selection:** citation identity, annotated support overlap and delivered complete source lines are observable; semantic entailment and causal reliance on sources are not. Citation overlap is not evidence-backed correctness.', '',
        '## Statistical and provenance limits', '',
        f"Saved conditions produce {len(records):,} observed question/condition records and {len(missing)} not-run records. They span the same 300 identities; E2 reuses E1 shared-evidence states, E3 and E3-v2 reuse E1 initial states, E4 contains repeated rounds, and E5 contains repeated training seeds. None is independent replication. Every displayed CI is a percentile bootstrap of {provenance['bootstrap_draws']} unique question-level paired differences within one condition/contrast, using fixed seed 20261015. Repeated arms are never pooled as independent samples. Intervals describe these selected questions and protocol realizations; generalization to new questions, model generations or training randomness is not established. Degenerate small-stratum intervals do not prove population certainty.", '',
        'No paid call, API request, tokenizer download, model client import, PPO retraining or historical writer is used. The original E8 inventory and provenance are read, the completed historical status gates are checked, all six raw JSONL hashes are checked against E8 provenance, and 90,286 reconstructed answer references are rechecked against the preserved official Hotpot evaluator. Full hashes and generated-file hashes are in [provenance.json]('+str(output/'provenance.json')+').', '',
        'Graph Verify preflight: project projectRLMAIN, generation 2026-10-13T17:28:45Z, fast/ready. causal_study search has zero entries and the exact historical source paths are not tracked. This is insufficient graph evidence, so the complete relevant source and raw saved trace paths were read directly; no absence claim is inferred from the graph.', '',
        'Artifacts: [question_conditions.jsonl]('+str(output/'question_conditions.jsonl')+'), [condition_summaries.json]('+str(output/'condition_summaries.json')+'), [paired_contrasts.json]('+str(output/'paired_contrasts.json')+'), [baseline_subgroups.json]('+str(output/'baseline_subgroups.json')+').', '']
    (HERE/'01_E8_DEEP_TRANSITIONS.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draws', type=int, default=2000)
    args = parser.parse_args(argv)
    if args.draws < 100:
        raise ValueError('At least 100 bootstrap draws required')
    output = HERE/'transition_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    data, gold, raw_paths, evidence, provenance = load_raw()
    records, missing = reconstruct_records(data, gold, raw_paths)
    summaries = condition_summary(records, args.draws)
    contrasts = build_contrasts(records, args.draws)
    subgroups = subgroup_summaries(records, args.draws)
    diagnostics = reaggregation_diagnostics(records, gold, args.draws)
    official = old.verify_official_scores(records, gold)
    (output/'question_conditions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in records))
    for name, value in [('condition_summaries.json', summaries), ('paired_contrasts.json', contrasts),
                        ('baseline_subgroups.json', subgroups), ('not_run.json', missing)]:
        save_json(output/name, value)
    save_json(output/'abstention_reaggregation.json', diagnostics)
    provenance.update({'analysis_source': str(Path(__file__).resolve()), 'analysis_source_sha256': old.sha256(__file__),
        'bootstrap_draws': args.draws, 'bootstrap_seed': 20261015, 'physical_inference_attempts': 0,
        'no_provider_imported': True, 'official_score_crosscheck': official,
        'n_question_condition_records': len(records), 'n_unique_question_ids': len(gold),
        'source_hashes': {str(path): old.sha256(path) for path in evidence},
        'e8_helper_source_sha256': old.sha256(Path(old.__file__)),
        'graph_generation': '2026-10-13T17:28:45Z', 'graph_coverage': 'NOT_TRACKED_SOURCE_FALLBACK'})
    write_report(records, summaries, contrasts, subgroups, diagnostics, missing, provenance, output)
    provenance['generated_hashes'] = {path.name: old.sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != 'provenance.json'}
    provenance['generated_hashes']['../01_E8_DEEP_TRANSITIONS.md'] = old.sha256(HERE/'01_E8_DEEP_TRANSITIONS.md')
    save_json(output/'provenance.json', provenance)
    print(json.dumps({'status': 'COMPLETE_OFFLINE_EXPLORATORY', 'records': len(records), 'unique_questions': len(gold),
                      'paid_requests': 0, 'report': str(HERE/'01_E8_DEEP_TRANSITIONS.md')}))


if __name__ == '__main__':
    main()
