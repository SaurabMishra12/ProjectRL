"""Source/abstention strata and exhaustive changed-question accounting, offline."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from .analysis import ROOT, WEB, RULES, load_inputs, normalize, aggregate, is_abstention, score_answer, save, jsonl, sha


def main():
    rows, gold, raw, _ = load_inputs()
    keys = {}; audit = Counter(); readout_changes = []; source_rows = []
    for line, row in enumerate(rows, 1):
        qid = row['id']; reference = gold[qid]['answer']
        for regime in ['P', 'R']:
            base = row['conditions'][regime + '_NoComm']; union = set(); roles = []
            for response in base['responses']:
                p = raw.parent / 'inference/responses' / (response['call_key'] + '.json')
                cached = json.loads(p.read_text()); request = cached['request']; payload = json.loads(request['messages'][1]['content'])
                if payload['question'] != gold[qid]['question']: raise ValueError('Initial request question differs')
                union |= {(f['title'], f['sentence_id']) for f in payload['evidence']}
                roles.append(request['messages'][0]['content'].split(' specialist answering')[0])
            support = set(map(tuple, gold[qid]['supporting_facts'])); present = support & union
            expected = support <= union
            if base['all_support_available'] != expected: raise ValueError('Saved annotated-support flag differs from initial request union')
            source_rows.append({'question_id': qid, 'regime': regime,
                'annotated_support_sentences_present': len(present), 'annotated_support_sentences_total': len(support),
                'annotated_support_titles_present': len({title for title, sentence in present}),
                'all_annotated_support_present': expected,
                'semantic_source_sufficiency': 'NOT_INDEPENDENTLY_ASSESSED',
                'roles_in_node_order': roles, 'node_order': [0, 1, 2],
                'raw_trace': str(raw) + ':' + str(line)})
        for arm, c in row['conditions'].items():
            rs = c['responses']; original = aggregate(rs, 'original'); orig = score_answer(original, reference)
            counts = Counter(normalize(r['answer']) for r in rs)
            for r in rs:
                if r['call_key'] in keys and keys[r['call_key']]['answer'] != r['answer']: raise ValueError('A cache key maps to multiple answers')
                keys[r['call_key']] = r
            for rule in RULES[1:]:
                answer = aggregate(rs, rule); new = score_answer(answer, reference)
                if normalize(answer) == normalize(original): continue
                selected_abstention = normalize(original) == 'noanswer' or not original.strip()
                readout_changes.append({'question_id': qid, 'arm': arm, 'readout': rule,
                    'before_answer': original, 'after_answer': answer, 'original_em': orig['em'], 'new_em': new['em'],
                    'em_delta': new['em'] - orig['em'], 'f1_delta': new['f1'] - orig['f1'],
                    'abstention_overrode_substantive_answer': selected_abstention and any(not is_abstention(r) for r in rs),
                    'abstention_won_by_plurality': selected_abstention and counts[normalize(original)] > max([n for a, n in counts.items() if a != normalize(original)], default=0),
                    'abstention_won_by_node_tie': selected_abstention and counts[normalize(original)] == max([n for a, n in counts.items() if a != normalize(original)], default=0),
                    'candidate_multiset_unchanged_by_readout': True,
                    'gold_used_only_for_scoring': True, 'final_answers': [r['answer'] for r in rs],
                    'missing_annotated_support': not c['all_support_available'],
                    'semantic_source_support': 'NOT_INDEPENDENTLY_ASSESSED', 'raw_trace': str(raw) + ':' + str(line)})
    for response in keys.values():
        audit['unique_provider_responses'] += 1
        audit['explicit_provider_refusals'] += bool(response.get('provider_refusal'))
        audit['raw_empty_answers'] += response['answer'] == ''
        audit['whitespace_only_nonempty_answers'] += bool(response['answer']) and not response['answer'].strip()
        audit['normalized_noanswer'] += normalize(response['answer']) == 'noanswer'
        audit['normalized_empty_but_nonempty_answers'] += bool(response['answer'].strip()) and not normalize(response['answer'])
    readouts = [json.loads(line) for line in (ROOT / 'results/question_readouts.jsonl').read_text().splitlines()]
    source_map = {(r['question_id'], r['regime']): r for r in source_rows}; strata = []
    for regime in ['P', 'R']:
        for title_count in [0, 1, 2]:
            ids = {q for (q, r), v in source_map.items() if r == regime and v['annotated_support_titles_present'] == title_count}
            for arm in [regime + '_NoComm', regime + '_SelfUpdate', regime + '_Broadcast']:
                for rule in RULES:
                    group = [r for r in readouts if r['question_id'] in ids and r['arm'] == arm and r['readout'] == rule]
                    strata.append({'regime': regime, 'annotated_support_titles_present': title_count, 'arm': arm, 'readout': rule,
                        'n': len(group), 'agent_abstentions': sum(r['n_abstentions'] for r in group),
                        'all_abstaining': sum(r['all_abstaining'] for r in group),
                        'correct': sum(r['em'] == 100 for r in group), 'any_correct': sum(r['final_any_correct'] for r in group),
                        'original_abstention_overrides': sum(r['original_abstention_override'] for r in group),
                        'interpretation': 'observational_annotated_support_stratum_not_semantic_necessity'})
    path = WEB / 'data/hotpot_dev_fullwiki_v1.json'
    official_rows = json.loads(path.read_text())
    reference_audit = {'n_official_questions': len(official_rows), 'official_data_sha256': sha(path),
        'normalized_noanswer_reference_ids': [r.get('_id', r.get('id')) for r in official_rows if normalize(r['answer']) == 'noanswer'],
        'normalized_empty_nonempty_reference_ids': [r.get('_id', r.get('id')) for r in official_rows if r['answer'].strip() and not normalize(r['answer'])],
        'empty_reference_ids': [r.get('_id', r.get('id')) for r in official_rows if not r['answer'].strip()]}
    save(ROOT / 'results/response_and_reference_audit.json', {'provider': dict(audit), 'references': reference_audit})
    jsonl(ROOT / 'results/readout_answer_changes.jsonl', readout_changes)
    jsonl(ROOT / 'results/source_strata.jsonl', source_rows)
    save(ROOT / 'results/evidence_abstention_strata.json', strata)
    print(json.dumps({'provider': dict(audit), 'references': reference_audit,
        'readout_answer_changes': len(readout_changes), 'source_strata': len(source_rows)}, indent=2))


if __name__ == '__main__': main()
