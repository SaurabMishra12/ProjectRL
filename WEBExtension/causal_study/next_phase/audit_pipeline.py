"""Read-only source/payload checks and preservation checks; writes only this phase."""
from __future__ import annotations
import json
import copy
from collections import Counter
from pathlib import Path

from WEBExtension.causal_study.provider import atomic_json, digest
from WEBExtension.causal_study.run import tokenizer
from .binding_design import ARMS, messages, validate_case
from .deep_transitions import load_raw
from .study import ROOT, preservation_check, sha, verify_prepared


def main():
    manifest = verify_prepared()
    baseline = preservation_check()
    historical = json.loads((ROOT.parent / 'artifact_manifest.json').read_text())
    for entry in historical['files']:
        if sha(ROOT.parent / entry['path']) != entry['sha256']:
            raise ValueError('Historical artifact changed: ' + entry['path'])
    data, gold, paths, _, _ = load_raw()
    cache = paths['E5'].parent / 'inference/responses'
    by_id = {r['id']: r for r in data['E5']}
    facts_checked = 0
    for row in data['E5']:
        docs = {d['title']: d['sentences'] for d in gold[row['id']]['documents']}
        for response in row['conditions']['P_NoComm']['responses']:
            saved = json.loads((cache / (response['call_key'] + '.json')).read_text())
            if saved['content_hash'] != digest({k: v for k, v in saved.items() if k != 'content_hash'}):
                raise ValueError('Historical cache checksum mismatch')
            if response['raw_text'] != saved['response']['choices'][0]['message']['content']:
                raise ValueError('Trace response differs from cached provider text')
            payload = json.loads(saved['request']['messages'][1]['content'])
            for fact in payload['evidence']:
                if (type(fact['sentence_id']) is not int or fact['title'] not in docs or
                    fact['sentence_id'] < 0 or fact['sentence_id'] >= len(docs[fact['title']]) or
                    docs[fact['title']][fact['sentence_id']] != fact['text']):
                    raise ValueError('Original request evidence differs from official source text')
                facts_checked += 1
    prepared = json.loads((ROOT / 'prepared/model_payloads.json').read_text())
    tok = tokenizer()
    checked = []
    for case in manifest['questions']:
        initial = by_id[case['question_id']]['conditions']['P_NoComm']['responses']
        if [r['answer'] for r in initial] != [c['answer'] for c in case['candidates']]:
            raise ValueError('Frozen candidate changed from original trace')
        packets = []
        for arm in ARMS:
            packet = messages(case, arm)
            stored = [p for p in prepared if p['question_id'] == case['question_id'] and p['arm_internal_only'] == arm]
            if len(stored) != 1 or stored[0]['messages'] != packet:
                raise ValueError('Prepared model-visible payload mismatch')
            if sum(tok.count(m['content']) for m in packet) != case['prompt_tokens'][arm]:
                raise ValueError('Pinned tokenizer measurement mismatch')
            obj = json.loads(packet[1]['content'])
            if set(obj) != {'question', 'candidates', 'sources'}:
                raise ValueError('Unexpected model-visible metadata')
            if any(set(c) != {'choice', 'answer', 'source_ids'} for c in obj['candidates']):
                raise ValueError('Unexpected candidate metadata')
            if any(set(s) != {'id', 'title', 'sentence_id', 'text'} for s in obj['sources']):
                raise ValueError('Unexpected source metadata')
            packets.append((packet[0], obj))
        (system_a, a), (system_b, b) = packets
        if system_a != system_b or a['question'] != b['question'] or a['sources'] != b['sources']:
            raise ValueError('Unintended treatment difference')
        changes = 0
        for left, right in zip(a['candidates'], b['candidates']):
            if left['choice'] != right['choice'] or left['answer'] != right['answer']:
                raise ValueError('Candidate set/order changed between arms')
            changes += sum(x != y for x, y in zip(left['source_ids'], right['source_ids']))
        if changes != 2:
            raise ValueError('Expected exactly two changed citation targets')
        # A bijective renaming must preserve the represented relations. This is
        # a local structural check, not a claim of model-behavior invariance.
        renamed = copy.deepcopy(case)
        names = [s['id'] for s in case['source_bank']]
        renaming = dict(zip(names, reversed(names)))
        for source in renamed['source_bank']:
            source['id'] = renaming[source['id']]
        for key in ['recorded_links', 'exchanged_links']:
            renamed[key] = [[renaming[s] for s in group] for group in case[key]]
        renamed.pop('case_sha256')
        renamed['case_sha256'] = digest(renamed)
        validate_case(renamed)
        for arm in ARMS:
            resolved = []
            for current in [case, renamed]:
                obj = json.loads(messages(current, arm)[1]['content'])
                bank = {s['id']: {k: v for k, v in s.items() if k != 'id'} for s in obj['sources']}
                resolved.append([(c['choice'], c['answer'], [bank[s] for s in c['source_ids']])
                                 for c in obj['candidates']])
            if resolved[0] != resolved[1]:
                raise ValueError('Source renaming changed represented support relations')
        checked.append(case['question_id'])
    if len(prepared) != 2 * len(checked):
        raise ValueError('Unexpected payload count')
    audit = {'status': 'PASS_OFFLINE_ONLY', 'paid_authorized': False,
        'paid_http_attempts_this_phase': 0, 'paid_completions_this_phase': 0,
        'historical_artifacts_verified_byte_identical': len(historical['files']),
        'historical_paid_ledger_sha256': baseline['paid_ledger_sha256'],
        'historical_paid_successes': baseline['paid_status']['usage']['successful_completions'],
        'cached_initial_responses_checked': 3 * len(data['E5']),
        'request_source_sentences_checked_against_official_text': facts_checked,
        'model_visible_packets_reviewed': len(prepared),
        'questions_reviewed': checked,
        'candidate_order_counts': dict(Counter(''.join(map(str, q['candidate_order'])) for q in manifest['questions'])),
        'arm_order_counts': dict(Counter('/'.join(q['arm_order']) for q in manifest['questions'])),
        'only_visible_treatment_difference': 'two citation-target values',
        'exact_equal_prompt_tokens_in_each_pair': True,
        'gold_labels_sent': False,
        'bijective_source_id_renaming_preserves_relations': True,
        'model_behavior_under_renaming_tested': False,
        'gold_string_in_ordinary_sources_is_not_reference_metadata_leakage': True,
        'semantic_necessity_or_entailment_certified': False,
        'scientific_selector_effect_observed': False}
    atomic_json(ROOT / 'PAYLOAD_AND_PRESERVATION_AUDIT.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
