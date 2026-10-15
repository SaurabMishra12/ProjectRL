"""Validate paired outputs and score frozen choices offline; never run inference.

The default CLI validates mock checkpoints but refuses to report their EM/F1.
score_records is available for separately collected, authorized real outputs.
UNIT_TEST_FIXTURE_ONLY is an explicit test mode, never an empirical result.
"""
from __future__ import annotations

import json
from pathlib import Path

from WEBExtension.causal_study.offline_e8 import answer_score
from WEBExtension.causal_study.provider import atomic_json, digest
from .binding_design import ARMS, messages, validate_manifest
from .planning import paired_analysis

ROOT = Path(__file__).resolve().parent


def validate_records(manifest, records, *, expected_mode):
    validate_manifest(manifest)
    cases = {q['question_id']: q for q in manifest['questions']}
    ids = [r['question_id'] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(cases):
        raise ValueError('Exact complete paired question set required; no silent dropping')
    for record in records:
        if record.get('content_hash') != digest({k: v for k, v in record.items() if k != 'content_hash'}):
            raise ValueError('Output checksum mismatch')
        if record.get('manifest_hash') != manifest['manifest_sha256']:
            raise ValueError('Output manifest mismatch')
        if record.get('execution_mode') != expected_mode:
            raise ValueError('Execution mode mismatch')
        if set(record.get('arms', {})) != set(ARMS):
            raise ValueError('Incomplete or extra treatment cells')
        case = cases[record['question_id']]
        for arm in ARMS:
            cell = record['arms'][arm]
            if cell.get('choice') not in ('C0', 'C1', 'C2'):
                raise ValueError('Invalid candidate choice')
            node = case['candidate_order'][int(cell['choice'][1])]
            if cell.get('selected_node') != node:
                raise ValueError('Choice/node mispairing')
            if cell.get('selected_frozen_answer') != case['candidates'][node]['answer']:
                raise ValueError('Chosen answer differs from frozen candidate')
            if cell.get('prompt_sha256') != digest(messages(case, arm)):
                raise ValueError('Treatment prompt mismatch')
    return {'paired_questions': len(cases), 'validated_cells': 2 * len(cases),
            'execution_mode': expected_mode, 'scientific_effects_computed': False}


def score_records(manifest, records, labels, *, fixture_only=False):
    if any(r.get('execution_mode') == 'MOCK_ONLY' for r in records):
        raise ValueError('Mock choices must never be scored as scientific effects')
    mode = 'UNIT_TEST_FIXTURE_ONLY' if fixture_only else 'PAID_AUTHORIZED'
    validate_records(manifest, records, expected_mode=mode)
    expected_ids = {q['question_id'] for q in manifest['questions']}
    if set(labels) != expected_ids:
        raise ValueError('Private reference labels do not exactly match question set')
    rows = []
    for record in records:
        qid = record['question_id']
        if not isinstance(labels[qid].get('answer'), str):
            raise ValueError('Reference answer must be string')
        row = {'question_id': qid}
        for arm in ARMS:
            score = answer_score(record['arms'][arm]['selected_frozen_answer'], labels[qid]['answer'])
            row.update({arm + '_' + metric: value for metric, value in score.items()})
        rows.append(row)
    return {'status': 'UNIT_TEST_ONLY_NOT_SCIENTIFIC_EVIDENCE' if fixture_only else 'OFFLINE_SCORED_REAL_OUTPUTS',
            'manifest_hash': manifest['manifest_sha256'], 'rows': rows,
            'analysis': paired_analysis(rows), 'inference_calls': 0}


def main():
    from .study import verify_prepared
    manifest = verify_prepared()
    records = [json.loads(path.read_text()) for path in sorted((ROOT / 'mock/questions').glob('*.json'))]
    validation = validate_records(manifest, records, expected_mode='MOCK_ONLY')
    result = {'status': 'NOT_RUN_REAL_INFERENCE', 'paid_authorized': False,
              'real_em_effect': None, 'real_f1_effect': None, 'real_p_value': None,
              'mock_validation': validation,
              'reason': 'All prepared outputs are MOCK_ONLY; no scientific outcome was generated.'}
    atomic_json(ROOT / 'mock/offline_validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
