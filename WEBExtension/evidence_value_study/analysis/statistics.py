"""Question-paired inference; mock outputs and missing outcomes are never results."""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import math
import re
import string
from pathlib import Path
from scipy.stats import norm
from WEBExtension.causal_study.statistics import analyze_e6
from WEBExtension.causal_study.provider import digest

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER_METRIC = ROOT.parent / 'causal_study/external_validation/sources/stonybrooknlp__musique/metrics/answer.py'
CELLS = ('complete_private', 'complete_peer', 'missing_private', 'missing_peer')


def _publisher_functions():
    # Execute only the preserved author's pure answer functions, with explicit
    # standard-library dependencies; the upstream Metric/CLI imports are not needed.
    tree = ast.parse(PUBLISHER_METRIC.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {'re': re, 'string': string, 'collections': collections}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PUBLISHER_METRIC), 'exec'), namespace)
    return namespace


def answer_score(prediction, references):
    if not isinstance(prediction, str) or not references or any(not isinstance(x, str) for x in references):
        raise ValueError('Observed prediction and nonempty publisher references required')
    official = _publisher_functions()
    return {'em': float(max(official['compute_exact'](gold, prediction) for gold in references)),
            'f1': float(max(official['compute_f1'](gold, prediction) for gold in references))}


def factorial(rows):
    mapped = [{'question_id': r['question_id'], 'colocated_nocomm_f1': r['A'],
               'colocated_comm_f1': r['B'], 'distributed_nocomm_f1': r['C'],
               'distributed_comm_f1': r['D']} for r in rows]
    result = analyze_e6(mapped, seed=20261016)
    contrasts = result['contrasts']
    contrasts['complete_peer_effect'] = contrasts.pop('colocated_comm_effect')
    contrasts['missing_peer_effect'] = contrasts.pop('distributed_comm_effect')
    result['experiment'] = 'E9'
    result['cell_means'] = {k: sum(r[k] for r in rows) / len(rows) for k in 'ABCD'} if rows else {}
    return result


def analyze_records(records, references, expected_ids, *, manifest=None, artifact_root=None):
    if any(r.get('mode') == 'MOCK' for r in records):
        raise ValueError('MOCK records cannot be analyzed as scientific outcomes')
    if any(r.get('mode') != 'REAL' for r in records):
        raise ValueError('Explicit REAL provenance required')
    if (not isinstance(manifest, dict) or manifest.get('status') != 'QUALIFIED_LOCKED_E9' or
        manifest.get('candidate_ids') != expected_ids or not manifest.get('source_qualification_sha256')):
        raise ValueError('Independent source qualification manifest required')
    if artifact_root is None:
        raise ValueError('Raw provider provenance directory required')
    root = Path(artifact_root).resolve()
    for record in records:
        if record.get('manifest_sha256') != digest(manifest):
            raise ValueError('Record/qualification manifest provenance mismatch')
        observed = list(record.get('cells', {}).items()) + [('baseline', record.get('baseline', {}))]
        for cell, item in observed:
            if item.get('status') not in ('VALID', 'UNCERTAIN', 'REFUSED', 'ABSTAINED'):
                continue  # Missing observations remain missing; no invented raw artifact.
            path = (root / item.get('provider_artifact', '')).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Observed cell lacks raw provider provenance')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != item.get('provider_artifact_sha256'):
                raise ValueError('Raw provider provenance checksum mismatch')
            saved = json.loads(data)
            if saved.get('mode') != 'REAL' or saved.get('sha256') != digest({k: v for k, v in saved.items() if k != 'sha256'}):
                raise ValueError('Raw provider provenance mode/body mismatch')
            expected_slot = f'{record["question_id"]}/baseline'
            if cell != 'baseline':
                place, update = cell.split('_', 1)
                expected_slot = f'{record["question_id"]}/update/{place}/{update}'
            if saved.get('slot') != expected_slot:
                raise ValueError('Raw provider provenance slot mismatch')
            if item['status'] in ('VALID', 'UNCERTAIN', 'ABSTAINED'):
                answer = json.loads(saved['raw']['choices'][0]['message']['content'])['answer']
                if answer != item.get('answer'):
                    raise ValueError('Scored answer differs from raw provider provenance')
    return summarize_cells(records, references, expected_ids)


def summarize_cells(records, references, expected_ids):
    """Pure paired arithmetic, also used for synthetic unit fixtures; no export."""
    if any(r.get('mode') == 'MOCK' for r in records):
        raise ValueError('MOCK records cannot be summarized as scientific outcomes')
    by_id = {r['question_id']: r for r in records}
    if len(by_id) != len(records) or set(by_id) - set(expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Duplicate or unassigned question identity')
    if set(references) != set(expected_ids):
        raise ValueError('Reference/assignment pairing differs')
    scored, scored_em, intervals, missing = [], [], [], []
    for qid in expected_ids:
        scores, ems = {}, {}
        for letter, cell in zip('ABCD', CELLS):
            item = by_id.get(qid, {}).get('cells', {}).get(cell, {})
            if item.get('status') in ('REFUSED', 'ABSTAINED'):
                scores[letter] = 0.  # Recorded explicit abstention, not a missing request.
                ems[letter] = 0.
            elif item.get('status') in ('VALID', 'UNCERTAIN') and isinstance(item.get('answer'), str):
                metrics = answer_score(item['answer'], references[qid])
                scores[letter], ems[letter] = metrics['f1'], metrics['em']
            else:
                scores[letter] = None
                ems[letter] = None
                missing.append({'question_id': qid, 'cell': cell, 'status': item.get('status', 'MISSING')})
        lower = upper = 0.
        for letter, coefficient in [('A', 1), ('B', -1), ('C', -1), ('D', 1)]:
            value = scores[letter]
            lower += coefficient * value if value is not None else min(0, coefficient)
            upper += coefficient * value if value is not None else max(0, coefficient)
        intervals.append((lower, upper))
        scored.append({'question_id': qid, **scores})
        scored_em.append({'question_id': qid, **ems})
    if missing or not expected_ids:
        n = len(expected_ids)
        return {'status': 'INSUFFICIENT_EVIDENCE', 'assigned_questions': n, 'primary_effect': None,
                'missing_cells': missing, 'question_level_outcomes': scored,
                'interaction_identification_bounds': [sum(x[i] for x in intervals) / n for i in range(2)] if n else None,
                'p_value': None, 'missing_policy': 'NO_SYNTHESIS_NO_COMPLETE_CASE_SUBSTITUTION'}
    result = factorial(scored)
    result['four_cell_f1_em'] = {letter: {'f1': sum(r[letter] for r in scored) / len(scored),
                                         'em': sum(r[letter] for r in scored_em) / len(scored_em)} for letter in 'ABCD'}
    result['exact_match_contrasts_descriptive'] = factorial(scored_em)['contrasts']
    baseline = []
    for qid in expected_ids:
        item = by_id[qid].get('baseline', {})
        if item.get('status') in ('REFUSED', 'ABSTAINED'):
            baseline.append({'em': 0., 'f1': 0.})
        elif item.get('status') in ('VALID', 'UNCERTAIN') and isinstance(item.get('answer'), str):
            baseline.append(answer_score(item['answer'], references[qid]))
    result['full_evidence_RAG'] = ({'status': 'OBSERVED', 'n': len(baseline),
                                  **{k: sum(x[k] for x in baseline) / len(baseline) for k in ('em', 'f1')}}
                                 if len(baseline) == len(expected_ids) else {'status': 'INSUFFICIENT_EVIDENCE'})
    return result


def power_grid():
    rows = []
    z = norm.ppf(.975)
    for n in (200, 300, 600):
        for difference in (.02, .05, .10):
            for sd in (.25, .5, 1., 2.):
                noncentrality = difference * math.sqrt(n) / sd
                power = norm.sf(z - noncentrality) + norm.cdf(-z - noncentrality)
                required = math.ceil(((z + norm.ppf(.8)) * sd / difference) ** 2)
                rows.append({'n_questions': n, 'difference_f1_points': int(difference * 100),
                             'contrast_sd_unit_f1': sd, 'approx_power_two_sided_alpha05': float(power),
                             'approx_n_for_80_percent': required, 'nominal_completions': 11 * n,
                             'basis': 'ASSUMED_CONTRAST_SD_NO_E9_PILOT'})
    return rows
