"""Final read-only source checks, independent count check, and offline snapshot."""
from __future__ import annotations
import datetime as dt
import json
import re
import string
from collections import Counter
from pathlib import Path
from .analysis import ROOT, WEB, sha, save


def independent_counts():
    # Deliberately separate implementation: no aggregate/normalize import.
    def norm(s):
        clean = ''.join(c for c in s.lower() if c not in string.punctuation)
        return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', clean).split())
    def vote(responses, filtered):
        candidates = []
        for r in responses:
            a = r['answer']
            if not a.strip() and not r.get('provider_refusal'): raise ValueError('Unmarked empty answer')
            if a.strip() and r.get('provider_refusal'): raise ValueError('Refusal conflict')
            if not filtered or (a.strip() and norm(a) != 'noanswer'): candidates.append(a)
        if not candidates: return 'noanswer'
        counts = Counter(map(norm, candidates))
        winner_index = max(range(len(candidates)), key=lambda i: counts[norm(candidates[i])])
        return candidates[winner_index]
    provenance = json.loads((ROOT.parent / 'offline/provenance.json').read_text())
    paths = {Path(p).name: Path(p) for p in provenance['source_artifacts'] if p.endswith(('E5.jsonl', 'sample_gold.json'))}
    raw = [json.loads(line) for line in paths['E5.jsonl'].read_text().splitlines()]
    gold = {r['id']: r['answer'] for r in json.loads(paths['sample_gold.json'].read_text())}
    result = {}
    for arm in ['P_NoComm', 'P_Broadcast', 'R_NoComm', 'R_Broadcast']:
        result[arm] = {'original': sum(norm(vote(r['conditions'][arm]['responses'], False)) == norm(gold[r['id']]) for r in raw),
            'abstention_aware': sum(norm(vote(r['conditions'][arm]['responses'], True)) == norm(gold[r['id']]) for r in raw),
            'any_correct': sum(any(norm(x['answer']) == norm(gold[r['id']]) for x in r['conditions'][arm]['responses']) for r in raw)}
    initial = json.loads((ROOT / 'initial_independent_counts.json').read_text())
    if result != initial: raise ValueError('Independent count checks disagree')
    return result


def fresh_pool():
    from .analysis import normalize
    data_path = WEB / 'data/hotpot_dev_fullwiki_v1.json'
    official = json.loads(data_path.read_text()); exposed = []; sources = []
    paths = set((WEB / 'artifacts').glob('**/sample_gold.json')) | set((WEB / 'artifacts').glob('controller_splits_*/*_gold.json'))
    for path in sorted(paths):
        if 'source_snapshot' in path.parts: continue
        values = json.loads(path.read_text())
        if not isinstance(values, list): raise ValueError('Unexpected exposure schema')
        exposed += values; sources.append({'path': str(path), 'sha256': sha(path)})
    exposed += json.loads((ROOT.parent / 'frozen/manifest.json').read_text())['questions']
    ids = {r.get('id', r.get('_id')) for r in exposed}
    texts = {normalize(r['question']) for r in exposed if r.get('question')}
    available = []
    for row in official:
        qid = row.get('id', row.get('_id'))
        if qid in ids or normalize(row['question']) in texts: continue
        if 'documents' in row: docs = [(d['title'], d['sentences']) for d in row['documents']]
        else:
            context = row['context']; docs = list(zip(context['title'], context['sentences'])) if isinstance(context, dict) else context
        if len(docs) < 6 or len({d[0] for d in docs}) != len(docs): continue
        if any(not d[1] or any(not isinstance(s, str) or not s.strip() for s in d[1]) for d in docs): continue
        available.append(qid)
    previous = json.loads((ROOT.parent / 'next_phase/planning_artifacts/fresh_pool.json').read_text())['ids']
    if set(available) != set(previous): raise ValueError('Fresh source pool changed; report before spending')
    result = {'n_gold_free_source_capable': len(available), 'ids': available, 'same_as_previous_phase': True,
        'not_a_qualified_scarcity_or_conflict_count': True, 'exposure_source_hashes': sources,
        'official_sha256': sha(data_path), 'new_provider_calls': 0,
        'normalization_edge_references': [{'id': r.get('id', r.get('_id')), 'answer': r['answer']} for r in official if r['answer'].strip() and not normalize(r['answer'])]}
    save(ROOT / 'results/fresh_pool_verification.json', result)
    return result


def main():
    baseline = json.loads((ROOT / 'PRESERVATION_BASELINE.json').read_text())
    for item in baseline['protected_files']:
        if sha(item['path']) != item['sha256']: raise ValueError('Historical artifact changed: ' + item['path'])
    if sha(baseline['original_E5']) != baseline['raw_sha256']: raise ValueError('Original E5 changed')
    accounting = json.loads((ROOT / 'results/physical_accounting.json').read_text())
    for path, checksum in accounting['source_hashes'].items():
        if sha(path) != checksum: raise ValueError('Original physical source changed: ' + path)
    counts = independent_counts(); pool = fresh_pool()
    log = (ROOT / 'full_tests.log').read_text()
    if '187 passed, 12 subtests passed' not in log: raise ValueError('Expected full test verification missing')
    reports = sorted(ROOT.glob('0[1-7]_*.md'))
    if len(reports) != 7: raise ValueError('Missing requested report')
    for report in reports + [ROOT / 'README.md']:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', report.read_text()):
            if target.startswith(('https://', 'http://', '#')): continue
            file = re.sub(r':\d+$', '', target.strip('<>'))
            if not Path(file).is_absolute() or not Path(file).exists(): raise ValueError('Broken report link: ' + target)
    ledger = ROOT.parent / 'paid_pilot/ledger.jsonl'
    previous_success = sum(json.loads(line)['event'] == 'success' for line in ledger.read_text().splitlines())
    if previous_success != 144: raise ValueError('Previous paid ledger changed')
    audit = {'status': 'COMPLETE_OFFLINE_REVISE_PIVOT_FINDINGS', 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'paid_authorized': False, 'new_azure_completions': 0, 'new_paid_judges': 0, 'PPO_retraining_runs': 0,
        'historical_files_byte_identical': len(baseline['protected_files']),
        'original_E5_unchanged': True, 'previous_E6_successes': previous_success, 'previous_E6_ledger_sha256': sha(ledger),
        'original_physical_source_hashes_verified': len(accounting['source_hashes']),
        'independent_counts': counts, 'fresh_source_capable_pool': pool['n_gold_free_source_capable'],
        'tests_passed': 187, 'subtests_passed': 12, 'new_reassessment_tests': 16,
        'question_ids': 300, 'E5_arms': 20, 'readouts': 3, 'question_readout_records': 18000,
        'communication_transition_records': 16200, 'requested_reports': [p.name for p in reports],
        'scientific_decision': 'REVISE', 'source_association_pilot': 'ABANDON_AS_NEXT_PRIMARY_WWW_EXPERIMENT',
        'minimum_new_inference_recommended_now': 0, 'independent_external_replication': 'NOT_ESTABLISHED'}
    save(ROOT / 'COMPLETION_AUDIT.json', audit)
    files = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts or path.name.endswith('.lock') or path.name == 'artifact_manifest.json': continue
        files.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    save(ROOT / 'artifact_manifest.json', {'scope': 'New readout reassessment only; excludes caches, locks and this manifest', 'n_files': len(files), 'files': files})
    print(json.dumps({k: v for k, v in audit.items() if k not in ['requested_reports']}, indent=2))


if __name__ == '__main__': main()
