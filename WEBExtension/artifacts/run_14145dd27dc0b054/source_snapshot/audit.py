"""Recompute existing legacy observations without certifying their inference provenance."""
import json
from pathlib import Path
import pandas as pd
from protocol import ROOT, load_official, file_hash, score, paired_effect, write_json

def audit_legacy(project=None):
    project = Path(project or ROOT.parent)
    dataset = {r['id']: r for r in load_official()}
    destinations = ['W1_web_cost_accuracy', 'W2_context_rationing', 'W3_adaptive_stopping',
                    'W4_counterfactual_routing', 'W3_corrected', 'W4_corrected']
    records, checks = [], []
    output = ROOT / 'artifacts/legacy_audit'
    output.mkdir(parents=True, exist_ok=True)
    for name in destinations:
        directory = project / 'results' / name
        path = directory / 'raw_results.jsonl'
        if not path.exists():
            checks.append({'experiment': name, 'issue': 'raw_results_missing'})
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        seen = set()
        for row in rows:
            qid = row['id']
            if qid in seen: raise ValueError(f'Duplicate legacy id: {name} {qid}')
            seen.add(qid)
            if qid not in dataset: raise ValueError('Legacy id outside official dataset')
            gold = dataset[qid]['answer']
            if row.get('gold_answer', gold) != gold: raise ValueError('Legacy gold differs from publisher data')
            for arm, result in row.items():
                if not isinstance(result, dict) or 'answer' not in result: continue
                answer = result['answer']
                if not isinstance(answer, str): raise ValueError('Non-string legacy answer')
                metrics = score(answer, gold)
                records.append({'experiment': name, 'id': qid, 'arm': arm, 'answer': answer,
                                **metrics, 'empty_answer': not bool(answer.strip()),
                                'saved_f1': result.get('f1'), 'word_proxy_comm': result.get('comm_tokens'),
                                'legacy_gen_tokens_unverified': result.get('gen_tokens'),
                                'recorded_rounds': result.get('rounds'),
                                'provider_input_tokens': None, 'provider_output_tokens': None})
        manifests = {}
        for filename in ['manifest.json', 'config.json', 'raw_results.jsonl', 'predictions.jsonl', 'stream.jsonl']:
            p = directory / filename
            if p.exists(): manifests[filename] = {'path': str(p), 'sha256': file_hash(p)}
        manifest = json.loads((directory / 'manifest.json').read_text()) if (directory / 'manifest.json').exists() else {}
        checks.append({'experiment': name, 'n_questions': len(rows), 'files': manifests,
                       'claimed_manifest': manifest,
                       'limitations': ['Historical source revision not established by hashes of current files',
                                       'Raw per-call prompt/response and separate provider usage not present in raw_results',
                                       'Measured answer scores do not authenticate historical backend execution'],
                       'matching_proxy_violations': sum(len({v.get('comm_tokens') for k,v in row.items()
                              if isinstance(v,dict) and 'answer' in v and 'Full' not in k}) > 1 for row in rows)
                            if name.startswith('W4') else None})
    df = pd.DataFrame(records)
    df.to_csv(output / 'question_level_rescore.csv', index=False)
    summary = df.groupby(['experiment', 'arm']).agg(n=('id', 'size'), F1=('f1', 'mean'), EM=('em', 'mean'),
            empty_answers=('empty_answer', 'sum'), mean_word_proxy=('word_proxy_comm', 'mean')).reset_index()
    summary.to_csv(output / 'legacy_summary.csv', index=False)
    effects = []
    for name, group in df.groupby('experiment'):
        arms = list(group['arm'].unique())
        ref = 'NoComm' if 'NoComm' in arms else arms[0]
        b = group[group.arm == ref].set_index('id')['f1'].to_dict()
        for arm in arms:
            if arm == ref: continue
            a = group[group.arm == arm].set_index('id')['f1'].to_dict()
            effects.append({'experiment': name, 'contrast': arm + ' - ' + ref,
                            'inference': 'exploratory legacy rescore', **paired_effect(a, b)})
    write_json(output / 'audit.json', {'status': 'LEGACY_RESCORE_ONLY', 'experiments': checks, 'paired_effects': effects})
    source_files = ['web_suite.py', 'llm_backend.py', 'dataset_loaders.py', 'utils.py', 'config.py',
                    'run_w1_web_cost_accuracy.py', 'run_w2_context_rationing.py', 'run_w3_corrected.py', 'run_w4_corrected.py',
                    'experiment_manifest.json']
    write_json(output / 'current_source_checksums.json', {f: file_hash(project/f) for f in source_files if (project/f).exists()})
    return summary, checks

if __name__ == '__main__':
    summary, checks = audit_legacy()
    print(summary.to_string(index=False))
