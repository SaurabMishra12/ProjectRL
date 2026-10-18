"""Preservation and deliverable verification. Never updates the historical baseline."""
from pathlib import Path
import ast, hashlib, json, re
ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify_hashes(items):
    for rel,digest in items.items():
        p=Path(rel)
        if not p.is_absolute():p=REPO/p
        assert p.exists(),p
        assert sha(p)==digest,('HISTORICAL_OR_INPUT_CHANGED',str(p))
    return len(items)
def main():
    baseline=json.loads((ROOT/'results/historical_baseline.json').read_text())
    n=verify_hashes(baseline)
    physical=json.loads((REPO/'WEBExtension/causal_study/readout_reassessment/results/physical_accounting.json').read_text())
    count=verify_hashes(physical['source_hashes'])
    inputs={}
    for f in ('analysis_input_hashes.json','history_input_hashes.json'):
        inputs.update(json.loads((ROOT/'results'/f).read_text()))
    ni=verify_hashes(inputs)
    for c in json.loads((ROOT/'results/evidence_ledger.json').read_text()):
        for p in c['artifacts']:assert Path(p).exists(),p
    historical=json.loads((REPO/'WEBExtension/causal_study/external_validation/results/historical_four_readouts.json').read_text())
    own=json.loads((ROOT/'results/offline_audit.json').read_text())['E5_readout_exact_counts']
    mapping={'original':'original','aware':'aware','lexical':'lex','strict_majority':'strict'}
    assert all(own[r['arm']][mapping[r['rule']]]==r['correct'] for r in historical['metrics'])
    assert len(historical['metrics'])==80
    candidates=json.loads((ROOT/'results/candidates.json').read_text())
    assert len(candidates)==9 and not any(c['survives'] for c in candidates)
    for p in (ROOT/'analyses').glob('*.py'):
        tree=ast.parse(p.read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        assert not any(x.startswith(('openai','httpx','requests','backend','pipeline','controller')) for x in imports),(p,imports)
    required=['01_EVIDENCE_LEDGER.md','02_LITERATURE_FRONTIER_AND_GAP_MAP.md','03_CANDIDATE_REJECTION_LOG.md',
              '04_TOP_THREE_RESEARCH_QUESTIONS.md','06_RESEARCH_LOOP_HISTORY.md','07_COMPARATIVE_RANKING.md',
              '08_FINAL_SCIENTIFIC_RECOMMENDATION.md','09_API_FEASIBILITY_AND_POWER.md','10_REVIEWER_RED_TEAM_REPORT.md','README.md']
    assert all((ROOT/p).is_file() and (ROOT/p).stat().st_size>100 for p in required)
    result=dict(status='PASS',historical_protected_files_unchanged=n,physical_source_hashes_unchanged=count,
                distinct_analysis_inputs_unchanged=ni,historical_readout_cells_independently_matched=80,
                candidates=9,survivors=0,decision='C',new_model_inference=0,new_paid_judge_calls=0,new_PPO_training=0,
                boundary='Inventory excludes binary interpretation and unreferenced bodies; full current scientific checks are documented in README.',
                e6_ledger_sha256=sha(REPO/'WEBExtension/causal_study/paid_pilot/ledger.jsonl'))
    (ROOT/'results/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json':
            files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    (ROOT/'artifact_manifest.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
