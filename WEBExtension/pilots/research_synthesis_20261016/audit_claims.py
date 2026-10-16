"""Offline checks for the shared-context / distributed-evidence synthesis."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow as pa
from WEBExtension.causal_study.readout_reassessment.analysis import paired, score_answer

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
WEB=REPO/'WEBExtension'


def main():
    hashes={}
    def read(path):
        path=Path(path);data=path.read_bytes();hashes[str(path)]=hashlib.sha256(data).hexdigest();return data
    def load(path):return json.loads(read(path))
    def jl(path):return [json.loads(line) for line in read(path).splitlines()]
    p4dir=REPO/'results/p4_definitive_suite/stream'
    p5dir=REPO/'results/p5_communication_control/stream'
    p4ref=jl(p4dir/'PPO_Seed1_400ep_V50_stream.jsonl')
    p4={}
    for budget in (50,100):
        cells=[]
        for p in sorted(p4dir.glob(f'S*_V{budget}_stream.jsonl')):
            rows=jl(p);assert len(rows)==150
            assert [r['question'] for r in rows]==[r['question'] for r in p4ref]
            cells.append({'support':p.stem,'n':150,'correct':sum(r['is_correct'] for r in rows),'accuracy':100*sum(r['is_correct'] for r in rows)/150})
        assert len(cells)==15
        p4[str(budget)]={'cells':cells,'min_accuracy':min(c['accuracy'] for c in cells),'max_accuracy':max(c['accuracy'] for c in cells)}
    cells={p.stem.removesuffix('_stream'):jl(p) for p in sorted(p5dir.glob('*.jsonl'))}
    assert len(cells)==4
    p5={}
    for name,rows in cells.items():
        assert len(rows)==100 and len({r['question'] for r in rows})==100
        assert [r['question'] for r in rows]==[r['question'] for r in p4ref[:100]]
        p5[name]={'n_questions':100,'correct':sum(r['is_correct'] for r in rows),
                  'mean_proxy_tokens':sum(r['total_tokens_transmitted'] for r in rows)/100,
                  'routing_decisions':sum(len(r['active_edges_per_round']) for r in rows)}
    fixed=cells['Cell_C_Learned_Fixed'];adaptive=cells['Cell_B_Static_Adaptive'];base=cells['Cell_A_Static_Fixed']
    assert all(sorted(e)==[[0,1],[0,2]] for r in fixed for e in r['active_edges_per_round'])
    assert p5['Cell_C_Learned_Fixed']['routing_decisions']==300
    p5['halting_wins']=sum(b['is_correct'] and not a['is_correct'] for a,b in zip(base,adaptive))
    p5['halting_losses']=sum(a['is_correct'] and not b['is_correct'] for a,b in zip(base,adaptive))
    p5['communication_proxy_reduction_pct']=100*(1-14/46)
    old=load(REPO/'results/p5_communication_control/p5_master_results.json')
    p5['reported_halting_interval_and_test']=old['paired_comparisons'][1]
    for path in (REPO/'utils.py', REPO/'run_p5_master_experiment.py', REPO/'dataset_loaders.py'):read(path)
    cached=Path('/home/saurab/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866')
    pools={}
    for split,expected in [('train',7473),('test',1319)]:
        path=cached/f'gsm8k-{split}.arrow';read(path)
        with pa.memory_map(str(path),'r') as stream:table=pa.ipc.open_stream(stream).read_all()
        assert len(table)==expected
        pools[split]={' '.join(q.split()) for q in table['question'].to_pylist()}
    read(cached/'dataset_info.json')
    identity={}
    for name,rows in [('P4_reference',p4ref),('P5',fixed)]:
        identity[name]={'n':len(rows),'matches_by_whitespace_normalized_question':{split:sum(' '.join(r['question'].split()) in pool for r in rows) for split,pool in pools.items()},
                        'first_five_questions':[r['question'] for r in rows[:5]]}
    hotpot=load(WEB/'pilots/hotpot_readout_20261016/results/summary.json')
    readouts=jl(WEB/'pilots/hotpot_readout_20261016/results/question_readouts.jsonl')
    shared={a:{r['id']:r['readouts']['abstention_aware'] for r in readouts if r['experiment']=='E5' and r['arm']==a}
            for a in ('R_NoComm','R_Broadcast')}
    shared_contrast={m:paired({q:r[m] for q,r in shared['R_Broadcast'].items()},{q:r[m] for q,r in shared['R_NoComm'].items()},binary=m=='em') for m in ('em','f1')}
    gold={r['id']:r for r in load(WEB/'artifacts/run_f6aa61bac2c3dfae/sample_gold.json')}
    e2path=WEB/'artifacts/run_f6aa61bac2c3dfae/E2.jsonl';e2=jl(e2path)
    status=load(e2path.with_name('E2_status.json'));assert status['sha256']==hashlib.sha256(e2path.read_bytes()).hexdigest() and status['status']=='COMPLETE'
    assert len(e2)==300 and {r['id'] for r in e2}==set(gold)
    e2delta=sum(score_answer(r['conditions']['R_CW']['answer'],gold[r['id']]['answer'])['f1']-score_answer(r['conditions']['R_CCW']['answer'],gold[r['id']]['answer'])['f1'] for r in e2)/300
    effects=load(WEB/'analysis/final_findings_20261014/paired_effects.json')
    e2effect=next(r for r in effects if r['experiment']=='E2');assert abs(e2delta-e2effect['effect'])<1e-8
    two=load(WEB/'pilots/2wiki_20261016/results/summary.json')
    token=lambda a:two['methods'][a]['input_token_reference_sum']+two['methods'][a]['output_token_reference_sum']
    result={'scope':'OFFLINE_CLAIM_CHECK; no new model calls; original artifacts preserved',
            'p4_support_spread':p4,'p5_archival_diagnostic':p5,'p4_p5_publisher_identity_check':identity,
            'publisher_identity_interpretation':'No saved P4 reference or P5 question matches either local cached publisher GSM8K split, including whitespace normalization. Dataset lineage is unresolved; do not label these archived scores validated GSM8K results. This check does not by itself establish the historical generation route.',
            'e2_shared_cycle_f1':e2effect,'e5_shared_matched_abstention_contrast':shared_contrast,
            'e5_shared_counts':{a:hotpot['summaries']['E5'][a]['abstention_aware'] for a in shared},
            'two_wiki_full_union_vs_dense':{'assigned_questions':20,'sealed_questions':19,'full_union_correct':8,'dense_correct':6,
                 'full_union_logical_tokens':token('full_union'),'dense_logical_tokens':token('dense'),'known_logical_reference_reduction_pct':100*(1-token('full_union')/token('dense')),
                 'scope':'Operational correct counts include invalid/missing as zero over all 20; known model-token reference costs cover only 19 sealed questions.'},
            'hotpot_projection_costs':hotpot['projection_costs'],'source_hashes':hashes}
    for path,sha in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    (ROOT/'claim_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'publisher_identity':identity,'P5':p5,'e2_f1':e2delta,'e5_shared':shared_contrast,'centralization':result['two_wiki_full_union_vs_dense']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
