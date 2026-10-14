"""Question-paired, seed-aware analysis of complete frozen controller evaluations."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from protocol import file_hash, paired_effect, write_json
from reporting import physical_usage

def report_controller(directory):
    directory=Path(directory);status=json.loads((directory/'E5_status.json').read_text())
    if status['status']!='COMPLETE' or file_hash(directory/'E5.jsonl')!=status['sha256']:
        raise ValueError('Controller evaluation incomplete or changed')
    gate=json.loads((directory/'evaluation_gate.json').read_text())
    if gate!=status['gate'] or file_hash(directory/'manifest.json')!=gate['plan_sha256']:
        raise ValueError('Frozen controller evaluation gate differs')
    plan=json.loads((directory/'manifest.json').read_text());seeds=plan['settings']['training_seeds'];regimes=plan['settings']['regimes']
    for seed,checksum in gate['checkpoint_hashes'].items():
        if file_hash(directory/f'seed_{seed}'/'selected.pt')!=checksum:raise ValueError('Selected checkpoint changed after evaluation')
    rows=[json.loads(l) for l in (directory/'E5.jsonl').read_text().splitlines()]
    expected=plan['settings']['test_questions']
    if len(rows)!=expected or len({r['id'] for r in rows})!=expected:raise ValueError('Held-out sample incomplete/duplicated')
    # Gate contains exact split manifest hash; derive IDs from verified frozen held-out input.
    split_dirs=list(directory.parent.glob('controller_splits_*'))
    split=next((p for p in split_dirs if file_hash(p/'manifest.json')==gate['split_manifest_sha256']),None)
    if split is None:raise ValueError('Frozen split manifest missing')
    from controller_splits import load_partition
    ids={r['id'] for r in load_partition(split,'heldout')}
    if {r['id'] for r in rows}!=ids:raise ValueError('Wrong held-out IDs')
    expected_arms={regime+'_'+label for regime in regimes for label in ['NoComm','Static','Broadcast','SelfUpdate']}
    expected_arms|={regime+f'_{method}_seed{seed}' for regime in regimes for method in ['Learned','MatchedRandom'] for seed in seeds}
    records=[]
    for row in rows:
        if set(row['conditions'])!=expected_arms:raise ValueError('Incomplete controller comparison arms')
        for label,result in row['conditions'].items():
            records.append({'id':row['id'],'type':row['type'],'condition':label,
                            **{k:result[k] for k in ['f1','em','payload_tokens','answer_support_recall','utility_gain']},
                            **{k:result['cost'][k] for k in ['input_tokens','output_tokens','logical_calls','usage_complete']},
                            'edge_count':len(result['action']['edges']), 'quota':result['action']['quota']})
    df=pd.DataFrame(records);out=directory/'reports';out.mkdir(exist_ok=True)
    df.to_csv(out/'E5_question_level.csv',index=False)
    summary=df.groupby('condition').agg(n=('id','size'),F1=('f1','mean'),EM=('em','mean'),
                    mean_payload=('payload_tokens','mean'),calls=('logical_calls','mean'),
                    support_recall=('answer_support_recall','mean'),utility=('utility_gain','mean'),
                    input_tokens=('input_tokens',lambda x:x.mean() if x.notna().all() else float('nan')),
                    output_tokens=('output_tokens',lambda x:x.mean() if x.notna().all() else float('nan'))).reset_index()
    summary.to_csv(out/'E5_summary.csv',index=False)
    effects=[]
    for regime in regimes:
        for metric in ['f1','utility_gain']:
            learned={r['id']:float(np.mean([r['conditions'][regime+f'_Learned_seed{s}'][metric] for s in seeds])) for r in rows}
            for baseline in ['Static','NoComm','SelfUpdate','Broadcast','MatchedRandom']:
                if baseline=='MatchedRandom':
                    comparison={r['id']:float(np.mean([r['conditions'][regime+f'_MatchedRandom_seed{s}'][metric] for s in seeds])) for r in rows}
                else:
                    comparison={r['id']:r['conditions'][regime+'_'+baseline][metric] for r in rows}
                effects.append({'regime':regime,'metric':metric,'contrast':'seed_mean_learned - '+baseline,
                                'inference':'exploratory; paired by question, seeds averaged within question',**paired_effect(learned,comparison)})
    frequencies={k:dict(Counter(r['learned_action_indices'][k] for r in rows)) for k in rows[0]['learned_action_indices']}
    stats={'status':'COMPLETE','heldout_questions':expected,'optimizer_seeds':seeds,
           'effects':effects,'action_frequencies':frequencies,
           'unique_actions':{k:len(v) for k,v in frequencies.items()},
           'no_comm_rates':{k:v.get(0,0)/expected for k,v in frequencies.items()},
           'physical_usage':physical_usage(directory/'inference'),
           'limits':['This is a whole-policy comparison; call/recipient/payload differences are measured, not controlled away',
                     'Seed averaging does not create more independent questions or independent model backbones',
                     'All contrasts here are exploratory relative to the separate E1 primary interaction',
                     'A single selected support or NoComm dominance must be reported without post-hoc retraining on held-out outcomes']}
    # One exploratory family across regimes and endpoints, including all comparisons.
    ordered=sorted(effects,key=lambda x:x['randomization_p']);previous=0.
    for i,effect in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-i)*effect['randomization_p']));effect['holm_p']=previous
    write_json(out/'E5_statistics.json',stats)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for _,r in summary.iterrows():
        ax.scatter(r['mean_payload'],r['F1']);ax.annotate(r['condition'],(r['mean_payload'],r['F1']),fontsize=7)
    ax.set(xlabel='Mean actual payload tokens/question',ylabel='Answer F1 (percentage points)',title='Frozen held-out SparseComm policy comparison')
    fig.tight_layout();fig.savefig(out/'E5_payload_frontier.png',dpi=180);plt.close(fig)
    return summary,stats
