"""Predeclared seed replication, with question-level clustering rather than pseudo-replication."""
from pathlib import Path
import json
import pandas as pd
from protocol import paired_effect, write_json
from experiments import run, preview
from reporting import load_completed

def configs(base, seeds=(11,29,47)):
    return [{**base,'partition_seed':s,'role_seed':s+100,'crop_seed':s+200,'replicate_id':f'seed_{s}'} for s in seeds]

def replicate(base, execute=False):
    runs=[]
    for cfg in configs(base):
        runs.append(run('E1',cfg,execute=execute))
    return runs

def analyze_seed_runs(directories):
    if len(directories)<3:raise ValueError('At least three prespecified complete seeds required')
    rows=[]; plans=[]
    for directory in directories:
        p=json.loads((Path(directory)/'frozen_manifest.json').read_text())['plan']
        plans.append(p)
        for row in load_completed(directory,'E1'):
            c=row['conditions']
            interaction=(c['P_CW']['f1']-c['P_NoComm']['f1'])-(c['R_CW']['f1']-c['R_NoComm']['f1'])
            rows.append({'id':row['id'],'seed':p['config']['partition_seed'],'interaction':interaction})
    if len({p['config']['partition_seed'] for p in plans})!=len(plans):raise ValueError('Duplicate partition seed')
    def comparable(p):
        c={k:v for k,v in p['config'].items() if k not in ['partition_seed','role_seed','crop_seed','replicate_id']}
        return json.dumps({'config':c,'source':p['sources'],'ids':p['question_ids'],'dataset':p['dataset_sha256']},sort_keys=True)
    if len({comparable(p) for p in plans})!=1:raise ValueError('Seed runs have incomparable protocols/models/samples')
    df=pd.DataFrame(rows)
    per_question=df.groupby('id').interaction.mean().to_dict()
    per_seed=df.groupby('seed').interaction.mean().to_dict()
    return {'primary_question_clustered':paired_effect(per_question,{q:0 for q in per_question}),
            'interaction_by_seed':per_seed,'independent_questions':len(per_question),
            'warning':'CI conditions on these seeds; three partition seeds are not independent model/backbone replications'}
