"""Saved pilot, static-validation and numerical provenance catalog; no inference."""
import json,hashlib
from pathlib import Path
import pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];A=json.loads((R/'artifacts/active_machine_runs.json').read_text());C=Path(A['controller_study']);B=Path(A['controlled_run'])
def scalars(v,p=''):
 if isinstance(v,dict):
  for k,x in v.items():yield from scalars(x,(p+'.'+k).lstrip('.'))
 elif isinstance(v,list):
  for i,x in enumerate(v):yield from scalars(x,f'{p}[{i}]')
 elif isinstance(v,(int,float,bool)) or v is None:yield p,v
flat=[];summ=[]
for name in ['pilot','static_validation']:
 p=C/(name+'.jsonl');rows=list(map(json.loads,p.read_text().splitlines()))
 for row in rows:
  for field,value in scalars(row):flat.append({'source':str(p),'record_key':row['key'],'field':field,'value':value})
  result=[(reg,v) for reg,vs in row['results'].items() for v in vs] if name=='pilot' else [(row['key'].split('_')[1],row['result'])]
  for regime,v in result:summ.append({'phase':name,'regime':regime,'key':row['key'],'action':v['action']['name'],**{k:x for k,x in v.items() if isinstance(x,(int,float,bool))},**{k:x for k,x in v['cost'].items() if isinstance(x,(int,float,bool)) or x is None}})
pd.DataFrame(flat).to_csv(O/'tables/pilot_static_all_numeric_fields.csv',index=False);d=pd.DataFrame(summ);d.to_csv(O/'tables/pilot_static_question_metrics.csv',index=False)
d.groupby(['phase','regime','action']).agg(n=('key','size'),F1=('f1','mean'),EM=('em','mean'),payload=('payload_tokens','mean'),calls=('logical_calls','mean'),utility=('utility_gain','mean')).reset_index().to_csv(O/'tables/pilot_static_summary.csv',index=False)
# Every numeric field in all main raw outputs, including nested node costs, responses and delivered facts.
# Use chunked CSV writing: original response text is preserved in raw input, not duplicated here.
p=O/'tables/all_raw_numeric_fields.csv';import csv
with p.open('w') as f:
 w=csv.writer(f);w.writerow(['experiment','id','field','value'])
 for e in ['E1','E2','E3','E4','E5']:
  for line in ((C if e=='E5' else B)/(e+'.jsonl')).open():
   row=json.loads(line)
   for field,value in scalars(row):w.writerow([e,row['id'],field,'UNKNOWN_OR_NOT_APPLICABLE' if value is None else value])
# Resource checkpoint inventory, file sizes/hashes; contents remain original and local.
files=[]
for seed in [7,19,31]:
 for p in (C/f'seed_{seed}').rglob('*'):
  if p.is_file() and (p.suffix in ['.pt','.json','.jsonl','.npz']):files.append({'seed':seed,'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
pd.DataFrame(files).to_csv(O/'tables/checkpoint_training_artifact_inventory.csv',index=False)
print('Supplemental numeric register complete')
