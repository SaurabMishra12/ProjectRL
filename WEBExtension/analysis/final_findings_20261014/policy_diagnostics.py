"""Evaluate frozen actor distributions on saved observations only; no provider."""
import os,sys,json
from pathlib import Path
import numpy as np,pandas as pd,torch
O=Path(__file__).resolve().parent;R=O.parents[1];sys.path.insert(0,str(R));os.environ['HF_HUB_OFFLINE']='1'
from learned_controller import CategoricalController,observable_features
from protocol import ExactTokenizer
A=json.loads((R/'artifacts/active_machine_runs.json').read_text());C=Path(A['controller_study']);B=Path(A['controlled_run']);p=json.loads((C/'manifest.json').read_text());cfg=p['base'];tok=ExactTokenizer(cfg['tokenizer'],cfg['tokenizer_revision'])
gold={r['id']:r for r in json.loads((Path(A['controller_split_dir'])/'heldout_inputs.json').read_text())};e1={r['id']:r for r in map(json.loads,(B/'E1.jsonl').read_text().splitlines())};e5=list(map(json.loads,(C/'E5.jsonl').read_text().splitlines()));models={}
for seed in [7,19,31]:
 d=torch.load(C/f'seed_{seed}/selected.pt',map_location='cpu',weights_only=True);m=CategoricalController(d['feature_width'],d['n_actions'],64);m.load_state_dict(d['model']);m.eval();models[seed]=m
out=[]
with torch.no_grad():
 for row in e5:
  for regime in ['R','P']:
   state=dict(e1[row['id']]['states'][regime]);state['responses']=row['conditions'][regime+'_NoComm']['responses'];f=observable_features(gold[row['id']]['question'],state,tok)
   for seed,m in models.items():
    dist,value=m(torch.as_tensor(f));probs=dist.probs.numpy();order=np.sort(probs);action=int(probs.argmax());expected=row['learned_action_indices'][regime+f'_seed{seed}'];assert action==expected,(row['id'],seed,action,expected)
    out.append({'id':row['id'],'regime':regime,'seed':seed,'action':action,'feature_count':len(f),'max_probability':float(order[-1]),'top2_probability_margin':float(order[-1]-order[-2]),'categorical_entropy_bits':float(dist.entropy()/np.log(2)),'uniform_entropy_bits':float(np.log2(len(probs))),'uniform_max_probability':1/len(probs),'value_prediction':float(value)})
pd.DataFrame(out).to_csv(O/'tables/frozen_actor_probability_diagnostics.csv',index=False)
pd.DataFrame(out).groupby(['regime','seed']).agg(n=('id','size'),mean_max_probability=('max_probability','mean'),mean_top2_margin=('top2_probability_margin','mean'),mean_entropy_bits=('categorical_entropy_bits','mean'),min_entropy_bits=('categorical_entropy_bits','min'),max_entropy_bits=('categorical_entropy_bits','max')).to_csv(O/'tables/frozen_actor_probability_summary.csv')
print('All 1800 selected actions reproduced exactly from frozen weights and saved observations')
