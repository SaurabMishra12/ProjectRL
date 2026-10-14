"""Read-only validation of the final analysis bundle; no API calls."""
import json,hashlib,sys,importlib.util,re
from pathlib import Path
import pandas as pd,numpy as np
from html.parser import HTMLParser
O=Path(__file__).resolve().parent;R=O.parents[1];A=json.loads((R/'artifacts/active_machine_runs.json').read_text());C=Path(A['controller_study']);B=Path(A['controlled_run']);S=Path(A['controller_split_dir'])
sys.modules.setdefault('ujson',json);spec=importlib.util.spec_from_file_location('official',R/'references/hotpot_evaluate_v1.py');off=importlib.util.module_from_spec(spec);spec.loader.exec_module(off)
F=pd.read_csv(O/'tables/all_condition_question_metrics.csv');assert len(F)==11696
legacy=F[F.experiment!='E3_v2'];new=F[F.experiment=='E3_v2']
assert len(legacy)==10500 and legacy.groupby(['experiment','condition']).size().eq(300).all()
assert legacy.groupby(['experiment','condition']).id.nunique().eq(300).all()
assert len(new)==1196 and new.groupby('condition').size().eq(299).all()
assert new.groupby('condition').id.nunique().eq(299).all()
V=Path(A['e3_corrected']['run_dir']);gold={r['id']:r for r in json.loads((B/'sample_gold.json').read_text())}
for r in map(json.loads,(V/'E3.jsonl').read_text().splitlines()):
 for v in r['conditions'].values():
  assert abs(100*off.f1_score(v['answer'],gold[r['id']]['answer'])[0]-v['f1'])<1e-8
  assert 100*off.exact_match_score(v['answer'],gold[r['id']]['answer'])==v['em']
vr=json.loads((V/'report.json').read_text());vv=list(map(json.loads,(V/'E3.jsonl').read_text().splitlines()));vv=[r for r in vv if r['eligible']]
for metric,e in vr['paired_prefix_minus_crop'].items():
 assert abs(np.mean([r['conditions']['R_prefix'][metric]-r['conditions']['R_crop'][metric] for r in vv])-e['effect'])<1e-10
training_scores=0
for phase,split in [('rollout','train'),('validation','validation')]:
 gold={r['id']:r for r in json.loads((S/(split+'_gold.json')).read_text())}
 for seed in [7,19,31]:
  for ep in range(3):
   for line in (C/f'seed_{seed}'/f'{phase}_epoch_{ep}.jsonl').read_text().splitlines():
    r=json.loads(line);v=r['result'];f1=100*off.f1_score(v['answer'],gold[r['id']]['answer'])[0];em=100*off.exact_match_score(v['answer'],gold[r['id']]['answer']);assert abs(f1-v['f1'])<1e-8 and em==v['em'];training_scores+=1
E=list(map(json.loads,(C/'E5.jsonl').read_text().splitlines()));stats=json.loads((C/'reports/E5_statistics.json').read_text())
for effect in stats['effects']:
 reg=effect['regime'];metric=effect['metric'];baseline=effect['contrast'].split(' - ')[1];d=[]
 for r in E:
  l=np.mean([r['conditions'][f'{reg}_Learned_seed{s}'][metric] for s in [7,19,31]])
  b=np.mean([r['conditions'][f'{reg}_MatchedRandom_seed{s}'][metric] for s in [7,19,31]]) if baseline=='MatchedRandom' else r['conditions'][f'{reg}_{baseline}'][metric];d.append(l-b)
 assert abs(np.mean(d)-effect['effect'])<1e-10
# Links and images must resolve locally. Browser separately checks rendering.
class Parser(HTMLParser):
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if (tag=='a' and k=='href') or (tag=='img' and k=='src'):
    if not v.startswith(('http:','https:','#','data:')):assert (O/v.split('#')[0]).exists(),v
for p in O.glob('*.html'):Parser().feed(p.read_text())
manifest=json.loads((O/'final_manifest.json').read_text())
for p,h in manifest['inputs_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
for p,h in manifest['derived_artifacts'].items():assert hashlib.sha256((O/p).read_bytes()).hexdigest()==h,p
print(json.dumps({'main_condition_rows':len(F),'condition_arms':F.groupby(['experiment','condition']).ngroups,'training_validation_scores_independently_verified':training_scores,'canonical_E5_contrast_point_estimates_verified':len(stats['effects']),'E3_v2_screened':300,'E3_v2_eligible':299,'E3_v2_scores_and_contrasts':'PASS','input_and_output_hashes':'PASS','local_html_links':'PASS'},indent=2))
