"""Offline post-hoc diagnostics. No model client or inference requests."""
import json,sys,math,hashlib,shutil
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;R=O.parents[1];sys.path.insert(0,str(R))
from protocol import paired_effect,file_hash
A=json.loads((R/'artifacts/active_machine_runs.json').read_text());C=Path(A['controller_study']);B=Path(A['controlled_run']);split=Path(A['controller_split_dir'])
F=pd.read_csv(O/'tables/all_condition_question_metrics.csv');M=pd.read_csv(O/'tables/E5_complete_seed_averages.csv');rows={e:[json.loads(l) for l in ((C if e=='E5' else B)/(e+'.jsonl')).read_text().splitlines() if l] for e in ['E1','E2','E3','E4','E5']}
def csv(n,d):pd.DataFrame(d).to_csv(O/'tables'/n,index=False)
def dump(n,d):(O/n).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
# Full distribution catalog: every stored or recomputed scalar, every arm.
allstats=[]
for (e,c),d in F.groupby(['experiment','condition']):
 for metric in d.select_dtypes(include=['number','bool']):
  x=d[metric].dropna().astype(float)
  allstats.append(dict(experiment=e,condition=c,metric=metric,n=len(d),known_n=len(x),missing_n=int(d[metric].isna().sum()),mean=float(x.mean()) if len(x) else None,std=float(x.std(ddof=1)) if len(x)>1 else None,sum_known=float(x.sum()) if len(x) else None,**{k:float(x.quantile(q)) if len(x) else None for k,q in [('min',0),('p05',.05),('p25',.25),('median',.5),('p75',.75),('p95',.95),('max',1)]}))
csv('all_scalar_distribution_metrics.csv',allstats)
# Preserve canonical final inference estimates and all original statistical diagnostics.
orig=O/'original_reports';orig.mkdir(exist_ok=True)
for name,run in [('controlled',B),('controller',C)]:shutil.copytree(run/'reports',orig/name,dirs_exist_ok=True)
# Subgroup differences are exploratory, never substitute for frozen primary endpoint.
sub=[]
for regime in ['R','P']:
 learn=M[M.condition==regime+'_Learned_seed_mean'].set_index('id')
 for field in ['stratum','type']:
  for group,d in learn.groupby(field):
   for base in ['NoComm','SelfUpdate','Static','Broadcast']:
    b=F[(F.experiment=='E5')&(F.condition==regime+'_'+base)].set_index('id').loc[d.index]
    diff=d.f1-b.f1;rng=np.random.default_rng(101);boot=rng.choice(diff.to_numpy(),(5000,len(diff)),replace=True).mean(1)
    sub.append({'regime':regime,'group_field':field,'group':group,'baseline':base,'n':len(diff),'learned_F1':d.f1.mean(),'baseline_F1':b.f1.mean(),'effect':diff.mean(),'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975),'helped':int((diff>1e-9).sum()),'harmed':int((diff< -1e-9).sum()),'ties':int((abs(diff)<=1e-9).sum()),'scope':'POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed'})
csv('E5_subgroup_diagnostics.csv',sub)
# Every arm by observed gold stratum and question type, descriptive only.
for field in ['stratum','type']:
 csv('all_arms_by_'+field+'.csv',F.groupby(['experiment','condition',field],dropna=False).agg(n=('id','size'),F1=('f1','mean'),EM=('em','mean'),payload=('payload_tokens','mean'),calls=('logical_calls','mean'),support_recall=('answer_support_recall','mean'),refused_questions=('provider_refusal_question','sum')).reset_index())
# Completion-prefix trend is an outcome-blind execution-order diagnostic, not model learning.
ordered=[x['id'] for x in rows['E5']];trend=[]
for end in [50,100,129,150,200,250,300]:
 ids=ordered[:end]
 for regime in ['R','P']:
  l=M[(M.id.isin(ids))&(M.condition==regime+'_Learned_seed_mean')]
  for base in ['NoComm','Broadcast']:
   b=F[(F.experiment=='E5')&F.id.isin(ids)&(F.condition==regime+'_'+base)]
   trend.append({'prefix_n':end,'regime':regime,'baseline':base,'learned_F1':l.f1.mean(),'baseline_F1':b.f1.mean(),'difference':l.f1.mean()-b.f1.mean()})
csv('execution_prefix_trends.csv',trend)
# Full seed distributions, F1/EM and resources; seed variation is descriptive n=3.
seeds=[]
for regime in ['R','P']:
 for seed in [7,19,31]:
  d=F[(F.experiment=='E5')&(F.condition==f'{regime}_Learned_seed{seed}')]
  seeds.append({'regime':regime,'seed':seed,'n':len(d),'F1':d.f1.mean(),'EM':d.em.mean(),'payload':d.payload_tokens.mean(),'calls':d.logical_calls.mean(),'utility':d.utility_gain.mean()})
csv('selected_seed_performance.csv',seeds)
# Raw factual casebook: extremes plus median positive/negative deltas; not representative anecdotes.
gold={r['id']:r for r in json.loads((split/'heldout_gold.json').read_text())};by={r['id']:r for r in rows['E5']};cases=[]
for regime,base in [('P','NoComm'),('P','Broadcast'),('R','NoComm')]:
 l=M[M.condition==regime+'_Learned_seed_mean'].set_index('id');b=F[(F.experiment=='E5')&(F.condition==regime+'_'+base)].set_index('id');diff=(l.f1-b.f1).sort_values();chosen=list(diff.head(3).index)+list(diff.tail(3).index)
 for q in dict.fromkeys(chosen):
  row=by[q];cases.append({'id':q,'selection_rule':'three smallest and three largest seed-mean F1 differences per listed comparison; ties by saved pandas ordering','comparison':regime+' Learned - '+base,'question':gold[q]['question'],'gold_answer':gold[q]['answer'],'type':gold[q]['type'],'stratum':l.loc[q,'stratum'],'delta':float(diff[q]),'baseline_answer':row['conditions'][regime+'_'+base]['answer'],'baseline_f1':row['conditions'][regime+'_'+base]['f1'],'learned':{str(s):{'answer':row['conditions'][regime+f'_Learned_seed{s}']['answer'],'f1':row['conditions'][regime+f'_Learned_seed{s}']['f1'],'action':row['conditions'][regime+f'_Learned_seed{s}']['action'],'payload':row['conditions'][regime+f'_Learned_seed{s}']['payload_tokens']} for s in [7,19,31]}})
dump('raw_output_casebook.json',cases)
# Additional audit: apparent perfect match to broadcast and required baseline comparisons.
dominance=[]
for regime in ['R','P']:
 for seed in [7,19,31]:
  same=eq=0
  for row in rows['E5']:
   x=row['conditions'][regime+f'_Learned_seed{seed}'];y=row['conditions'][regime+'_Broadcast'];same+=x['answer']==y['answer'];eq+=x['f1']==y['f1']
  dominance.append({'regime':regime,'seed':seed,'answers_equal_to_broadcast':same,'F1_equal_to_broadcast':eq,'n':300})
csv('learned_broadcast_identity.csv',dominance)
# Diagnostic refusal strata are treatment-dependent; never excluded from headline scores.
ref=[]
for (e,c),d in F.groupby(['experiment','condition']):
 for flag,g in d.groupby('provider_refusal_question'):
  ref.append({'experiment':e,'condition':c,'has_terminal_refusal':int(flag),'n':len(g),'F1':g.f1.mean(),'EM':g.em.mean(),'scope':'selection on treatment outcome; descriptive only'})
csv('refusal_descriptive_groups.csv',ref)
# Runtime quantiles and totals only from captured attempts; not arm-randomized performance.
N=pd.read_csv(O/'tables/network_timings.csv');nt=[]
for (p,status),d in N.groupby(['pipeline','http_status'],dropna=False):
 for col in ['network_wall_seconds','end_to_end_wall_seconds']:
  x=d[col].dropna();nt.append({'pipeline':p,'http_status':status if pd.notna(status) else 'unknown','metric':col,'recorded_n':len(x),'sum':x.sum(),'mean':x.mean(),'p50':x.quantile(.5),'p95':x.quantile(.95),'max':x.max()})
csv('network_timing_summary.csv',nt)
# Full numeric training/validation records; retain identifiers and raw source pointer.
training=[]
for seed in [7,19,31]:
 for phase in ['rollout','validation']:
  for epoch in range(3):
   p=C/f'seed_{seed}'/f'{phase}_epoch_{epoch}.jsonl'
   for line in p.read_text().splitlines():
    r=json.loads(line);v=r['result'];training.append({'seed':seed,'phase':phase,'epoch':epoch+1,'id':r['id'],'regime':r['regime'],'action_index':r['action_index'],**{k:x for k,x in r.items() if isinstance(x,(float,int,bool))},**{k:x for k,x in v.items() if isinstance(x,(float,int,bool))},**{k:x for k,x in v['cost'].items() if isinstance(x,(float,int,bool)) or x is None},'raw_source':str(p)})
csv('all_training_validation_metrics.csv',training)
# Readable plots, no point annotations that collide.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,n):
 for ext in ['png','pdf']:fig.savefig(O/'figures'/(n+'.'+ext),bbox_inches='tight',dpi=160)
 plt.close(fig)
D=pd.DataFrame(sub);fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for ax,base in zip(axes,['NoComm','Broadcast']):
 d=D[(D.regime=='P')&(D.group_field=='stratum')&(D.baseline==base)];y=np.arange(len(d));ax.errorbar(d.effect,y,xerr=[d.effect-d.ci95_low,d.ci95_high-d.effect],fmt='o',capsize=4);ax.axvline(0,color='gray');ax.set_yticks(y,[f'{r.group} (n={r.n})' for r in d.itertuples()]);ax.set(xlabel='Seed-mean F1 difference (points), marginal 95% CI',title='P: Learned minus '+base)
save(fig,'15_E5_subgroups')
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');T=pd.DataFrame(trend)
for ax,regime in zip(axes,['R','P']):
 for base,g in T[T.regime==regime].groupby('baseline'):ax.plot(g.prefix_n,g.difference,marker='o',label='Learned − '+base)
 ax.axhline(0,color='gray');ax.set(xlabel='Completed questions in frozen execution order',ylabel='Cumulative mean F1 difference (points)',title=regime+' diagnostic prefix trend');ax.legend()
save(fig,'16_prefix_drift')
# A complete E5 plot for each scalar metric.
for metric in F.select_dtypes(include=['number','bool']):
 fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
 for ax,regime in zip(axes,['R','P']):
  d=F[(F.experiment=='E5')&F.condition.str.startswith(regime+'_')].groupby('condition')[metric].mean();ax.barh(d.index,d.values,color='#2166ac');ax.set(xlabel=metric+' (known-value mean; see missingness table)',title='E5 '+regime+' • 300 questions per arm')
 save(fig,'E5_metric_'+metric)
print('Deep diagnostics complete')
