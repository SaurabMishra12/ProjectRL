"""Recompute saved-data diagnostics only; never constructs a provider client."""
import os,sys,json,math,hashlib,importlib.util,itertools,shutil
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/saurab/Documents/projectRLMAIN/WEBExtension')
OUT=Path(__file__).resolve().parent;(OUT/'figures').mkdir(parents=True,exist_ok=True);(OUT/'tables').mkdir(exist_ok=True);(OUT/'sources').mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT));sys.modules.setdefault('ujson',json)
from protocol import file_hash,digest,paired_effect,ExactTokenizer
from controller_splits import load_partition,assert_disjoint,document_keys
from analysis.gold_strata import classify_gold,contrasts
from analysis.durable_storage import original_unresolved_attempts
spec=importlib.util.spec_from_file_location('official_hotpot',ROOT/'references/hotpot_evaluate_v1.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
A=json.loads((ROOT/'artifacts/active_machine_runs.json').read_text());MAIN=Path(A['controlled_run']);CTRL=Path(A['controller_study']);SPLIT=Path(A['controller_split_dir'])
plan=json.loads((CTRL/'manifest.json').read_text());cfg=plan['base'];settings=plan['settings']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':180})
BLUE='#2166ac';ORANGE='#b85c00';GRAY='#777777'
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,np.floating)):x=x.item()
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def dump(name,obj):(OUT/name).write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def csv(name,data):pd.DataFrame(data).to_csv(OUT/'tables'/name,index=False)
def save(fig,name):
 fig.savefig(OUT/'figures'/(name+'.png'),bbox_inches='tight');fig.savefig(OUT/'figures'/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def mean_ci(values,seed=42):
 x=np.asarray(values,float);rng=np.random.default_rng(seed)
 b=rng.choice(x,(5000,len(x)),replace=True).mean(axis=1)
 return float(x.mean()),*np.quantile(b,[.025,.975]).tolist()
def wilson(k,n):
 z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return c-h,c+h
GOLD={r['id']:r for r in load_partition(SPLIT,'heldout',gold=True)}
parts={n:load_partition(SPLIT,n,gold=True) for n in ['train','validation','heldout','pilot']}
leakage=assert_disjoint(parts)
old=ROOT/'artifacts/controller_splits_e219bbccd74a5888'
leakage_info={'counts':{k:len(v) for k,v in parts.items()},'types':{k:dict(Counter(r['type'] for r in v)) for k,v in parts.items()},'pairwise_checks':leakage,'heldout_bytes_unchanged':{s:file_hash(SPLIT/('heldout_'+s+'.json'))==file_hash(old/('heldout_'+s+'.json')) for s in ['gold','inputs']},'scope':'Exact IDs, normalized questions and NFKC/casefold candidate titles only; not semantic near duplicates or pretraining contamination'}
DATA={e:rows((CTRL if e=='E5' else MAIN)/(e+'.jsonl')) for e in ['E1','E2','E3','E4','E5']}
integrity={};inputs={r['id'] for r in json.loads((MAIN/'sample_inputs.json').read_text())}
for e,rr in DATA.items():
 p=(CTRL if e=='E5' else MAIN)/(e+'.jsonl');st=json.loads((p.parent/(e+'_status.json')).read_text());ids=[r['id'] for r in rr]
 integrity[e]={'n':len(rr),'target':300,'unique_ids':len(set(ids)),'ids_within_frozen_test':set(ids)<=inputs,'status':st['status'],'sha256':file_hash(p),'complete_sha_matches_status':file_hash(p)==st.get('sha256') if st['status']=='COMPLETE' else None}
 assert len(ids)==len(set(ids)) and set(ids)<=inputs
 assert len(rr)==300 and st['status']=='COMPLETE' and integrity[e]['complete_sha_matches_status']
checks={n:file_hash(ROOT/n)==sha for n,sha in plan['sources'].items()}
gate=json.loads((CTRL/'evaluation_gate.json').read_text());checkpoints={s:file_hash(CTRL/f'seed_{s}/selected.pt')==h for s,h in gate['checkpoint_hashes'].items()}
assert all(checks.values()) and all(checkpoints.values())
strata={};retrieval=[]
for row in DATA['E1']:
 gold=GOLD[row['id']];state=row['states']['P'];cl=classify_gold(gold,state['assignment']);strata[row['id']]=cl['stratum'];titles={t for t,_ in gold['supporting_facts']}
 ct={d['title'] for d in gold['documents']};rt={d['title'] for d in state['pool']};parts_titles=[{d['title'] for d in ds} for ds in state['assignment']]
 retrieval.append({'id':row['id'],'type':row['type'],'stratum':cl['stratum'],'gold_titles_in_candidates':len(titles&ct),'gold_titles_in_top6':len(titles&rt),'candidate_count':len(gold['documents']),'top6_count':len(state['pool']),'P_exclusive_union_matches_pool':set.union(*parts_titles)==rt and sum(map(len,parts_titles))==len(rt)})
assert all(r['P_exclusive_union_matches_pool'] for r in retrieval)
flat=[];mismatches=[];deliveries={};cross_suite=[]
for e,rr in DATA.items():
 for row in rr:
  gold=GOLD[row['id']]
  for label,c in row['conditions'].items():
   f,ap,ar=official.f1_score(c['answer'],gold['answer']);em=100*official.exact_match_score(c['answer'],gold['answer'])
   if abs(100*f-c['f1'])>1e-8 or abs(em-c['em'])>1e-8:mismatches.append([e,row['id'],label])
   coalition=([c['responses'][1]] if e=='E3' else [v for v in c['responses'] if official.normalize_answer(v['answer'])==official.normalize_answer(c['answer'])])
   pred={tuple(z) for v in coalition for z in v['citations']};gs=set(map(tuple,gold['supporting_facts']));tp=len(pred&gs);sp=tp/max(1,len(pred));sr=tp/max(1,len(gs));sf=2*sp*sr/(sp+sr) if sp+sr else 0;sem=int(pred==gs);jp=ap*sp;jr=ar*sr;jf=2*jp*jr/(jp+jr) if jp+jr else 0
   rec={'experiment':e,'id':row['id'],'type':row['type'],'condition':label,'stratum':strata[row['id']],'scope':'COMPLETE_300',
        **{k:v for k,v in c.items() if isinstance(v,(int,float,bool))},**{k:v for k,v in c['cost'].items() if isinstance(v,(int,float,bool)) or v is None},
        'support_label_f1':100*sf,'support_label_em':100*sem,'label_joint_f1':100*jf,'label_joint_em':em*sem,
        'new_support_facts_delivered':sum(c['new_support_facts_delivered_by_node']),'provider_refusal_question':int(c.get('provider_refusal_count',0)>0),
        'noanswer':int(official.normalize_answer(c['answer'])=='noanswer'),'explicit_empty_abstention':int(not c['answer'].strip())}
   flat.append(rec)
   for d in c['deliveries']:deliveries[d['text']]=d['tokens']
assert not mismatches
F=pd.DataFrame(flat);F.to_csv(OUT/'tables/all_condition_question_metrics.csv',index=False)
summary=[]
for (e,label),df in F.groupby(['experiment','condition']):
 f,lo,hi=mean_ci(df.f1);ec=wilson(int((df.em==100).sum()),len(df))
 rec={'experiment':e,'condition':label,'n':len(df),'scope':df.scope.iloc[0],'F1':f,'F1_CI_low':lo,'F1_CI_high':hi,'EM':df.em.mean(),'EM_Wilson_low':100*ec[0],'EM_Wilson_high':100*ec[1]}
 for col in df.select_dtypes(include=['number','bool']).columns:
  if col in ['f1','em']:continue
  rec[col+'_known_mean']=df[col].mean();rec[col+'_missing_n']=int(df[col].isna().sum())
  if col in ['input_tokens','output_tokens']:rec[col+'_all_question_mean']=df[col].mean() if df[col].notna().all() else None
 summary.append(rec)
S=pd.DataFrame(summary);S.to_csv(OUT/'tables/condition_summary.csv',index=False)
# Seed averages are formed within each question; never count n*3 as independent.
E5mean=[]
for regime in ['R','P']:
 for row in DATA['E5']:
  for method in ['Learned','MatchedRandom']:
   df=F[(F.experiment=='E5')&(F.id==row['id'])&F.condition.isin([regime+f'_{method}_seed{s}' for s in settings['training_seeds']])]
   rec={'id':row['id'],'type':row['type'],'stratum':strata[row['id']],'condition':regime+'_'+method+'_seed_mean'}
   for col in df.select_dtypes(include=['number','bool']).columns:rec[col]=df[col].mean() if df[col].notna().all() else None
   E5mean.append(rec)
M=pd.DataFrame(E5mean);M.to_csv(OUT/'tables/E5_complete_seed_averages.csv',index=False)
new_effects=[]
for label,aa,bb in contrasts(DATA['E5'],'E5',settings):
 new_effects.append({'experiment':'E5','contrast':label,'scope':'COMPLETE_300_EXPLORATORY',**paired_effect(aa,bb,draws=5000)})
# Preserved published-script inferential estimates, independently checked point estimates.
effects=[];stratum_effects=[]
for e in ['E1','E2','E3','E4']:
 st=json.loads((MAIN/'reports'/e/'statistics.json').read_text())
 effects += [{'experiment':e,**x} for x in st['effects']]
 strat=json.loads((MAIN/'reports'/e/'gold_strata/statistics.json').read_text());stratum_effects += [{'experiment':e,**x} for x in strat['effects']]
for e in ['E1','E2','E3','E4']:
 for label,aa,bb in contrasts(DATA[e],e):
  source=next((x for x in effects if x['experiment']==e and x['contrast']==label),None)
  if source:assert abs(np.mean([aa[q]-bb[q] for q in aa])-source['effect'])<1e-8
# Transitions for every same-question comparison; EM transitions are explicit.
transitions=[]
comparisons={'E1':[('P_CW','P_NoComm'),('R_CW','R_NoComm'),('P_CW','P_SelfUpdate'),('R_CW','R_SelfUpdate')], 'E2':[('R_CW','R_CCW')],'E3':[('R_prefix','R_crop')],'E4':[('P_stopped','P_final')]}
for e,pairs in comparisons.items():
 for aa,bb in pairs:
  for row in DATA[e]:
   x,y=row['conditions'][aa],row['conditions'][bb];transitions.append({'experiment':e,'contrast':aa+' - '+bb,'id':row['id'],'stratum':strata[row['id']],'F1_change':x['f1']-y['f1'],'correct_to_wrong':int(y['em']==100 and x['em']<100),'wrong_to_correct':int(y['em']<100 and x['em']==100),'both_correct':int(x['em']==y['em']==100),'both_wrong':int(x['em']<100 and y['em']<100)})
T=pd.DataFrame(transitions);T.to_csv(OUT/'tables/paired_question_transitions.csv',index=False)
# Invariants for source identity and exact delivery budgets.
e1={r['id']:r for r in DATA['E1']}
e2_same=all(e1[r['id']]['conditions'][k]==r['conditions'][k] for r in DATA['E2'] for k in ['R_CW','R_CCW'])
e2_budgets=all(r['conditions']['R_CW']['payload_tokens']==r['conditions']['R_CCW']['payload_tokens'] and r['conditions']['R_CW']['cost']['logical_calls']==r['conditions']['R_CCW']['cost']['logical_calls'] for r in DATA['E2'])
e3_equal=sum(r['crop_equals_prefix'] for r in DATA['E3']);e3_budget=all(r['conditions']['R_prefix']['payload_tokens']==r['conditions']['R_crop']['payload_tokens'] for r in DATA['E3'])
stop=DATA['E4'];depths=dict(Counter(r['stop_depth'] for r in stop));savedcalls=sum(r['hypothetical_calls_avoided'] for r in stop);early=sum(r['stop_depth']<3 for r in stop);wrong=sum(r['wrong_unanimity'] for r in stop)
matched=[];actionrows=[]
for regime in ['R','P']:
 for seed in settings['training_seeds']:
  aa=[]
  for row in DATA['E5']:
   c=row['conditions'][regime+f'_Learned_seed{seed}'];b=row['conditions'][regime+f'_MatchedRandom_seed{seed}'];aa.append(row['learned_action_indices'][regime+f'_seed{seed}'])
   matched.append({'id':row['id'],'regime':regime,'seed':seed,'edges_equal':len(c['action']['edges'])==len(b['action']['edges']),'quota_equal':c['action']['quota']==b['action']['quota'],'calls_equal':c['cost']['logical_calls']==b['cost']['logical_calls'],'payload_equal':c['payload_tokens']==b['payload_tokens']})
  counts=Counter(aa);probs=np.array(list(counts.values()))/len(aa)
  actionrows.append({'regime':regime,'seed':seed,'n':len(aa),'unique_actions':len(counts),'top_action':counts.most_common(1)[0][0],'top_action_fraction':max(counts.values())/len(aa),'NoComm_fraction':counts.get(0,0)/len(aa),'entropy_bits':float(-np.sum(probs*np.log2(probs))),'frequencies':dict(counts)})
J=pd.DataFrame(matched);J.to_csv(OUT/'tables/E5_complete_resource_matching.csv',index=False);csv('E5_complete_action_adaptivity.csv',actionrows)
learning=[]
for seed in settings['training_seeds']:
 for ep in range(settings['collection_epochs']):
  folder=CTRL/f'seed_{seed}';v=json.loads((folder/f'validation_metrics_epoch_{ep}.json').read_text());o=json.loads((folder/f'optimizer_epoch_{ep}.json').read_text());rr=rows(folder/f'rollout_epoch_{ep}.jsonl');sel=json.loads((folder/'selection.json').read_text())
  learning.append({'seed':seed,'epoch':ep+1,**v,**{'optimizer_'+k:v for k,v in o.items()},'train_mean_reward':float(np.mean([r['reward'] for r in rr])),'train_mean_f1':float(np.mean([r['result']['f1'] for r in rr])),'train_rollouts':len(rr),'selected_epoch':sel['epoch']+1,'train_action_entropy_bits':float(-sum((n/len(rr))*math.log2(n/len(rr)) for n in Counter(r['action_index'] for r in rr).values()))})
csv('controller_learning_curves.csv',learning)
# Physical costs: independent known dimensions, globally de-duplicated original attempt IDs.
by_attempt={};ledger_sources={}
for p in (ROOT/'artifacts').glob('**/calls.jsonl'):
 for r in rows(p):
  uid=r.get('attempt_id')
  if not uid:continue
  by_attempt.setdefault(uid,{}).setdefault(r['event'],r);ledger_sources.setdefault(uid,set()).add(str(p))
physical=[]
for uid,ev in by_attempt.items():
 start=ev.get('started');succ=ev.get('success');u=(succ or {}).get('usage') or {};term=next((k for k in ['success','rejected','reconciled_not_generated'] if k in ev),'uncertain')
 physical.append({'attempt_id':uid,'key':(start or succ or {}).get('key'),'outcome':term,'input_tokens_reported':u.get('prompt_tokens'),'output_tokens_reported':u.get('completion_tokens'),'provider_refusal':bool((succ or {}).get('provider_refusal')),'started_utc':(start or {}).get('utc'),'reported_end_utc':(succ or ev.get('uncertain') or ev.get('rejected') or {}).get('utc'),'source_ledgers':' | '.join(sorted(ledger_sources[uid]))})
P=pd.DataFrame(physical);P.to_csv(OUT/'tables/physical_attempts_deduplicated.csv',index=False)
physical_summary={'scope':'All original attempt IDs in WEBExtension/artifacts, including archived caps, pilots and preserved prior study; ledger copies de-duplicated by attempt ID','unique_attempts':len(P),'outcomes':dict(Counter(P.outcome)),'reported_input_tokens_lower_bound':int(P.input_tokens_reported.sum()),'reported_output_tokens_lower_bound':int(P.output_tokens_reported.sum()),'attempts_missing_input_usage':int(P.input_tokens_reported.isna().sum()),'attempts_missing_output_usage':int(P.output_tokens_reported.isna().sum()),'refusals':int(P.provider_refusal.sum()),'monetary_cost':None,'reason':'No verified deployment price; missing/refusal/error usage remains unknown, not zero'}
active_cost=[]
for label,run in [('controlled',MAIN),('controller',CTRL/'inference')]:
 lr=rows(run/'calls.jsonl');success=[r for r in lr if r['event']=='success'];inputs_known=[r['usage']['prompt_tokens'] for r in success if (r.get('usage') or {}).get('prompt_tokens') is not None];outputs_known=[r['usage']['completion_tokens'] for r in success if (r.get('usage') or {}).get('completion_tokens') is not None]
 active_cost.append({'pipeline':label,'successes':len(success),'reported_input_tokens':sum(inputs_known),'successful_inputs_missing':len(success)-len(inputs_known),'reported_output_tokens_lower_bound':sum(outputs_known),'successful_outputs_missing':len(success)-len(outputs_known),'original_unresolved_attempts':len(original_unresolved_attempts(lr)),'attempts':sum(r['event']=='started' for r in lr)})
csv('active_pipeline_physical_usage.csv',active_cost)
network=[]
for parent in [MAIN/'network_attempts',CTRL/'inference/network_attempts']:
 for p in parent.glob('*.json'):
  if p.name.endswith(('.receipt.json','.dispatch.json')):continue
  r=json.loads(p.read_text());network.append({'pipeline':'controlled' if parent.parent==MAIN else 'controller','attempt_id':r['attempt_id'],'http_status':r.get('http_status'),'error_type':r.get('error_type'),'network_wall_seconds':r.get('network_wall_seconds'),'end_to_end_wall_seconds':r.get('end_to_end_wall_seconds'),'utc_dispatch':r.get('utc_dispatch'),'utc_received':r.get('utc_received')})
N=pd.DataFrame(network);N.to_csv(OUT/'tables/network_timings.csv',index=False)
sessionrows=[]
for parent in [MAIN,CTRL]:
 for folder in parent.glob('runtime_sessions/*'):
  st=folder/'session_start.json';end=folder/'session_end.json'
  if st.exists():
   ss=json.loads(st.read_text());ee=json.loads(end.read_text()) if end.exists() else {};sessionrows.append({'pipeline':parent.name,'session':folder.name,'start_utc':ss.get('utc'),'end_utc':ee.get('utc'),'status':ee.get('status'),'wall_seconds_observed':ee.get('wall_seconds')})
csv('worker_session_timings.csv',sessionrows)
# Actual tokenizer re-count on unique recorded deliveries, never generated replacements.
os.environ['HF_HUB_OFFLINE']='1';tokenizer=ExactTokenizer(cfg['tokenizer'],cfg['tokenizer_revision']);token_errors=[]
for text,n in deliveries.items():
 if tokenizer.count(text)!=n:token_errors.append({'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'stored':n,'actual':tokenizer.count(text)})
csv('payload_token_verification_errors.csv',token_errors)
# Conservative missing-test F1 bounds: honest identified ranges, no filled rows.
missing=300-len(DATA['E5']);partial_bounds=[]
for label,df in F[F.experiment=='E5'].groupby('condition'):
 total=df.f1.sum();partial_bounds.append({'condition':label,'observed_n':len(df),'missing_n':missing,'observed_mean_F1':df.f1.mean(),'full300_worst_case_bound':total/300,'full300_best_case_bound':(total+100*missing)/300})
csv('E5_complete_missing_data_bounds.csv',partial_bounds)
# Scientific figures: separate labelled panels and constrained layout, no point-label collisions.
fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
for ax,e in zip(axes.flat,['E1','E2','E3','E4']):
 d=S[S.experiment==e].sort_values('condition');y=np.arange(len(d));ax.barh(y,d.F1,color=BLUE,alpha=.85);ax.errorbar(d.F1,y,xerr=[d.F1-d.F1_CI_low,d.F1_CI_high-d.F1],fmt='none',ecolor='black',capsize=3);ax.set_yticks(y,d.condition);ax.set(xlabel='Mean answer F1 (0–100), paired-question bootstrap CI',title=e+' · complete n=300');ax.set_xlim(0,max(45,float(d.F1_CI_high.max())+5))
save(fig,'01_complete_answer_quality')
fig,ax=plt.subplots(figsize=(11,7),layout='constrained');y=np.arange(len(effects));v=np.array([d['effect'] for d in effects]);lo=np.array([d['ci95'][0] for d in effects]);hi=np.array([d['ci95'][1] for d in effects]);ax.errorbar(v,y,xerr=[v-lo,hi-v],fmt='o',color=BLUE,capsize=4);ax.axvline(0,color=GRAY,lw=1);ax.set_yticks(y,[d['experiment']+' '+d['contrast'] for d in effects]);ax.set(xlabel='Paired mean F1 change (percentage points), 95% bootstrap CI',title='Complete E1–E4 contrasts; only the E1 interaction is primary');save(fig,'02_paired_effects')
selected=[d for d in stratum_effects if d['experiment']=='E1' and d.get('status')=='ESTIMATED' and d['contrast'] in ['P_CW - NoComm','(CW-NoComm)_P - (CW-NoComm)_R']]
fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained');cnt=Counter(strata.values());axes[0].barh(list(cnt),list(cnt.values()),color=BLUE);axes[0].set(xlabel='Questions (n=300)',title='Gold location after actual BM25 top-6 retrieval')
for i,d in enumerate(selected):axes[1].errorbar(d['effect'],i,xerr=[[d['effect']-d['ci95'][0]],[d['ci95'][1]-d['effect']]],fmt='o',color=BLUE,capsize=3)
axes[1].set_yticks(range(len(selected)),[d['stratum']+' · '+('P gain' if d['contrast'].startswith('P') else 'interaction') for d in selected]);axes[1].axvline(0,color=GRAY);axes[1].set(xlabel='F1 change (points), exploratory 95% CI',title='Communication effects by observed gold stratum');save(fig,'03_gold_strata')
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');R=pd.DataFrame(retrieval);cross=pd.crosstab(R.gold_titles_in_candidates,R.gold_titles_in_top6).reindex(index=[0,1,2],columns=[0,1,2],fill_value=0);im=axes[0].imshow(cross,cmap='Blues',vmin=0)
for (i,j),v in np.ndenumerate(cross.to_numpy()):axes[0].text(j,i,str(v),ha='center',va='center',color='white' if v>cross.values.max()/2 else 'black')
axes[0].set(xticks=[0,1,2],yticks=[0,1,2],xlabel='Gold article count after top-6 selection',ylabel='Gold article count in publisher candidates',title='Retrieval availability (question counts)');fig.colorbar(im,ax=axes[0],label='Questions')
for col,lab,color in [('gold_titles_in_candidates','Publisher candidates',GRAY),('gold_titles_in_top6','BM25 top-6',BLUE)]:
 vals=R[col].value_counts().reindex([0,1,2],fill_value=0);offset=-.18 if color==GRAY else .18;axes[1].bar(np.array([0,1,2])+offset,vals,width=.36,label=lab,color=color)
axes[1].set(xticks=[0,1,2],xlabel='Available gold articles',ylabel='Questions (n=300)',title='Source availability vs selection loss');axes[1].legend();save(fig,'04_retrieval_audit')
fig,ax=plt.subplots(figsize=(12,5),layout='constrained');group=T.groupby(['experiment','contrast']);labs=[];helped=[];harmed=[];unchanged=[]
for (e,c),d in group:labs.append(e+' '+c);helped.append((d.F1_change>1e-9).sum());harmed.append((d.F1_change<-1e-9).sum());unchanged.append((d.F1_change.abs()<=1e-9).sum())
y=np.arange(len(labs));ax.barh(y,helped,color=BLUE,label='F1 improves');ax.barh(y,harmed,left=helped,color=ORANGE,label='F1 worsens');ax.barh(y,unchanged,left=np.array(helped)+harmed,color='#dddddd',label='Unchanged');ax.set_yticks(y,labs);ax.set(xlabel='Questions (300 per comparison)',title='Paired outcomes: improvements, harms and ties');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.2),ncol=3);save(fig,'05_paired_transitions')
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');d=S[S.experiment=='E1'];labels=[]
for i,(_,v) in enumerate(d.iterrows()):axes[0].scatter(v.payload_tokens_known_mean,v.F1,s=70,marker='o');axes[1].scatter(v.logical_calls_known_mean,v.F1,s=70,marker='o');labels.append(v.condition)
# Numeric keys in a separate legend/table prevent annotation overlap.
for ax in axes:
 for i,(_,v) in enumerate(d.iterrows()):
  x=v.payload_tokens_known_mean if ax is axes[0] else v.logical_calls_known_mean;ax.annotate(str(i+1),(x,v.F1),xytext=(5,4),textcoords='offset points',fontsize=8)
 ax.set_ylabel('Mean answer F1 (0–100)')
axes[0].set(xlabel='Mean transmitted payload tokens/question',title='E1 payload–quality tradeoff');axes[1].set(xlabel='Logical calls/question (shared initial calls included)',title='E1 call–quality tradeoff');fig.suptitle('Point key: '+'; '.join(f'{i+1}={s}' for i,s in enumerate(labels)),fontsize=9);save(fig,'06_resource_frontiers')
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');axes[0].bar(['Identical payload','Changed payload'],[e3_equal,300-e3_equal],color=[GRAY,BLUE]);axes[0].set(ylabel='Questions (n=300)',title='E3 intervention fidelity')
lengths=[r['conditions']['R_prefix']['payload_tokens'] for r in DATA['E3']];axes[1].hist(lengths,bins=20,color=BLUE);axes[1].axvline(cfg['payload_budget'],color=ORANGE,label='Declared quota');axes[1].set(xlabel='Actual one-edge payload tokens',ylabel='Questions',title='Short messages make most crop arms identical');axes[1].legend();save(fig,'07_content_intervention')
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');axes[0].bar(range(4),[depths.get(i,0) for i in range(4)],color=BLUE);axes[0].set(xticks=range(4),xlabel='Selected stopping depth (0=initial)',ylabel='Questions',title=f'E4: {early}/300 stop early; {wrong} wrong early consensuses')
for ep in range(4):
 vals=[r['trajectory'][ep]['f1'] for r in stop];m,l,h=mean_ci(vals);axes[1].errorbar(ep,m,yerr=[[m-l],[h-m]],fmt='o',color=BLUE,capsize=4)
axes[1].set(xticks=range(4),xlabel='Communication round',ylabel='Mean answer F1 (0–100)',title=f'Shared trajectory; {savedcalls}/3600 calls hypothetically avoided');save(fig,'08_stopping_trajectory')
L=pd.DataFrame(learning);fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
for seed,g in L.groupby('seed'):
 for ax,col,title in zip(axes,['mean_f1','mean_utility_gain','train_mean_reward'],['Validation answer F1 (0–100)','Validation communication utility gain','Mean training reward']):ax.plot(g.epoch,g[col],marker='o',label=f'Seed {seed}');ax.set(xlabel='Collection epoch',ylabel=title,xticks=[1,2,3],title=title)
for ax in axes:ax.legend()
save(fig,'09_controller_training')
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for ax,regime in zip(axes,['R','P']):
 d=F[(F.experiment=='E5')&F.condition.isin([regime+'_'+x for x in ['NoComm','SelfUpdate','Static','Broadcast']])].groupby('condition').f1.mean();labels=d.index.tolist();vals=d.tolist()
 for method in ['Learned','MatchedRandom']:
  v=M[M.condition==regime+'_'+method+'_seed_mean'];labels.append(regime+'_'+method+'_seed_mean');vals.append(v.f1.mean())
 ax.barh(labels,vals,color=BLUE);ax.set(xlabel='Held-out mean answer F1 (0–100)',title=f'E5 {regime} · complete {len(DATA["E5"])}/300')
save(fig,'10_E5_complete_quality')
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for ax,regime in zip(axes,['R','P']):
 g=[r for r in actionrows if r['regime']==regime];y=np.arange(len(g));ax.bar(y,[r['top_action_fraction']*100 for r in g],width=.4,label='Dominant action',color=BLUE);ax.bar(y+.4,[r['NoComm_fraction']*100 for r in g],width=.4,label='NoComm',color=GRAY);ax.set(xticks=y+.2,xticklabels=[f'Seed {r["seed"]}\n{r["unique_actions"]} actions' for r in g],ylabel='Held-out questions (%)',title=f'E5 {regime}: held-out policy adaptivity');ax.legend()
save(fig,'11_E5_complete_action_adaptivity')
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');success=N[N.http_status==200]
for label,d in success.groupby('pipeline'):axes[0].hist(d.network_wall_seconds.dropna(),bins=35,alpha=.5,label=label)
axes[0].set(xlabel='Network round-trip seconds (new saved timing scope)',ylabel='Returned responses',title='Physical latency; excludes queue and cache time');axes[0].set_xscale('log');axes[0].legend()
disp=Counter(P.outcome);axes[1].barh(list(disp),list(disp.values()),color=BLUE);axes[1].set(xlabel='Unique physical attempt IDs (all preserved runs)',title='Provider attempt disposition; unknown charges retained');save(fig,'12_latency_and_accounting')
fig,ax=plt.subplots(figsize=(12,5),layout='constrained');selected=[d for d in new_effects if 'MatchedRandom' not in d['contrast']]
y=range(len(selected));v=np.array([d['effect'] for d in selected]);lo=np.array([d['ci95'][0] for d in selected]);hi=np.array([d['ci95'][1] for d in selected]);ax.errorbar(v,list(y),xerr=[v-lo,hi-v],fmt='o',color=BLUE,capsize=3);ax.axvline(0,color=GRAY);ax.set_yticks(list(y),[d['contrast'] for d in selected]);ax.set(xlabel='Paired seed-mean F1 difference (points), exploratory CI',title=f'E5 Complete held-out n={len(DATA["E5"])}; exploratory whole-policy comparison');save(fig,'13_E5_complete_paired_effects')
# Source/source hashes and immutable evidence references for every derived artifact.
report={'utc':datetime.now(timezone.utc).isoformat(),'verdict':'NOT_READY_FOR_A_STRONG_WWW2027_SUBMISSION','runs_stopped':True,'no_inference_requested':True,'integrity':integrity,'source_hash_checks':checks,'checkpoint_hash_checks':checkpoints,'score_mismatches':mismatches,'token_count_mismatches':token_errors,'unique_payloads_recounted':len(deliveries),'leakage':leakage_info,'stratum_counts':dict(Counter(strata.values())),'retrieval_availability':{'candidate_gold_count':dict(Counter(R.gold_titles_in_candidates)),'top6_gold_count':dict(Counter(R.gold_titles_in_top6))},'complete_effects':effects,'stratum_effects':stratum_effects,'complete_E5_effects':new_effects,'manipulations':{'E2_matches_E1_R_conditions_exactly':e2_same,'E2_tokens_and_calls_matched':e2_budgets,'E3_identical_payload_questions':e3_equal,'E3_budget_matched':e3_budget,'E4_depth_counts':depths,'E4_early_stops':early,'E4_wrong_early_stops':wrong,'E4_hypothetical_calls_avoided':savedcalls,'E4_full_logical_calls':3600,'E4_saving_fraction':savedcalls/3600,'E5_resource_matching':{'comparisons':len(J),'edges_equal':int(J.edges_equal.sum()),'quota_equal':int(J.quota_equal.sum()),'call_count_mismatches':int((~J.calls_equal).sum()),'payload_mismatches':int((~J.payload_equal).sum())}},'controller_learning':learning,'E5_complete_action_statistics':actionrows,'physical_usage_global':physical_summary,'physical_usage_active':active_cost,'sources':{'official_evaluator_sha256':file_hash(ROOT/'references/hotpot_evaluate_v1.py'),'main_manifest_sha256':file_hash(MAIN/'frozen_manifest.json'),'controller_manifest_sha256':file_hash(CTRL/'manifest.json'),'split_manifest_sha256':file_hash(SPLIT/'manifest.json')},'limits':['E5 contains all 300 frozen test questions; its contrasts are exploratory relative to the E1 primary endpoint','Only one model deployment, one benchmark and one primary partition/role/crop seed; controller optimizer seeds are not independent backbone or partition replications','Input and output known token totals are counted independently; global totals include archived runs, not just the active primary study','Model deployment alias has no immutable provider snapshot; no pretraining contamination guarantee','Supporting-label agreement/joint diagnostics do not establish textual entailment or citation faithfulness','No predeclared noninferiority margin; nonsignificance and identical observed F1 do not prove general safety','Data subsets reused across optimizer seeds/epochs and test questions reused across paired experiments are intentional; independent-unit count stays at questions','Publication readiness is an evidence assessment, not a guarantee of acceptance or novelty completeness']}
dump('audit_metrics.json',report);dump('paired_effects.json',effects);dump('E5_COMPLETE_exploratory_effects.json',new_effects);dump('split_overlap_audit.json',leakage_info);dump('physical_usage_lower_bounds.json',physical_summary);csv('gold_retrieval_question_audit.csv',retrieval)
for f in ['protocol.py','experiments.py','controller_study.py','controller_splits.py','learned_controller.py','backend.py','reporting.py','controller_reporting.py','analysis/gold_strata.py']:
 target=OUT/'sources'/f.replace('/','__');shutil.copyfile(ROOT/f,target)
shutil.copyfile(ROOT/'references/hotpot_evaluate_v1.py',OUT/'sources/hotpot_evaluate_v1.py')
dump('analysis_artifact_manifest.json',{'utc':report['utc'],'derived_artifacts':{str(p.relative_to(OUT)):file_hash(p) for p in OUT.rglob('*') if p.is_file() and p.name!='analysis_artifact_manifest.json'},'input_results':{e:file_hash((CTRL if e=='E5' else MAIN)/(e+'.jsonl')) for e in DATA},'inference_calls_issued_by_analysis':0})
print(json.dumps({'verdict':report['verdict'],'integrity':{e:x['n'] for e,x in integrity.items()},'score_mismatches':len(mismatches),'payload_token_mismatches':len(token_errors),'figures':len(list((OUT/'figures').glob('*.png'))),'manipulations':report['manipulations'],'active_cost':active_cost},indent=2))
