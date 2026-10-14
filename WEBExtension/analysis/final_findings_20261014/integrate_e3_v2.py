"""Offline E3-v2 integration; preserve original records and update paper bundle."""
import csv,json,hashlib,shutil,sys,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;R=O.parents[1];ARCHIVE=O.with_name('final_findings_20261014_before_E3_v2_20261015')
sys.path.insert(0,str(R))
from protocol import file_hash,digest,ExactTokenizer,delivered_facts,normalize,paired_effect
sys.modules.setdefault('ujson',json)
spec=importlib.util.spec_from_file_location('e3_official',R/'references/hotpot_evaluate_v1.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)

def dump(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def numeric(x,path=''):
 if isinstance(x,dict):
  for k,v in x.items():yield from numeric(v,path+'.'+k if path else k)
 elif isinstance(x,list):
  for i,v in enumerate(x):yield from numeric(v,f'{path}[{i}]')
 elif isinstance(x,(int,float,bool)):yield path,x

def ci(x):
 x=np.asarray(x,float);rng=np.random.default_rng(42);b=rng.choice(x,(5000,len(x)),replace=True).mean(axis=1)
 return float(x.mean()),*map(float,np.percentile(b,[2.5,97.5]))
def wilson(k,n):
 z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return 100*(c-h),100*(c+h)
def plot(fig,name):
 for ext in ['png','pdf']:fig.savefig(O/'figures'/(name+'.'+ext),dpi=160,bbox_inches='tight')
 plt.close(fig)

def prepare_tables():
 active=json.loads((R/'artifacts/active_machine_runs.json').read_text())['e3_corrected'];D=Path(active['run_dir'])
 m=json.loads((D/'frozen_manifest.json').read_text())['plan'];status=json.loads((D/'E3_status.json').read_text());rep=json.loads((D/'report.json').read_text())
 assert status['status']=='COMPLETE' and file_hash(D/'E3.jsonl')==status['sha256']==rep['result_sha256']
 assert file_hash(D/'frozen_manifest.json')==active['frozen_manifest_sha256']
 for name,h in m['sources'].items():assert file_hash(R/name)==h,name
 for name,h in m['source_files'].items():assert file_hash(Path(m['source_run'])/name)==h,name
 plans={p['id']:p for p in json.loads((D/'payload_plan.json').read_text())};assert digest(list(plans.values()))==m['payload_plan_sha256']
 rows=[json.loads(l) for l in (D/'E3.jsonl').read_text().splitlines()];assert len(rows)==len({r['id'] for r in rows})==300 and [r['id'] for r in rows]==m['question_ids']
 cfg=m['config'];t=ExactTokenizer(cfg['tokenizer'],cfg['tokenizer_revision']);assert t.identity==m['tokenizer']
 G={r['id']:r for r in json.loads((Path(m['source_run'])/'sample_gold.json').read_text())}
 oldF=pd.read_csv(ARCHIVE/'tables/all_condition_question_metrics.csv');strata=oldF[['id','stratum']].drop_duplicates().set_index('id').stratum.to_dict()
 flat=[];details=[];pairs=[];cases=[];mismatch=0
 for r in rows:
  q=plans[r['id']];assert digest(q)==r['payload_plan_hash'] and q['eligible']==r['eligible']
  if not r['eligible']:assert r['reason']=='NO_SENDER_EVIDENCE_AVAILABLE' and not q['source_text'];continue
  assert q['prefix']['text']!=q['crop']['text']
  for label,c in r['conditions'].items():
   f,ap,ar=official.f1_score(c['answer'],G[r['id']]['answer']);em=100*official.exact_match_score(c['answer'],G[r['id']]['answer']);assert abs(100*f-c['f1'])<1e-8 and em==c['em']
   response=c['responses'][1];pred=set(map(tuple,response['citations']));gold=set(map(tuple,G[r['id']]['supporting_facts']));sp=len(pred&gold)/max(1,len(pred));sr=len(pred&gold)/max(1,len(gold));sf=2*sp*sr/(sp+sr) if sp+sr else 0;sem=int(pred==gold);jp=ap*sp;jr=ar*sr;jf=2*jp*jr/(jp+jr) if jp+jr else 0
   rec={'experiment':'E3_v2','id':r['id'],'type':r['type'],'condition':label,'stratum':strata[r['id']], 'scope':'EXPLORATORY_299_ELIGIBLE_OF_300',
        **{k:v for k,v in c.items() if isinstance(v,(int,float,bool))},**{k:v for k,v in c['cost'].items() if isinstance(v,(int,float,bool)) or v is None},
        'support_label_f1':100*sf,'support_label_em':100*sem,'label_joint_f1':100*jf,'label_joint_em':em*sem,
        'new_support_facts_delivered':sum(c['new_support_facts_delivered_by_node']), 'provider_refusal_question':int(c['provider_refusal_count']>0),
        'noanswer':int(normalize(c['answer'])=='noanswer'),'explicit_empty_abstention':int(not c['answer'].strip())}
   flat.append(rec)
   if label in ['R_prefix','R_crop']:
    mode=label[2:];delivery=c['deliveries'][0];assert len(c['deliveries'])==1 and delivery['sender']==0 and delivery['recipient']==1 and delivery['text']==q[mode]['text'];assert t.count(delivery['text'])==delivery['tokens']==c['payload_tokens']==128 and c['cost']['logical_calls']==4
    raw=json.loads((D/'responses'/(response['call_key']+'.json')).read_text());assert raw['content_hash']==digest({k:v for k,v in raw.items() if k!='content_hash'})
    details.append({'id':r['id'],'type':r['type'],'stratum':strata[r['id']],'condition':label,'source_tokens':q[mode]['source_tokens'],'start_token':q[mode]['start_token'],'payload_tokens':128,'complete_facts':len(delivered_facts(delivery['text'])),'text':delivery['text'],'source_hash':q[mode]['source_hash'],'answer':c['answer'],'call_key':response['call_key'],'arm_order':json.dumps(q['arm_order']), 'question_wall_seconds':r['question_wall_seconds'],'question_client_cpu_seconds':r['question_client_cpu_seconds']})
  a,b=r['conditions']['R_prefix'],r['conditions']['R_crop'];pairs.append({'id':r['id'],'type':r['type'],'stratum':strata[r['id']], 'prefix_F1':a['f1'],'crop_F1':b['f1'],'prefix_minus_crop_F1':a['f1']-b['f1'],'same_answer_text':a['answer']==b['answer'],'same_normalized_answer':normalize(a['answer'])==normalize(b['answer']),'same_F1':a['f1']==b['f1'],'prefix_correct_crop_wrong':a['em']==100 and b['em']<100,'crop_correct_prefix_wrong':b['em']==100 and a['em']<100, 'prefix_complete_facts':len(a['deliveries'][0]['facts']),'crop_complete_facts':len(b['deliveries'][0]['facts'])})
  if a['f1']!=b['f1']:cases.append({'id':r['id'],'question':G[r['id']]['question'],'gold_answer':G[r['id']]['answer'],'prefix_answer':a['answer'],'crop_answer':b['answer'],'prefix_F1':a['f1'],'crop_F1':b['f1'],'prefix_payload':a['deliveries'][0]['text'],'crop_payload':b['deliveries'][0]['text']})
 assert len(flat)==1196 and len(pairs)==299
 V=pd.DataFrame(flat);V.to_csv(O/'tables/E3_v2_question_metrics.csv',index=False);pd.DataFrame(details).to_csv(O/'tables/E3_v2_payloads_answers_timings.csv',index=False);P=pd.DataFrame(pairs);P.to_csv(O/'tables/E3_v2_paired_transitions.csv',index=False)
 pd.DataFrame([{'id':r['id'],'eligible':r['eligible'],'reason':r.get('reason','ELIGIBLE'),'included_in_paired_estimand':r['eligible']} for r in rows]).to_csv(O/'tables/E3_v2_eligibility.csv',index=False)
 # Restore baseline version on each build, so integration is idempotent.
 for name in ['all_condition_question_metrics.csv','condition_summary.csv','all_scalar_distribution_metrics.csv','paired_question_transitions.csv']:
  d=pd.read_csv(ARCHIVE/'tables'/name);d.loc[d.experiment=='E3','experiment']='E3_original'
  if name=='all_condition_question_metrics.csv':d=pd.concat([d,V],ignore_index=True)
  elif name=='condition_summary.csv':
   s=[]
   for label,g in V.groupby('condition'):
    f,lo,hi=ci(g.f1);el,eh=wilson(int((g.em==100).sum()),len(g));v={'experiment':'E3_v2','condition':label,'n':len(g),'scope':g.scope.iloc[0],'F1':f,'F1_CI_low':lo,'F1_CI_high':hi,'EM':g.em.mean(),'EM_Wilson_low':el,'EM_Wilson_high':eh}
    for col in g.select_dtypes(include=['number','bool']).columns:
     if col in ['f1','em']:continue
     v[col+'_known_mean']=g[col].mean();v[col+'_missing_n']=int(g[col].isna().sum())
     if col in ['input_tokens','output_tokens']:v[col+'_all_question_mean']=g[col].mean() if g[col].notna().all() else None
    s.append(v)
   pd.DataFrame(s).to_csv(O/'tables/E3_v2_condition_summary.csv',index=False);d=pd.concat([d,pd.DataFrame(s)],ignore_index=True)
  elif name=='all_scalar_distribution_metrics.csv':
   s=[]
   for label,g in V.groupby('condition'):
    for col in g.select_dtypes(include=['number','bool']).columns:
     x=g[col].dropna().astype(float);v={'experiment':'E3_v2','condition':label,'metric':col,'n':len(g),'known_n':len(x),'missing_n':len(g)-len(x),'mean':x.mean(),'std':x.std(),'sum_known':x.sum(),'min':x.min(),'max':x.max()}
     v.update(dict(zip(['p05','p25','median','p75','p95'],x.quantile([.05,.25,.5,.75,.95]))));s.append(v)
   d=pd.concat([d,pd.DataFrame(s)],ignore_index=True)
  else:
   new=pd.DataFrame([{'experiment':'E3_v2','contrast':'R_prefix - R_crop','id':p['id'],'stratum':p['stratum'],'F1_change':p['prefix_minus_crop_F1'],'correct_to_wrong':int(p['crop_correct_prefix_wrong']),'wrong_to_correct':int(p['prefix_correct_crop_wrong']),'both_correct':int(p['prefix_F1']==p['crop_F1']==100),'both_wrong':int(p['prefix_F1']<100 and p['crop_F1']<100)} for p in pairs]);d=pd.concat([d,new],ignore_index=True)
  d.to_csv(O/'tables'/name,index=False)
 # Every nested numeric raw field, including receiver and inherited initial outputs.
 with (O/'tables/E3_v2_all_raw_numeric_fields.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['experiment','id','field','value'])
  for r in rows:
   for k,v in numeric(r):w.writerow(['E3_v2',r['id'],k,v])
 with (O/'tables/all_raw_numeric_fields.csv').open('w') as out:
  w=csv.writer(out)
  with (ARCHIVE/'tables/all_raw_numeric_fields.csv').open() as f:
   rd=csv.reader(f);w.writerow(next(rd))
   for row in rd:
    if row[0]=='E3':row[0]='E3_original'
    w.writerow(row)
  with (O/'tables/E3_v2_all_raw_numeric_fields.csv').open() as f:
   rd=csv.reader(f);next(rd);w.writerows(rd)
 effects=[]
 for metric,e in rep['paired_prefix_minus_crop'].items():effects.append({'experiment':'E3_v2','contrast':'R_prefix - R_crop','metric':metric,'scope':'EXPLORATORY_PRIMARY_F1_SECONDARY_DESCRIPTIVE',**e})
 pd.DataFrame([{**e,'ci95':json.dumps(e['ci95'])} for e in effects]).to_csv(O/'tables/E3_v2_paired_effects.csv',index=False)
 sub=[]
 for variable in ['type','stratum']:
  for value,g in P.groupby(variable):
   delta=np.asarray(g.prefix_minus_crop_F1);_,lo,hi=ci(delta)
   sub.append({'variable':variable,'group':value,'n':len(g),'prefix_F1':g.prefix_F1.mean(),'crop_F1':g.crop_F1.mean(),'prefix_minus_crop_F1':delta.mean(),'CI95_low':lo,'CI95_high':hi,'status':'POST_HOC_DESCRIPTIVE_NO_MULTIPLICITY_CONFIRMED_CLAIM'})
 pd.DataFrame(sub).to_csv(O/'tables/E3_v2_subgroups.csv',index=False)
 ledger=[json.loads(l) for l in (D/'calls.jsonl').read_text().splitlines()];starts={r['attempt_id']:r for r in ledger if r['event']=='started'};ends={r['attempt_id']:r for r in ledger if r['event'] in ['success','rejected','uncertain']};physical=[]
 for aid,r in starts.items():
  e=ends[aid];u=e.get('usage') or {};physical.append({'attempt_id':aid,'key':r['key'],'outcome':e['event'],'input_tokens_reported':u.get('prompt_tokens'),'output_tokens_reported':u.get('completion_tokens'),'provider_refusal':bool(e.get('provider_refusal')),'started_utc':r['utc'],'reported_end_utc':e['utc'],'source_ledgers':str(D/'calls.jsonl')})
 pd.DataFrame(physical).to_csv(O/'tables/E3_v2_physical_attempts.csv',index=False)
 phys=pd.read_csv(ARCHIVE/'tables/physical_attempts_deduplicated.csv');assert not set(phys.attempt_id)&set(starts);phys=pd.concat([phys,pd.DataFrame(physical)],ignore_index=True);phys.to_csv(O/'tables/physical_attempts_deduplicated.csv',index=False)
 successes=[r for r in physical if r['outcome']=='success'];usage={'pipeline':'E3_v2','successes':len(successes),'reported_input_tokens':sum(r['input_tokens_reported'] or 0 for r in successes),'successful_inputs_missing':sum(r['input_tokens_reported'] is None for r in successes),'reported_output_tokens_lower_bound':sum(r['output_tokens_reported'] or 0 for r in successes),'successful_outputs_missing':sum(r['output_tokens_reported'] is None for r in successes),'original_unresolved_attempts':sum(r['outcome']=='uncertain' for r in physical),'attempts':len(physical)}
 d=pd.read_csv(ARCHIVE/'tables/active_pipeline_physical_usage.csv');pd.concat([d,pd.DataFrame([usage])],ignore_index=True).to_csv(O/'tables/active_pipeline_physical_usage.csv',index=False)
 audits=json.loads((O/'audit_metrics.json').read_text());audits['physical_usage_global'].update(unique_attempts=len(phys),outcomes=phys.outcome.value_counts().to_dict(),reported_input_tokens_lower_bound=int(phys.input_tokens_reported.sum()),reported_output_tokens_lower_bound=int(phys.output_tokens_reported.sum()),attempts_missing_input_usage=int(phys.input_tokens_reported.isna().sum()),attempts_missing_output_usage=int(phys.output_tokens_reported.isna().sum()),refusals=int(phys.provider_refusal.sum()))
 audits['E3_v2']={'screened':300,'eligible':299,'result_sha256':status['sha256'],'status':'COMPLETE_EXPLORATORY','scope':'FOLLOWUP_AFTER_ORIGINAL_TEST_INSPECTION'};dump(O/'audit_metrics.json',audits)
 timings=[]
 for session in sorted((D/'runtime_sessions').iterdir()):
  end=session/'session_end.json'
  timings.append({'session':session.name,'status':json.loads(end.read_text())['status'] if end.exists() else 'END_NOT_RECORDED','wall_seconds':json.loads(end.read_text())['wall_seconds'] if end.exists() else None})
 pd.DataFrame(timings).to_csv(O/'tables/E3_v2_session_timings.csv',index=False)
 out={'report':rep,'physical_usage':usage,'paired_answers_identical_normalized':int(P.same_normalized_answer.sum()),'paired_answers_changed_normalized':int((~P.same_normalized_answer).sum()),'F1_ties':int(P.same_F1.sum()),'subgroups':sub,'completion_verification':{'scores':'PASS','payload_counts':'PASS','raw_cache_checksums':'PASS','source_hashes':'PASS'},'scope':'EXPLORATORY FOLLOWUP; primary paired F1; no superiority/equivalence conclusion','run_dir':str(D),'cases':sorted(cases,key=lambda r:abs(r['prefix_F1']-r['crop_F1']),reverse=True)[:12]};dump(O/'E3_v2_updated_analysis.json',out)
 for n in ['report.json','REPORT.md','completion_gate_verified.json','frozen_manifest.json','preflight.json']:shutil.copyfile(D/n,O/'original_reports'/('E3_v2_'+n))
 shutil.copyfile(D/'source_snapshot/e3_corrected.py',O/'sources/e3_corrected.py');shutil.copyfile(D/'source_snapshot/analysis/durable_runtime.py',O/'sources/E3_v2_durable_runtime.py')
 # Figure: distinguish historical failure from corrected outcome; do not pool protocols.
 fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');axs[0].bar(['Original distinct','Original identical','v2 distinct','v2 ineligible'],[2,298,299,1],color=['#777','#bbb','#2166ac','#bbb']);axs[0].tick_params(axis='x',rotation=20);axs[0].set(ylabel='Questions',title='E3 intervention checks, by version')
 e=rep['paired_prefix_minus_crop']['f1'];axs[1].errorbar([e['effect']],[0],xerr=[[e['effect']-e['ci95'][0]],[e['ci95'][1]-e['effect']]],fmt='o',capsize=5);axs[1].axvline(0,color='gray');axs[1].set(yticks=[0],yticklabels=['v2 prefix − crop'],xlabel='Paired answer F1 difference; 95% CI',title='299 eligible pairs; exploratory');plot(fig,'07_content_intervention')
 s=pd.read_csv(O/'tables/E3_v2_condition_summary.csv');fig,ax=plt.subplots(figsize=(9,4),layout='constrained');ax.errorbar(s.F1,range(4),xerr=[s.F1-s.F1_CI_low,s.F1_CI_high-s.F1],fmt='o',capsize=4);ax.set(yticks=range(4),yticklabels=s.condition,xlabel='Answer F1; marginal 95% bootstrap CI',title='E3-v2 receiver endpoints, same 299 eligible questions');plot(fig,'65_E3_v2_quality')
 fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.bar(['Prefix higher F1','Crop higher F1','Equal F1'],[8,18,273]);ax.set(ylabel='Paired questions',title='E3-v2 paired F1 transitions (n=299)');plot(fig,'66_E3_v2_transitions')
 fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.bar(['Prefix','Crop'],[641,500]);ax.set(ylabel='Complete parseable Evidence lines across 299 payloads',title='E3-v2: equal 128 tokens, different complete-fact counts');plot(fig,'67_E3_v2_complete_facts')
 # Existing per-metric plots now display current E3-v2 instead of the old intervention.
 full=pd.read_csv(O/'tables/all_condition_question_metrics.csv')
 for p in (O/'figures').glob('metric_*.png'):
  field=p.stem[len('metric_'):]
  if field not in full:continue
  fig,axs=plt.subplots(2,2,figsize=(13,10),layout='constrained')
  for ax,exp in zip(axs.flat,['E1','E2','E3_v2','E4']):
   g=full[full.experiment==exp].groupby('condition')[field].mean();ax.barh(g.index,g.values,color='#2166ac');ax.set(xlabel=field+' · mean over known values',title=exp+(' · 299 eligible pairs' if exp=='E3_v2' else ' · n=300'))
  plot(fig,p.stem)
 return out

def update_reports(reports,table,figure):
 z=json.loads((O/'E3_v2_updated_analysis.json').read_text());rep=z['report'];tab=lambda n:table(pd.read_csv(O/'tables'/n));version='''**E3 record update — 15 October 2026:** E3-v2 is the current content-intervention record. Original E3 is retained as `E3_original` / historical 256-token answer-and-citation slicing, with 298/300 identical pairs. E3-v2 is a distinct protocol: 300 screened IDs, 299 eligible distinct exact-128-token evidence-bank pairs, and one explicit no-document case. These are the same question identities, not 299 additional independent test questions. It is an exploratory follow-up after original held-out results were inspected. The pre-update bundle is preserved separately.\n\n'''
 for name,text in reports.items():
  text=text.replace('Scope: completed frozen WEBExtension E1–E5,','Scope: completed frozen WEBExtension E1–E5 plus corrected exploratory E3-v2,')
  text=text.replace('Each experiment uses the same 300 held-out question identities;','Original experiments use the same 300 held-out identities; E3-v2 screens all 300 and pairs 299 eligible evidence-bearing questions;')
  text=text.replace('ineffective E3 manipulation, quality loss','original ineffective E3 manipulation (now corrected in E3-v2, with an inconclusive contrast), quality loss')
  text=text.replace('E3 has two actual changes, E4','Original E3 has two actual changes; E3-v2 has 299 distinct equal-token pairs but its CI crosses zero. E4')
  text=text.replace('new E3 is degenerate. Change to an uncertain point estimate in the old protocol and an ineffective content intervention in E3.','original E3 was degenerate; E3-v2 prefix minus crop is −1.79539 F1, CI [−4.19733,0.470781], p=.140186. Report the corrected exploratory contrast without superiority or equivalence claims.')
  text=text.replace("Keep E3's ineffective manipulation visible instead of deleting its null result.","Keep original E3's ineffective manipulation as historical context and show corrected E3-v2's distinct intervention and inconclusive contrast.")
  text=text.replace("that does not repair E3's ineffective manipulation or small subgroups.","E3-v2 repairs the original identical-input manipulation, while its contrast remains inconclusive and small subgroups remain uncertain.")
  text=text.replace('## E3: content selection at exact token count','## Original E3 (historical): content selection at exact token count')
  text=text.replace('278/300','278/300')
  # Version notice directly after title, so old references cannot be mistaken for current evidence.
  title,rest=text.split('\n',1);reports[name]=title+'\n\n'+version+rest
 n1='01_COMPLETE_RESULTS_AND_METRICS.md';n2='02_DEEP_RAW_OUTPUT_FINDINGS.md';n3='03_PAPER_NARRATIVE_AND_REVISION_PLAN.md';n4='04_TECHNICAL_METHODS_AND_PSEUDOCODE.md'
 results='## Current E3-v2: complete results, resources and eligibility\n\n'+tab('E3_v2_condition_summary.csv')+'\n\n'+tab('E3_v2_paired_effects.csv')+'\n\nPrimary contrast is receiver F1, prefix minus crop; EM/support are descriptive secondary endpoints. Both arms update receiver 1 once and transmit exactly 128 tokens over edge 0→1. Four logical calls include three reused initial outputs, while only the receiver call is physically new. Baseline anchors are restricted to the same 299 eligible IDs. The no-document question is retained in `E3_v2_eligibility.csv`, without a fabricated treatment or outcome. All 299 pairs are distinct.\n\n'+table([z['physical_usage']])+'\n\nThe corrected run has 598 saved receiver responses from 601 attempts, with three explicit HTTP429 rejections and no uncertain attempts. Six refusals remain abstentions (three per treatment). All successful input usage is known; six output-usage values are unknown, so 166,539 reported output tokens is a lower bound. No currency bill or monetary savings is inferred. Global cost tables now include these attempts exactly once. Original initials, cached anchors and failed startup are not charged again by this analysis.\n\n'+tab('E3_v2_session_timings.csv')+'\n\n[Every E3-v2 nested numeric field](tables/E3_v2_all_raw_numeric_fields.csv) · [Payloads, answers and captured timings](tables/E3_v2_payloads_answers_timings.csv) · [Physical attempts](tables/E3_v2_physical_attempts.csv)\n\n'+figure('65_E3_v2_quality','E3-v2 receiver answer F1 with marginal uncertainty; paired inference is in the contrast table.')
 reports[n1]=reports[n1].replace('## E5 canonical whole-policy effects',results+'\n\n## E5 canonical whole-policy effects')
 deeper='''## Corrected E3-v2: intervention strength improved, quality advantage unresolved

Prefix receiver F1 is 31.23295 and crop F1 is 33.02834. Prefix minus crop is −1.79539 points, CI [−4.19733,0.470781], p=.140186. Prefix yields higher F1 on 8 questions, crop on 18, and 273 are tied. Equal F1 is not necessarily the same answer. '''+f"Normalized answer text changes on {z['paired_answers_changed_normalized']}/299 pairs.\n\n"+'''Prefix retains 641 complete parseable facts, versus 500 for crop, yet has the lower F1 point estimate. This does not establish that fewer facts cause better answers or that random cropping is optimal: these are two treatment means and the paired accuracy CI crosses zero. Both recipients already see the full shared evidence; neither arm introduces otherwise inaccessible facts. Structured lines cut by token boundaries do not count as complete delivered facts. The intervention tests redundant evidence presentation, not entailment or novel-evidence handoff.

The fixed receiver's saved NoComm F1 is 32.56618 and SelfUpdate F1 is 31.53239 on the same 299 questions. These provide context, not a new trained policy or a multiplicity-confirmed superiority claim. Six treatment refusals are counted in headline results. The one zero-document question is retained transparently but has no paired content treatment; this eligibility decision was frozen before E3-v2 generation.

'''+tab('E3_v2_subgroups.csv')+'\n\nSubgroups are post-hoc descriptive; the gold-location classification describes the same question’s original P assignment, while E3-v2 itself uses R. They are not P-regime treatment results. Do not select favorable subgroups to replace the pooled endpoint.\n\n'+figure('66_E3_v2_transitions','Paired F1 changes: 8 favor prefix, 18 favor crop, 273 ties.')+figure('67_E3_v2_complete_facts','Complete structured evidence facts survive differently despite equal token counts.')+'\n### Actual changed-answer cases\n\n'
 for c in z['cases'][:6]:deeper+=f"- `{c['id']}` — Question: {c['question']} Gold: **{c['gold_answer']}**; prefix: **{c['prefix_answer']}** (F1 {c['prefix_F1']:.4g}); crop: **{c['crop_answer']}** (F1 {c['crop_F1']:.4g}).\n"
 deeper+='\nCases are selected descriptively by absolute F1 change, not used to select questions or estimate the treatment effect. Exact payloads are in the casebook and full payload table.\n'
 reports[n2]=reports[n2].replace('## E2 and E3 expose',deeper+'\n\n## E2 and historical E3 expose')
 narrative='''## Updated E3 interpretation and paper-ready results text

E3-v2 removes the original identical-input failure but supplies no statistically decisive quality advantage. The paper can now report a functioning, resource-matched evidence-presentation intervention with an inconclusive exploratory effect. It still cannot claim general content-insensitivity, prefix superiority, equivalence, or that a semantic selector has been validated. The source-bank definition, cap, eligibility and follow-up status must be stated.

**Replacement E3 results paragraph:** “The original content intervention produced identical prefix and crop messages on 298 of 300 questions because the 256-token cap exceeded most sender messages. We therefore conducted an exploratory follow-up using all sender-visible Wikipedia sentences as the message source, while reusing frozen initial agent states. Among 300 screened questions, 299 supplied evidence and supported distinct 128-token prefix and seeded contiguous-crop payloads; the remaining question had no documents and was retained as an explicit ineligible case. Each arm updated the same receiver once. Mean receiver answer F1 was 31.23 for prefix and 33.03 for crop. The paired prefix-minus-crop difference was −1.80 points (95% bootstrap CI −4.20 to 0.47; randomization p=.140). This interval does not establish superiority or equivalence. Because the receiver already possessed the shared documents and crops could cut structured evidence lines, the comparison measures redundant evidence presentation within this protocol.”

**Abstract/discussion change:** Replace an assertion that E3 remains uncorrected with: “A corrected exploratory content intervention produced distinct messages at identical token counts but an inconclusive F1 contrast, underscoring the difference between intervention fidelity and evidence of quality preservation.” Retain the original failure as a disclosed motivation and version history, not as the current paired result.

**Core-narrative limitation remains:** E3-v2 is not a topology/adaptivity experiment. E5's edge/quota-matched random control differs in recipient calls and actual transmitted tokens. A prospective resource-matched dynamic in-star versus validation-selected static in-star, random in-star and action-shuffled control remains proposed—not executed. With a fixed single recipient and three nodes, the two-edge in-star is unique; matching recipient count and fixing recipient identity answer different questions. Use a fresh uninspected held-out set for confirmation.

'''
 reports[n3]=reports[n3].replace('## Claim-by-claim changes',narrative+'\n## Claim-by-claim changes')
 reports[n3]=reports[n3].replace('Only two of 300 content pairs differ.','Original E3 changed only two content pairs; the exploratory E3-v2 follow-up changed all 299 eligible pairs at 128 tokens, with an inconclusive F1 contrast.')
 methods='''## Current E3-v2: executed protocol and pseudocode

Run: `e3_corrected_b5c4ff9db9b9b2bc`. Model, roles, initial answers, BM25 pool and receiver evidence are reused from the original frozen study. Sender 0 and receiver 1 remain fixed in R. The source bank contains every sender-visible Wikipedia sentence in retrieval rank and original sentence order; it does not depend on gold answers or successful generated citations. Cap 128 and all slices/eligibility/order are frozen before new inference. Seeded non-prefix candidates must be valid exact-token substrings and differ from the prefix; no padding is used. The source-empty ID `5a8cfee555429941ae14df5c` receives no fabricated message.

```text
verify original run completion and all scientific input hashes
load the same 300 gold-free inputs and saved R initial states
for each question:
    bank = all sentences actually visible to sender 0 in frozen order
    if no bank:
        freeze NO_SENDER_EVIDENCE_AVAILABLE; retain screened record
    else:
        prefix = valid first 128 exact tokenizer tokens
        crop = seeded nonidentical valid contiguous 128-token slice
        assert exact_count(prefix) = exact_count(crop) = 128
        freeze bank hash, slices, offsets and randomized arm order
for each unfinished screened question:
    for arm in frozen order, if eligible:
        reuse valid saved exact-request response if available
        otherwise persist request and dispatch real Azure completion
        persist raw HTTP body before parsing; count refusal as abstention
        update only receiver 1 with same own evidence and previous answer
    attach saved receiver NoComm and SelfUpdate anchors
    score after generation; fsync record and timings
verify 300 unique screened IDs, 299 distinct equal-token pairs and checksums
paired primary = mean(receiver_F1_prefix - receiver_F1_crop), n=299
bootstrap/sign-randomization draws=10000, seed=23
report all results and uncertainty; secondary EM/support are descriptive
```

Operational recovery resumed only unfinished questions after three explicit 429 rejections. Successful cached responses were reused; there are no unresolved attempts and no uncertain-cost replacements in this corrected run. A no-inference startup failure caused by a lazy-module import was corrected operationally; its evidence remains saved. Science sources and payloads are bound by the final run's manifest. Reports use saved outputs only, and token/frame counts do not imply exactly matched generated-token costs or latency.

Reproduce this report update offline with `HF_HUB_OFFLINE=1 python WEBExtension/analysis/final_findings_20261014/write_reports.py`, followed by `verify_bundle.py`. The generator calls `integrate_e3_v2.prepare_tables()` and `update_reports()`; neither creates a provider client or invokes inference. Frozen generation code is copied to `sources/e3_corrected.py` for audit, not execution by this analysis.

'''
 reports[n4]=reports[n4].replace('## Original E3 (historical):',methods+'\n## Original E3 (historical):')
 return reports
