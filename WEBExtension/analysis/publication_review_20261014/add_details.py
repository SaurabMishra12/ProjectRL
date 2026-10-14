import json,sys,math,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;ROOT=Path('/home/saurab/Documents/projectRLMAIN/WEBExtension');sys.path.insert(0,str(ROOT))
from protocol import file_hash
from analysis.gold_strata import contrasts
A=json.loads((ROOT/'artifacts/active_machine_runs.json').read_text());MAIN=Path(A['controlled_run']);CTRL=Path(A['controller_study'])
audit=json.loads((OUT/'audit_metrics.json').read_text());F=pd.read_csv(OUT/'tables/all_condition_question_metrics.csv');S=pd.read_csv(OUT/'tables/condition_summary.csv');M=pd.read_csv(OUT/'tables/E5_partial_seed_averages.csv');L=pd.read_csv(OUT/'tables/controller_learning_curves.csv')
rr={e:[json.loads(l) for l in ((CTRL if e=='E5' else MAIN)/(e+'.jsonl')).read_text().splitlines()] for e in ['E1','E2','E3','E4','E5']}
e1={r['id']:r for r in rr['E1']}
checks=[]
for r in rr['E2']:
 for lab in ['R_CW','R_CCW']:
  x=e1[r['id']]['conditions'][lab];y=r['conditions'][lab]
  checks.append({'id':r['id'],'condition':lab,'answers_identical':x['answer']==y['answer'],'F1_identical':x['f1']==y['f1'],'EM_identical':x['em']==y['em'],'call_keys_identical':x['cost']['call_keys']==y['cost']['call_keys'],'source_payloads_identical':[(d['sender'],d['recipient'],d['text'],d['tokens']) for d in x['deliveries']]==[(d['sender'],d['recipient'],d['text'],d['tokens']) for d in y['deliveries']]})
C=pd.DataFrame(checks);C.to_csv(OUT/'tables/E2_E1_shared_outcome_audit.csv',index=False)
audit['manipulations']['E2_E1_scientific_outcomes_and_provider_keys_identical']=bool(C.iloc[:,2:].all().all())
audit['manipulations']['E2_full_record_difference_note']='Full records differ in cache/timing metadata; scientific answers, F1/EM, payloads and provider call keys are identical.'
# Fair prefix-level seed mean versus baselines, including tradeoffs instead of just wins.
trade=[]
for regime in ['R','P']:
 learned=M[M.condition==regime+'_Learned_seed_mean']
 for base in ['NoComm','SelfUpdate','Static','Broadcast']:
  d=F[(F.experiment=='E5')&(F.condition==regime+'_'+base)]
  trade.append({'regime':regime,'baseline':base,'n':len(d),'learned_seed_mean_F1':learned.f1.mean(),'baseline_F1':d.f1.mean(),'F1_change':learned.f1.mean()-d.f1.mean(),'learned_mean_payload_tokens':learned.payload_tokens.mean(),'baseline_payload_tokens':d.payload_tokens.mean(),'payload_reduction_percent':100*(1-learned.payload_tokens.mean()/d.payload_tokens.mean()) if d.payload_tokens.mean()>0 else None,'learned_mean_logical_calls':learned.logical_calls.mean(),'baseline_calls':d.logical_calls.mean(),'call_reduction_percent':100*(1-learned.logical_calls.mean()/d.logical_calls.mean()),'scope':'PARTIAL_PREFIX_ONLY'})
pd.DataFrame(trade).to_csv(OUT/'tables/E5_partial_quality_resource_tradeoffs.csv',index=False);audit['E5_partial_tradeoffs']=trade
# Within-regime source-call reuse is intentional, not new independent evidence.
audit['metric_definitions']={
 'f1':'Official HotpotQA normalized answer token F1, multiplied by100; mean over questions.',
 'em':'Official normalized answer exact match, multiplied by100.',
 'support_available_recall':'Fraction of gold supporting (title,sentence index) labels present in selected candidate pool.',
 'all_support_available':'All gold supporting labels available somewhere in selected pool; not an entailment check.',
 'split_support_across_nodes':'Gold supporting labels available in pool but not all in any one P shard.',
 'answer_support_recall_precision':'Gold supporting-label agreement for citations of responses agreeing with ensemble answer.',
 'support_label_f1_em':'Additional official-style support-set overlap diagnostics on that citation union, NOT evaluated sentence entailment.',
 'label_joint_f1_em':'Official joint-metric algebra applied to ensemble answer/citation-union diagnostics; not an official leaderboard evaluation.',
 'new_support_facts_delivered':'Gold support labels newly delivered to recipients according to serialized message facts, not guaranteed comprehension.',
 'payload_tokens':'Actual pinned-tokenizer count summed over deliveries; source broadcast message counted once per edge.',
 'logical_calls':'Calls required by this arm including shared initial inference; not a count of new physical charges.',
 'cache_reads_this_execution':'Execution/recovery metadata; do not interpret as a treatment property.',
 'input_output_tokens':'Reported provider dimensions; missing counts remain NA; known-subset means differ from all-question means.',
 'provider_refusal_count':'Explicit refused terminal responses among three current agents; provider_refusal_calls also includes required initial/receiver logical calls.',
 'utility_gain':'F1gain/100 minus .05 payload/fullbroadcast minus .02 distinct updated recipients/3.',
 'ci95':'Question bootstrap intervals; completed stored contrasts use10000 draws, added summaries/partial E5 use5000 deterministic draws.',
 'pvalues':'Preserved paired sign-randomization values; E1 primary interaction is unique primary endpoint. E5 partial values are exploratory, not final or selection targets.',
 'overlap':'Across train/validation/test/pilot, exact ID/normalized-question/candidate-title intersections; across E1–E4 and E5, shared held-out questions are intentionally paired.'}
# Additional global multiplicity sensitivity, distinct from original prespecified families.
E=audit['complete_effects'];secondary=[e for e in E if not e.get('primary')];ordered=sorted(secondary,key=lambda e:e['randomization_p']);prev=0
for i,e in enumerate(ordered):prev=max(prev,min(1,(len(ordered)-i)*e['randomization_p']));e['holm_across_completed_secondary_contrasts']=prev
ordered=sorted(audit['partial_E5_effects'],key=lambda e:e['randomization_p']);prev=0
for i,e in enumerate(ordered):prev=max(prev,min(1,(len(ordered)-i)*e['randomization_p']));e['holm_exploratory_partial_family']=prev
# Compare suites on only identical completed question IDs, exposing namespace/provider randomness.
cross=[]
for r in rr['E5']:
 for label in ['R_NoComm','P_NoComm','R_SelfUpdate','P_SelfUpdate']:
  c=r['conditions'][label];b=e1[r['id']]['conditions'][label];cross.append({'id':r['id'],'condition':label,'answers_identical':c['answer']==b['answer'],'F1_change_E5_minus_E1':c['f1']-b['f1'],'same_initial_call_keys':c['cost']['call_keys'][:3]==b['cost']['call_keys'][:3]})
X=pd.DataFrame(cross);X.to_csv(OUT/'tables/cross_suite_same_question_baselines.csv',index=False)
audit['cross_suite_baseline_overlap']={'comparisons':len(X),'answer_disagreements':int((~X.answers_identical).sum()),'same_initial_provider_keys':int(X.same_initial_call_keys.sum()),'interpretation':'E5 uses controller namespace and separate responses; compare treatments within the same suite, do not pool as independent questions.'}
# Remove potentially overlapping point annotations: exact points with a separate legend.
BLUE='#2166ac';ORANGE='#b85c00';GRAY='#777777'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained');d=S[S.experiment=='E1'];markers={'NoComm':'o','SelfUpdate':'s','CW':'^','CCW':'D'}
for _,v in d.iterrows():
 regime,method=v.condition.split('_',1);color=BLUE if regime=='R' else ORANGE
 for ax,col in zip(axes,['payload_tokens_known_mean','logical_calls_known_mean']):ax.scatter(v[col],v.F1,s=80,marker=markers[method],color=color,label=v.condition,alpha=.8)
for ax in axes:ax.set_ylabel('Mean answer F1 (0–100)');ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8)
axes[0].set(xlabel='Mean actual payload tokens/question',title='E1 transmitted payload vs quality');axes[1].set(xlabel='Logical calls/question including shared initials',title='E1 calls vs quality')
for ext in ['png','pdf']:fig.savefig(OUT/'figures'/('06_resource_frontiers.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
# Plot every scalar metric available in the condition table, with clear units/scopes.
fields=[c for c in F.select_dtypes(include=['number','bool']).columns if c not in ['f1','em']]
for field in fields:
 fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
 for ax,e in zip(axes.flat,['E1','E2','E3','E4']):
  g=F[F.experiment==e].groupby('condition')[field].mean();ax.barh(g.index,g.values,color=BLUE);ax.set(xlabel=field+' · mean over observed/known values',title=e+' · n=300; missing values excluded')
 for ext in ['png','pdf']:fig.savefig(OUT/'figures'/('metric_'+field+'.'+ext),dpi=160,bbox_inches='tight')
 plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
for seed,g in L.groupby('seed'):
 for ax,col in zip(axes,['optimizer_updates','optimizer_loss','optimizer_max_kl']):ax.plot(g.epoch,g[col],marker='o',label=f'Seed {seed}');ax.set(xlabel='Collection epoch',ylabel=col,title=col,xticks=[1,2,3])
for ax in axes:ax.legend()
for ext in ['png','pdf']:fig.savefig(OUT/'figures'/('14_optimizer_diagnostics.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
(OUT/'audit_metrics.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print('Detailed figures',len(list((OUT/'figures').glob('*.png'))),'E2 underlying outcomes identical',audit['manipulations']['E2_E1_scientific_outcomes_and_provider_keys_identical'])
