import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/home/saurab/Documents/projectRLMAIN/WEBExtension/analysis/final_findings_20261014')
a=json.loads((O/'audit_metrics.json').read_text());effects=[e for e in a['complete_effects'] if e.get('metric','f1')=='f1']
fig,ax=plt.subplots(figsize=(12,max(5,.42*len(effects))),layout='constrained')
for i,e in enumerate(effects):
 color='#b85c00' if e['experiment']=='E3_v2' else '#2166ac'
 ax.errorbar(e['effect'],i,xerr=[[e['effect']-e['ci95'][0]],[e['ci95'][1]-e['effect']]],fmt='o',capsize=3,color=color)
ax.axvline(0,color='gray');ax.set(yticks=range(len(effects)),yticklabels=[e['experiment']+' · '+e['contrast'] for e in effects],xlabel='Paired answer F1 change (points); 95% CI',title='Original frozen contrasts and corrected exploratory E3-v2; no protocol pooling')
for ext in ['png','pdf']:fig.savefig(O/'figures'/('02_paired_effects.'+ext),dpi=160,bbox_inches='tight')
plt.close(fig)
t=pd.read_csv(O/'tables/paired_question_transitions.csv');groups=t.groupby(['experiment','contrast']);labels=[];helped=[];harmed=[]
for (exp,contrast),g in groups:labels.append(exp+' · '+contrast+' · n='+str(len(g)));helped.append(g.wrong_to_correct.sum());harmed.append(g.correct_to_wrong.sum())
fig,ax=plt.subplots(figsize=(12,max(5,.55*len(labels))),layout='constrained');y=np.arange(len(labels));ax.barh(y-.17,helped,height=.32,label='Wrong → correct',color='#2166ac');ax.barh(y+.17,harmed,height=.32,label='Correct → wrong',color='#b85c00');ax.set(yticks=y,yticklabels=labels,xlabel='Question count (answer EM transitions)',title='Paired transitions; original E3 retained as historical');ax.legend(loc='lower right')
for ext in ['png','pdf']:fig.savefig(O/'figures'/('05_paired_transitions.'+ext),dpi=160,bbox_inches='tight')
plt.close(fig)
