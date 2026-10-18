"""Plot assumed question-level power sensitivity; no E9 outcome data."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/e9-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'analysis/power_sensitivity.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(12,3.9),sharey=True)
for ax,difference in zip(axes,(2,5,10)):
    for sd,style in zip((.25,.5,1.,2.),('-','--','-.',':')):
        group=[r for r in rows if r['difference_f1_points']==difference and r['contrast_sd_unit_f1']==sd]
        ax.plot([r['n_questions'] for r in group],[r['approx_power_two_sided_alpha05'] for r in group],style,marker='o',label=f'Contrast SD {sd:g}')
    ax.axhline(.8,color='gray',linewidth=.7)
    ax.set_title(f'Assumed interaction: {difference} F1 points',fontsize=10)
    ax.set_xlabel('Independent questions (count)')
    ax.set_ylim(0,1);ax.set_xticks([200,300,600]);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Approximate two-sided test power (probability)')
axes[-1].legend(frameon=False,fontsize=8,loc='lower right')
fig.suptitle('E9 planning sensitivity: assumed variance, no pilot outcomes',fontsize=12)
fig.text(.5,.005,'Source: analysis/power_sensitivity.json; normal approximation, alpha=0.05. SD is of paired question interaction in unit F1.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.05,1,.94))
out=ROOT/'analysis/figures';out.mkdir(exist_ok=True)
fig.savefig(out/'interaction_power_sensitivity.png',dpi=170)
fig.savefig(out/'interaction_power_sensitivity.svg')
