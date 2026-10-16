"""Render the audited offline readouts as a research report and scientific figure."""
import json
import os
from pathlib import Path
from .analyse import ROOT, RULES, prior


def main():
    out=ROOT/'results';s=json.loads((out/'summary.json').read_text());groups=s['summaries']
    os.environ['MPLCONFIGDIR']=str((out/'.mplcache').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    panels=[('E5: partitioned evidence', [('E5','P_NoComm','No communication'),('E5','P_Broadcast','Broadcast'),
              ('E5','P_Learned_seed31','Saved In-Star, cap 128'),('E5_projection','P_InStar256_reused','In-Star projection, cap 256*')]),
            ('E1: partitioned evidence', [('E1','P_NoComm','No communication'),('E1','P_CW','Clockwise ring'),
              ('E1','P_CCW','Counterclockwise ring'),('E1','P_SelfUpdate','Private revision')])]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    colors=['#9aa7b3','#237c72','#243b53'];labels=['Original vote','Abstention-aware vote','Hub 1 only']
    plot_data=[]
    for ax,(title,arms) in zip(axes,panels):
        y=np.arange(len(arms))
        for j,rule in enumerate(RULES):
            values=[groups[exp][arm][rule]['correct'] for exp,arm,_ in arms]
            pos=y+(j-1)*.23;ax.barh(pos,values,height=.21,color=colors[j],label=labels[j])
            for i,value in enumerate(values):ax.text(value+.6,pos[i],str(value),va='center',fontsize=8)
            plot_data.extend({'experiment':exp,'arm':arm,'readout':rule,'correct':value,'denominator':300}
                             for (exp,arm,_),value in zip(arms,values))
        ax.set_yticks(y,[a[2] for a in arms]);ax.invert_yaxis()
        ax.set(title=title,xlabel='Correct answers among 300 questions',xlim=(0,77))
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    handles,legends=axes[0].get_legend_handles_labels();fig.legend(handles,legends,ncol=3,loc='lower center',bbox_to_anchor=(.5,.07),frameon=False)
    fig.suptitle('Offline readout audit · 300 shared question identities · zero new model calls',fontsize=12)
    fig.text(.01,.015,'* Projection reuses Broadcast Hub-1 output after exact request reconstruction; it is not an independent generated treatment.',fontsize=8)
    fig.tight_layout(rect=(0,.17,1,.93))
    for extension in ('png','svg'):
        target=out/f'readout_results.{extension}';fig.savefig(target,dpi=180,bbox_inches='tight')
        if extension=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig);prior.save(out/'figure_data.json',plot_data)
    em=s['contrasts']['E5:P_Learned_seed31-P_Broadcast:sink1']['em']
    aa=s['contrasts']['E5:P_Broadcast-P_NoComm:abstention_aware']['em']
    pc=s['projection_costs']['P']
    lines=['# Offline E5/E1 Sink and Abstention-Aware Readout Audit','',
      '**Decision:** keep the one-sender 2Wiki run stopped. Existing E5 supports exact one-round Hub-1 request preservation under a matched-cap In-Star projection, and a strong abstention-aware baseline. It does not establish learned SparseComm superiority or statistical equivalence of the original saved policies. No N=100 2Wiki run was launched.','',
      '## What was actually run','',
      'Recomputed the existing pure readout functions on all 300 saved questions in E5 and all 300 in E1. These are exactly the same question IDs, not 600 independent questions. Both shared (R) and partitioned (P) regimes, every saved arm, and original voting, abstention-aware voting, and fixed Hub 1 readout were included. Zero Azure calls, credential reads, downloads, training steps, or ledger writes were used. Original artifacts and earlier reports were preserved.','',
      'Abstention-aware voting excludes normalized `noanswer` and explicitly marked empty provider refusals, then uses the original vote and agent-order tie rule. All-abstaining pools return `noanswer`. Hub 1 is the fixed zero-based node 1; it is never selected using the gold answer. Gold labels are used only for offline evaluation.','',
      '![Observed readout results](results/readout_results.png)','',
      '## E5: matched Hub-1 projection versus saved In-Star','',
      'The frozen runner updates each recipient synchronously from its own initial state and incoming initial messages. For Hub 1, Broadcast sends exactly the ordered messages from nodes 0 and 2. Deleting the four edges that do not enter Hub 1 leaves its prompt, visible-source set, model parameters, serving identity, and tokenizer identity unchanged when the message cap is held fixed. The script executes only the inspected pure prompt/update definitions with an offline capture object, reconstructs the request, and verifies its full cache key against the saved Broadcast request.','',
      '**All 300 partitioned and all 300 shared Hub requests matched exactly.** Reusing these saved outputs under the cap-256 In-Star projection gives the same Hub outcomes as Broadcast: partitioned **37/300 correct, F1 17.3639**; shared **69/300 correct, F1 31.7159**. This is a structural one-round result, not 600 independent outcomes or a separately generated treatment.','',
      'The actually saved partitioned seed-31 policy used the 2-edge In-Star on all 300 questions, but with cap 128. Its Hub request matches Broadcast on 291 questions; nine requests differ. Answers agree on 295 questions. Its Hub scores are **36/300 correct, F1 16.6715**, compared with Broadcast 37/300 and 17.3639. It has zero EM wins, one loss, and 299 ties. The difference is '+f'{em["delta_pp"]:+.3f} percentage points, bootstrap 95% interval [{em["ci95_pp"][0]:.3f}, {em["ci95_pp"][1]:.3f}], exact McNemar p={em["exact_mcnemar_p"]:.3g}.'+' A nonsignificant difference is not an equivalence or noninferiority test; no margin was prespecified.','',
      'Thus the defensible claim is **same-cap, one-round, fixed-sink request sufficiency**. It is not a claim that all saved In-Star policies equal Broadcast, that sparse majority equals dense majority, or that the result extends to further rounds.','',
      '## E5: readout repair and the stronger baseline','',
      '- NoComm: original vote **10/300**, abstention-aware **62/300**; F1 5.0598 to 28.8823.',
      '- Broadcast: original vote **34/300**, abstention-aware **65/300**; F1 14.9920 to 30.1132.',
      '- Saved cap-128 In-Star (seed 31): original vote **17/300**, abstention-aware **63/300**; F1 7.5241 to 29.0562.',
      '- Fixed Hub-only readout scores lower than abstention-aware voting here: Broadcast 37 versus 65; saved In-Star 36 versus 63. These readouts answer different operational questions and must not be conflated.','',
      'With abstention-aware readout matched on both sides, Broadcast gains only '+f'{aa["delta_pp"]:.3f} EM percentage point over NoComm: 95% interval [{aa["ci95_pp"][0]:.3f}, {aa["ci95_pp"][1]:.3f}], exact McNemar p={aa["exact_mcnemar_p"]:.3f}.'+' This is exploratory; the old 34-versus-62 comparison mixes readout rules. The projected cap-256 In-Star gives 63/300 under abstention-aware voting versus Broadcast 65/300 because its non-hub agents retain their initial answers.','',
      '## E1 cross-check','',
      'E1 has clockwise and counterclockwise 3-edge rings, not saved full Broadcast or 2-edge In-Star arms. Its separate initial-state namespace was retained. Partitioned NoComm changes from **11/300** under original voting to **62/300** under abstention-aware voting. Clockwise changes **20 to 64**; counterclockwise **19 to 62**; private revision **11 to 60**. Hub-only counts are 34, 34, 34, and 31, respectively. This corroborates the readout bottleneck; it does not independently test In-Star versus Broadcast.','',
      '## Communication and inference costs','',
      '- Across the 300 partitioned projections, directed deliveries fall from **1,800 to 600**, exactly **66.7% fewer**.',
      '- Recipient update calls fall from **900 to 300**, also **66.7% fewer**. Including the common 900 initial calls, logical inference calls fall from **1,800 to 1,200**, a **33.3%** reduction.',
      f'- Observed delivered payload tokens fall from **{pc["dense_payload_tokens"]:,} to {pc["instar_payload_tokens"]:,}**, a **{100*(1-pc["instar_payload_tokens"]/pc["dense_payload_tokens"]):.2f}%** reduction. Edges and payload tokens are different cost units.',
      '- Model usage references have missing fields (nine dense and six projected In-Star input/output fields in P). Known totals are lower bounds; no exact all-in token or monetary saving is claimed. Existing cached-reference totals are not new billing.','',
      '## Complete readout results','',
      'Every cell has 300 questions. Format: correct count / mean answer F1 (0–100). The projection rows reuse saved Broadcast receiver outputs.','',
      '| Experiment | Arm | Original vote | Abstention-aware vote | Hub 1 |',
      '|---|---|---:|---:|---:|']
    for exp,arms in groups.items():
        for arm,values in arms.items():
            cells=[f'{values[r]["correct"]} / {values[r]["f1"]:.4f}' for r in RULES]
            lines.append('| '+' | '.join([exp,arm,*cells])+' |')
    lines+=['','## Verification and reproduction','',
      'The offline run verified both completion-status hashes, exact 300-ID pairing, all four routing/backend source hashes against the frozen E5 manifest, 27,600 response references against raw cached content, and 35,400 answer scores against the preserved official HotpotQA evaluator. All 13,573 consumed source files, cached responses, and protected ledgers retained their hashes after analysis. Graph coverage was unavailable for these new/ignored paths; direct source and artifact checks supplied the evidence.','',
      'Paired intervals resample whole questions 10,000 times. All comparisons are post hoc and exploratory; the 300 IDs overlap earlier analyses. Serving weights are a deployment alias, not an authenticated immutable provider snapshot. The structural request identity establishes a valid cache reuse under the frozen pipeline, not equality of separately sampled future generations. No future two-round result is implied.','',
      '```bash',
      'python -m WEBExtension.pilots.hotpot_readout_20261016.analyse',
      'python -m WEBExtension.pilots.hotpot_readout_20261016.report',
      '```','',
      '[Summary and paired intervals](results/summary.json) · [All readouts, compressed JSONL](results/question_readouts.jsonl.gz) · [Question-level Hub request audit](results/hub_request_audit.jsonl) · [Input hashes, compressed JSON](results/provenance.json.gz) · [Exact figure data](results/figure_data.json)','',
      '**Next-step decision:** this audit supplies the requested zero-inference Web diagnostic and a narrow structural cost result. It does not justify spending on the unchanged one-sender 2Wiki design. A complementary k∈{0,1,2} router and a frozen N=100 comparison would be a separate, later experiment; neither has been run here.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Saved REPORT.md and scientific plots.')


if __name__=='__main__':main()
