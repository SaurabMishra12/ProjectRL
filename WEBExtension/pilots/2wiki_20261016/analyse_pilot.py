"""Offline analysis. Strict incomplete-cell bounds and operational failures stay distinct."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from WEBExtension.web_reasoning.cli import main as cli
from WEBExtension.web_reasoning.statistics import paired_summary
from WEBExtension.web_reasoning.storage import atomic_json
from WEBExtension.web_reasoning.preparation import verify_run,file_sha

ROOT=Path('WEBExtension/web_reasoning/artifacts/2wiki_pilot_20261016')
RUN=ROOT/'representative_run'
NAMES={'nocomm':'No communication','private':'Private revision','static':'Static sparse','random':'Random sparse',
       'selected':'SparseComm (heuristic)','instar':'Complete In-Star','dense':'Dense broadcast','full_union':'Single agent, full union'}


def summary(records,name):
    rows=[r['arms'][name] for r in records];scores=[r['sink'] for r in rows]
    known=[s['em'] for s in scores if s.get('em') is not None];f1=[s['f1'] for s in scores if s.get('f1') is not None]
    n=len(rows);missing=n-len(known)
    return {'name':NAMES[name],'n':n,'correct':sum(s.get('em')==100 for s in scores),
            'operational_em':sum(known)/n,'operational_f1':sum(f1)/n,
            'official_em_identification_bounds':[sum(known)/n,(sum(known)+100*missing)/n],
            'scorable_endpoints':len(known),'incomplete_endpoints':missing,
            'sink_outcomes':dict(Counter(r['sink_outcome'] for r in rows)),
            'logical_calls':sum(r['cost']['logical_calls'] for r in rows),
            'mean_logical_calls':sum(r['cost']['logical_calls'] for r in rows)/n,
            'input_token_reference_sum':sum(r['cost']['logical_input_token_reference_sum'] for r in rows),
            'output_token_reference_sum':sum(r['cost']['logical_output_token_reference_sum'] for r in rows),
            'usage_missing':sum(r['cost']['input_usage_missing']+r['cost']['output_usage_missing'] for r in rows),
            'payload_tokens':sum(r['payload_tokens'] for r in rows),'control_tokens':sum(r['control_tokens'] for r in rows),
            'data_deliveries':sum(r['data_deliveries'] for r in rows),'control_frames':sum(r['control_frames'] for r in rows),
            'complete_delivered_atoms':sum(r['complete_delivered_atoms'] for r in rows),'omitted_atoms':sum(r['omitted_atoms'] for r in rows),
            'support_visible_at_sink':sum(r.get('annotated_support_visible_to_sink') is True for r in rows),
            'readout_diagnostics':{rule:{'operational_em':sum((r[rule].get('em') or 0) for r in rows)/n,
                                       'incomplete':sum(r[rule].get('em') is None for r in rows)} for rule in ('abstention_vote','original_vote_diagnostic')}}


def figures(stats,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=list(NAMES);y=np.arange(len(keys));labels=[NAMES[k] for k in keys]
    fig,axes=plt.subplots(1,3,figsize=(15,5.1),gridspec_kw={'width_ratios':[1.15,1,1]},sharey=True)
    colors=['#237c72' if k=='selected' else '#243b53' if k=='instar' else '#9aa7b3' for k in keys]
    em=[stats[k]['operational_em'] for k in keys];unknown=[stats[k]['incomplete_endpoints']*100/stats[k]['n'] for k in keys]
    axes[0].barh(y,em,color=colors,height=.63,label='Operational correct')
    axes[0].barh(y,unknown,left=em,color='#eeeeee',edgecolor='#999999',hatch='///',height=.63,label='Unresolved score width')
    for i,value in enumerate(em):axes[0].text(value+.8,i,f'{value:.0f}%',va='center',fontsize=9)
    axes[0].set(xlim=(0,108),xlabel='Answer EM (% of all 20 questions)',title='Accuracy and incomplete-score bounds')
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis()
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.15),fontsize=8,ncol=1)
    inputs=[stats[k]['input_token_reference_sum']/stats[k].get('cost_question_records',stats[k]['n'])/1000 for k in keys];outputs=[stats[k]['output_token_reference_sum']/stats[k].get('cost_question_records',stats[k]['n'])/1000 for k in keys]
    axes[1].barh(y,inputs,color='#94a3b8',height=.63,label='Input');axes[1].barh(y,outputs,left=inputs,color='#243b53',height=.63,label='Output')
    axes[1].set(xlabel='Tokens per question (thousands)',title='Standalone logical token references');axes[1].legend(fontsize=8)
    data=[stats[k]['payload_tokens']/stats[k].get('cost_question_records',stats[k]['n']) for k in keys];control=[stats[k]['control_tokens']/stats[k].get('cost_question_records',stats[k]['n']) for k in keys]
    axes[2].barh(y,data,color='#237c72',height=.63,label='Evidence');axes[2].barh(y,control,left=data,color='#b8c4cb',height=.63,label='Control')
    axes[2].set(xlabel='Transmitted tokens per question',title='Observed communication traffic');axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle(f'2WikiMultihopQA pilot · N=3 · {stats[keys[0]]["n"]} assigned, {stats[keys[0]].get("cost_question_records",stats[keys[0]]["n"])} sealed questions · gpt-oss-120b',fontsize=12)
    fig.text(.01,.01,'Exploratory pilot. Accuracy includes every assigned question; hatching shows strict-score uncertainty. Cost means cover sealed questions only; cross-arm references overlap.',fontsize=8)
    fig.tight_layout(rect=(0,.11,1,.95))
    for extension in ('png','svg'):
        target=out/f'pilot_results.{extension}'
        fig.savefig(target,dpi=180,bbox_inches='tight')
        if extension=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    verification=verify_run(RUN)
    cli(['report','--run',str(RUN),'--cohort','pilot','--scientific'])
    official=json.loads((RUN/'reports/pilot/report.json').read_text());records=official['records']
    assert official['coverage']=={'expected':20,'saved':20,'missing':0}
    assert all(r['status']=='SAVED' and r['team_size']==3 and r['mode']=='REAL' for r in records)
    stats={name:summary(records,name) for name in NAMES}
    assert all(s['n']==20 for s in stats.values())
    op_rows=[{'question_key':r['question_key'],'mode':'REAL','arms':{name:{'em':r['arms'][name]['sink'].get('em') or 0,'f1':r['arms'][name]['sink'].get('f1') or 0} for name in NAMES}} for r in records]
    strict_rows=[{'question_key':r['question_key'],'mode':'REAL','arms':{name:{'em':r['arms'][name]['sink'].get('em')} for name in NAMES}} for r in records]
    comparisons={}
    for a,b in [('selected','random'),('selected','nocomm'),('selected','private'),('selected','static'),('selected','dense'),('selected','instar'),('instar','dense'),('selected','full_union')]:
        comparisons[a+'_minus_'+b]={'operational_em':paired_summary(op_rows,(a,b)),
                                    'strict_em':paired_summary(strict_rows,(a,b)),
                                    'operational_f1':paired_summary(op_rows,(a,b),metric='f1')}
    raw_runs=[json.loads(p.read_text())['run'] for p in (RUN/'results').glob('*.json')]
    assert sum(len(r['logical_refs']) for r in raw_runs)==240
    initial_hash=json.loads((RUN/'initial_preflight.json').read_text())['initial_state_hash']
    from WEBExtension.web_reasoning.serialization import run_from_dict
    from WEBExtension.web_reasoning.storage import digest
    first=json.loads((RUN/'initial_preflight.json').read_text())['question_key']
    assert digest(next(run_from_dict(r).initial for r in raw_runs if r['question_key']==first))==initial_hash
    prediction_rows=[];errors=Counter();finish=Counter();reported_models=Counter()
    raw_by_key={r['question_key']:r for r in raw_runs}
    targets=json.loads((RUN/'pilot_targets.json').read_text());publics={r['key']:r for r in json.loads((RUN/'pilot_inputs.json').read_text())}
    placement=Counter();types=Counter()
    for record in records:
        key=record['question_key'];raw=raw_by_key[key];gold=targets[key];types[dict(gold['strata'])['type']]+=1
        titles={t for t,_ in gold['supporting_locators']};sink=raw['arms']['selected']['sink']
        shard_titles=[{a['title'] for a in s['visible_atoms']} for s in raw['initial']]
        placement['gold_co_located' if any(titles<=s for s in shard_titles) else 'gold_disjoint']+=1
        placement['sink_all_gold' if titles<=shard_titles[sink] else 'sink_some_gold' if titles&shard_titles[sink] else 'sink_no_gold']+=1
        row={'question_key':key,'question':publics[key]['question'],'gold_answer':gold['answer'],'type':dict(gold['strata'])['type'],'arms':{}}
        for name in NAMES:
            outcome=raw['arms'][name]['outcome'];proposal=outcome['proposal']
            row['arms'][name]={'answer':proposal['answer'] if proposal else None,'status':proposal['status'] if proposal else None,
                               'outcome_kind':outcome['kind'],'reason':outcome['reason'],'em':record['arms'][name]['sink'].get('em'),'f1':record['arms'][name]['sink'].get('f1')}
            if outcome['kind']=='invalid':errors[outcome['reason']]+=1
        prediction_rows.append(row)
    for path in (RUN/'inference/attempts').glob('*.json'):
        raw=json.loads(path.read_text())['raw']
        for choice in raw.get('choices',[]):finish[choice.get('finish_reason')]+=1
        reported_models[raw.get('model','unknown')]+=1
    result={'status':'COMPLETED_EXPLORATORY_REAL_PILOT','date':'2026-10-16','n_questions':20,'team_size':3,
            'generated_at_utc':datetime.now(timezone.utc).isoformat(),'methods':stats,'comparisons':comparisons,
            'physical_usage':verification['physical_usage'],'strict_report_status':official['status'],
            'sink_invalid_reasons_across_arms':dict(errors),'provider_finish_reasons':dict(finish),'reported_models':dict(reported_models),
            'sample_types':dict(types),'placement':dict(placement),'manifest_sha256':file_sha(RUN/'manifest.json'),
            'analysis_sha256':file_sha(__file__),
            'operational_metric':'Number correct among all 20 assigned questions; invalid/unresolved inference outputs are pipeline failures worth zero. This is distinct from the official adapter, which leaves their answer scores incomplete.',
            'cost_definition':'Per-arm input/output references include shared initial agent calls. Cross-arm sums overlap and are not the physical bill. Communication includes complete envelope and control tokens.',
            'limitations':['Exploratory training-split pilot, no held-out test claim.','20 paired questions cannot establish equivalence or a scaling law.','Common improved prompts do not isolate a prompt-change effect.','Selected router is the untrained deterministic metadata heuristic; learned weights were not evaluated.','Source-valid eligibility excludes duplicate-title publisher questions.','Global source-component sampling was amended before inference to avoid selecting only the disconnected tail.','Serving weights are a deployment alias; immutable snapshot is not authenticated.','Valid citations establish visibility, not semantic faithfulness.']}
    out=ROOT/'analysis';out.mkdir(exist_ok=True);atomic_json(out/'summary.json',result);atomic_json(out/'predictions.json',prediction_rows)
    figures(stats,out)
    lines=['# 2WikiMultihopQA live pilot','',
           'Completed 20 real publisher training questions with N=3 and the existing Azure gpt-oss-120b deployment. This is exploratory, not a held-out test result.','',
           '## Measured outcomes','',
           '| Method | Correct / 20 | Operational EM | Operational F1 | Incomplete sink outputs | Logical calls | Input + output references | Evidence + control tokens |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in stats.items():lines.append(f'| {s["name"]} | {s["correct"]}/20 | {s["operational_em"]:.1f}% | {s["operational_f1"]:.2f}% | {s["incomplete_endpoints"]} | {s["logical_calls"]} | {s["input_token_reference_sum"]+s["output_token_reference_sum"]:,} | {s["payload_tokens"]+s["control_tokens"]:,} |')
    lines+=['',result['operational_metric'],'','Official score bounds and endpoint statuses remain in [the strict report](../representative_run/reports/pilot/report.json).','',
            '![Measured pilot outcomes](pilot_results.png)','','## Common readout diagnostics','',
            '| Method | Sink operational EM | Abstention-aware vote operational EM | Original vote operational EM | Incomplete abstention-aware votes |',
            '|---|---:|---:|---:|---:|']
    for name,s in stats.items():
        aa=s['readout_diagnostics']['abstention_vote'];original=s['readout_diagnostics']['original_vote_diagnostic']
        lines.append(f'| {s["name"]} | {s["operational_em"]:.1f}% | {aa["operational_em"]:.1f}% | {original["operational_em"]:.1f}% | {aa["incomplete"]} |')
    lines+=['','These operational vote figures include all assigned questions; incomplete readouts count as pipeline failures. Strict statuses remain in the official-adapter report.','',
            '## Paired comparisons','',
            'Differences are percentage points. Intervals resample whole questions 10,000 times. Only selected-minus-random was designated primary; other contrasts are exploratory. No multiple-comparison significance claim is made.','']
    for label,c in comparisons.items():
        op=c['operational_em'];strict=c['strict_em'];lines.append(f'- {label}: operational EM {op["effect"]:+.1f} pp, 95% CI [{op["ci95"][0]:.1f}, {op["ci95"][1]:.1f}], exact McNemar p={op["p_value"]:.4g}. Strict status: {strict["status"]}; identification bounds {strict["bounds"]}.')
    lines+=['','## Actual physical usage','','```json',json.dumps(result['physical_usage'],indent=2),'```','',result['cost_definition'],'',
            'The three preflight calls are reused by the final run. Persistent logical reference events therefore include three preflight entries in addition to the 240 references in the sealed question records. No extra generation was requested to repair malformed outputs.','',
            '## Source and protocol','',
            'Publisher: [official 2Wiki repository](https://github.com/Alab-NII/2wikimultihop), corrected April 7 release. [Source and sampling provenance](../source/representative_provenance.json), [frozen authorization and limits](../pre_inference_review.json), [complete question predictions](predictions.json).','',
            '## Limitations','']+[f'- {s}' for s in result['limitations']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'methods':stats,'comparisons':comparisons,'physical_usage':result['physical_usage'],'invalid_reasons':dict(errors),'finish_reasons':dict(finish)},indent=2))


if __name__=='__main__':main()
