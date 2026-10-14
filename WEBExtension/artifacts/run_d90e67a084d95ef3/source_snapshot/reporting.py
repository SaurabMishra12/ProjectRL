"""Validated question-paired reports from completed real-inference experiment artifacts."""
import json
from pathlib import Path
import pandas as pd
from protocol import ROOT, digest, file_hash, paired_effect, write_json

def load_completed(run_dir, experiment):
    run_dir = Path(run_dir)
    status_path = run_dir / (experiment + '_status.json')
    if not status_path.exists(): raise FileNotFoundError(f'{experiment} has not run')
    status = json.loads(status_path.read_text())
    path = run_dir / (experiment + '.jsonl')
    if status['status'] != 'COMPLETE' or file_hash(path) != status['sha256']:
        raise ValueError('Incomplete or changed experiment; cannot publish a complete result')
    frozen = json.loads((run_dir / 'frozen_manifest.json').read_text())
    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    expected = frozen['plan']['question_ids']
    if len(rows) != len(expected) or set(r['id'] for r in rows) != set(expected):
        raise ValueError('Sample incomplete, duplicated, or changed')
    return rows

def condition_table(rows):
    records = []
    for row in rows:
        for label, c in row['conditions'].items():
            records.append({'id': row['id'], 'type': row['type'], 'experiment': row['experiment'], 'condition': label,
                            **{k:c[k] for k in ['answer', 'f1', 'em', 'payload_tokens', 'support_available_recall',
                                                'all_support_available', 'split_support_across_nodes',
                                                'answer_support_recall', 'answer_support_precision', 'invalid_citation_count']},
                            'new_support_facts_delivered':sum(c['new_support_facts_delivered_by_node']),
                            'provider_refusal_count':c.get('provider_refusal_count',0),
                            'provider_refusal_calls':c['cost'].get('provider_refusal_calls',0),
                            **{k:v for k,v in c['cost'].items() if k not in ['call_keys']}})
    return pd.DataFrame(records)

def effects(rows, experiment):
    bylabel = {}
    for r in rows:
        for label, c in r['conditions'].items(): bylabel.setdefault(label, {})[r['id']] = c['f1']
    if experiment == 'E1':
        contrasts = [('R_CW','R_NoComm'),('P_CW','P_NoComm'),('R_CCW','R_NoComm'),('P_CCW','P_NoComm')]
        out = [{'contrast':a+' - '+b, 'primary':False, **paired_effect(bylabel[a],bylabel[b])} for a,b in contrasts]
        p = {q:bylabel['P_CW'][q]-bylabel['P_NoComm'][q] for q in bylabel['P_CW']}
        r = {q:bylabel['R_CW'][q]-bylabel['R_NoComm'][q] for q in bylabel['R_CW']}
        out.insert(0, {'contrast':'(CW-NoComm)_P - (CW-NoComm)_R','primary':True, **paired_effect(p,r)})
        if 'R_SelfUpdate' in bylabel:
            for regime in ['R','P']:
                out.append({'contrast':regime+'_CW - '+regime+'_SelfUpdate','primary':False,
                            **paired_effect(bylabel[regime+'_CW'],bylabel[regime+'_SelfUpdate'])})
    else:
        a,b = {'E2':('R_CW','R_CCW'),'E3':('R_prefix','R_crop'),'E4':('P_stopped','P_final')}[experiment]
        out = [{'contrast':a+' - '+b,'primary':False, **paired_effect(bylabel[a],bylabel[b])}]
    # Holm correction across exploratory contrasts within this experiment.
    secondary = sorted((x for x in out if not x['primary']), key=lambda x:x['randomization_p'])
    running = 0
    for i,x in enumerate(secondary):
        running = max(running, min(1, (len(secondary)-i)*x['randomization_p']))
        x['holm_p_within_experiment'] = running
    return out

def report(run_dir, experiment):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = load_completed(run_dir, experiment)
    directory = Path(run_dir) / 'reports' / experiment
    directory.mkdir(parents=True, exist_ok=True)
    df = condition_table(rows)
    df.to_csv(directory / 'question_level.csv', index=False)
    summary = df.groupby(['experiment','condition']).agg(n=('id','size'), F1=('f1','mean'), EM=('em','mean'),
        support_available_recall=('support_available_recall','mean'), support_recall=('answer_support_recall','mean'),
        support_precision=('answer_support_precision','mean'), payload_tokens=('payload_tokens','mean'),
        mean_input_tokens=('input_tokens',lambda x:x.mean() if x.notna().all() else float('nan')),
        mean_output_tokens=('output_tokens',lambda x:x.mean() if x.notna().all() else float('nan')),
        calls=('logical_calls','mean'), usage_complete=('usage_complete','all'),
        refused_terminal_responses=('provider_refusal_count','sum'),
        refused_logical_calls=('provider_refusal_calls','sum')).reset_index()
    summary.to_csv(directory / 'summary.csv',index=False)
    tests = effects(rows, experiment)
    details = {'experiment':experiment,'effects':tests,'scale':'F1/EM percentage points',
               'provider_refusal_policy':'Explicit provider refusals are abstentions with preserved raw evidence and reported costs; malformed or truncated responses still stop the run',
               'citation_caveat':'Supporting-label overlap is a diagnostic; it does not prove entailment',
               'cost_caveat':'Condition cost includes shared initial calls. Physical measured usage is separate.',
               'stopping_caveat':'Replay savings are hypothetical, not observed bill reductions'}
    if experiment == 'E3': details['identical_prefix_crop_questions'] = sum(r['crop_equals_prefix'] for r in rows)
    if experiment == 'E4':
        details.update({'wrong_unanimity':sum(r['wrong_unanimity'] for r in rows),
                        'continuation_corrected_wrong_stop':sum(r['continuation_corrected_wrong_stop'] for r in rows),
                        'hypothetical_calls_avoided':sum(r['hypothetical_calls_avoided'] for r in rows),
                        'stop_depth_by_question':{r['id']:r['stop_depth'] for r in rows}})
        cfg=json.loads((Path(run_dir)/'frozen_manifest.json').read_text())['plan']['config']
        margin=cfg.get('noninferiority_margin_f1')
        if margin is not None:
            if not cfg.get('noninferiority_justification') or margin <= 0: raise ValueError('Require justified positive margin')
            details['noninferiority']={'margin_f1':margin, 'conservative_two_sided_lower_bound':tests[0]['ci95'][0],
                'supported':tests[0]['ci95'][0]>-margin,
                'method':'Conservative two-sided 95% paired percentile interval lower bound; not absence of significance'}
    write_json(directory/'statistics.json',details)
    for x in ['payload_tokens','inference_tokens']:
        fig,ax=plt.subplots(figsize=(7,4))
        for _,r in summary.iterrows():
            value=r['payload_tokens'] if x=='payload_tokens' else r['mean_input_tokens']+r['mean_output_tokens']
            if pd.isna(value): continue
            ax.scatter(value,r['F1']); ax.annotate(r['condition'],(value,r['F1']),fontsize=8)
        ax.set(xlabel='Mean '+x.replace('_',' ')+' per question',ylabel='Answer F1 (percentage points)',title=experiment)
        fig.tight_layout();fig.savefig(directory/(x+'_frontier.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    y=list(range(len(tests)))
    for i,t in enumerate(tests):
        ax.plot(t['ci95'],[i,i],color='black');ax.scatter(t['effect'],i)
    ax.axvline(0,linestyle='--',color='gray');ax.set_yticks(y,[t['contrast'] for t in tests])
    ax.set_xlabel('Paired F1 change with 95% question-bootstrap interval')
    fig.tight_layout();fig.savefig(directory/'paired_effects.png',dpi=180);plt.close(fig)
    examples=[]
    for label,group in df.groupby('condition'):
        baseline='R_NoComm' if label.startswith('R_') else 'P_NoComm'
        if baseline not in set(df.condition) or label==baseline: continue
        joined=group.merge(df[df.condition==baseline][['id','f1','em']],on='id',suffixes=('','_base'))
        joined['delta']=joined.f1-joined.f1_base
        for direction,g in [('helped',joined[joined.delta>0]),('harmed',joined[joined.delta<0])]:
            # Deterministic largest effects for examples; no replacement of whole-sample analysis.
            for _,r in g.sort_values(['delta','id'],ascending=[direction=='harmed',True]).head(5).iterrows():
                examples.append({'condition':label,'direction':direction,'id':r['id'],'f1_delta':r['delta'],
                                 'raw_record':next(q for q in rows if q['id']==r['id'])})
    write_json(directory/'evidence_trace_examples.json',examples)
    return summary,details

def physical_usage(run_dir):
    path=Path(run_dir)/'calls.jsonl'
    if not path.exists(): return {'status':'NOT_RUN','attempted_calls':0,'input_tokens':None,'output_tokens':None}
    logs=[json.loads(l) for l in path.read_text().splitlines() if l]
    successful=[r for r in logs if r['event']=='success']
    complete=bool(successful) and all(r.get('usage') and r['usage'].get('prompt_tokens') is not None and
                                     r['usage'].get('completion_tokens') is not None for r in successful)
    return {'attempted_calls':sum(r['event']=='started' for r in logs),'successful_calls':len(successful),
            'provider_refusal_calls':sum(bool(r.get('provider_refusal')) for r in successful),
            'error_calls':sum(r['event'] in ['error','uncertain','rejected'] for r in logs), 'usage_complete_for_successes':complete,
            'input_tokens':sum(r['usage']['prompt_tokens'] for r in successful) if complete else None,
            'output_tokens':sum(r['usage']['completion_tokens'] for r in successful) if complete else None,
            'caveat':'Errors/in-flight calls may be billable and have unknown usage. No price or monetary saving inferred.'}
