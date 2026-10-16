"""Engineering evidence and scientific scores have separate explicit entry points."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from .contracts import PublicExample,Source,Outcome,Proposal
from .readout import abstention_vote,original_vote,normalize
from .evaluators import OfficialEvaluator
from .statistics import paired_summary
from .storage import atomic_json,Journal,digest
from .accounting import valid_usage


def source_union(run):
    sources={}
    for state in run.initial:
        for atom in state.visible_atoms:sources.setdefault((atom.document_id,atom.title),{})[atom.atom_id]=atom
    return PublicExample(run.question_key,run.dataset,'report-only source registry',
                         tuple(Source(d,t,tuple(sorted(atoms.values(),key=lambda a:a.sentence_id))) for (d,t),atoms in sorted(sources.items())))


def referenced_cost(refs):
    ledgers={};known_in=known_out=missing_in=missing_out=0;physical=set()
    for ref in refs:
        path=ref['source_ledger']
        if path not in ledgers:ledgers[path]={e['attempt_id']:e for e in Journal(path).read() if e['event']=='success'}
        record=ledgers[path].get(ref['attempt_id'])
        if record is None or record['key']!=ref['key']:raise ValueError('logical_reference_lacks_physical_success')
        physical.add((path,ref['attempt_id']));usage=record.get('usage') or {}
        a,b=usage.get('prompt_tokens'),usage.get('completion_tokens')
        if valid_usage(a):known_in+=a
        else:missing_in+=1
        if valid_usage(b):known_out+=b
        else:missing_out+=1
    return {'logical_calls':len(refs),'referenced_unique_physical_successes':len(physical),
            'logical_input_token_reference_sum':known_in,'logical_output_token_reference_sum':known_out,
            'input_usage_missing':missing_in,'output_usage_missing':missing_out,
            'interpretation':'logical reference sums are not additive billed cost across arms; physical attempts live in source ledgers'}


def write_report(question_runs,targets,output_dir,scientific=False,publics=None,provenance=None,
                 expected_cells=None,cohort=None,dataset=None,mode=None,expected_arms=None):
    datasets={r.dataset for r in question_runs}|({dataset} if dataset else set())
    if len(datasets)!=1:raise ValueError('one_explicit_benchmark_required')
    dataset=next(iter(datasets))
    if scientific and (any(r.mode!='REAL' for r in question_runs) or (mode is not None and mode!='REAL') or (not question_runs and mode!='REAL')):
        raise ValueError('mock_is_not_scientific_evidence')
    saved={(r.question_key,r.n_agents):r for r in question_runs}
    if len(saved)!=len(question_runs):raise ValueError('duplicate_report_cell')
    expected=list(expected_cells) if expected_cells is not None else list(saved)
    if not expected or len(set(expected))!=len(expected) or not set(saved)<=set(expected):raise ValueError('invalid_expected_report_population')
    if cohort and any(r.cohort not in ('', 'evaluation' if cohort=='scaling' else cohort) for r in question_runs):raise ValueError('report_cohort_mismatch')
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    records=[];paired=[]
    for run in question_runs:
        public=(publics or {}).get(run.question_key) or source_union(run)
        evaluator=OfficialEvaluator(run.dataset) if scientific else None
        target=targets[run.question_key] if scientific else None
        arms={};initial_refs=tuple(r for r in run.logical_refs if r.get('arm')=='initial')
        for name,arm in run.arms.items():
            refs=arm.references if name=='full_union' else initial_refs+arm.references
            item={'cost':referenced_cost(refs),'sink_outcome':arm.outcome.kind,
                  'endpoint_states':dict(Counter(s.outcome.kind for s in arm.states)),
                  'data_deliveries':sum(p.kind!='control' for p in arm.transmissions),
                  'control_frames':sum(p.kind=='control' for p in arm.transmissions),
                  'payload_tokens':sum(p.tokens for p in arm.transmissions if p.kind!='control'),
                  'control_tokens':sum(p.tokens for p in arm.transmissions if p.kind=='control'),
                  'complete_delivered_atoms':sum(len(p.facts) for p in arm.transmissions),
                  'omitted_atoms':sum(len(p.omitted_atom_ids) for p in arm.transmissions),'stop_reason':arm.stop_reason}
            if scientific:
                sink_state=next(s for s in arm.states if s.agent_id==arm.sink)
                item['sink']=evaluator.score(run.question_key,arm.outcome,target,public=public,visible_atoms=sink_state.visible_atoms)
                for label,rule in (('abstention_vote',abstention_vote),('original_vote_diagnostic',original_vote)):
                    try:
                        vote=rule(arm.states,tuple(s.agent_id for s in arm.states))
                        winner=next((s for s in arm.states if s.outcome.proposal and s.outcome.proposal.answer is not None and normalize(s.outcome.proposal.answer)==normalize(vote)),None)
                        item[label]=evaluator.score(run.question_key,winner.outcome if winner else Outcome('valid',Proposal('no_evidence',None)),target,
                                                    public=public,visible_atoms=winner.visible_atoms if winner else ())
                    except ValueError:item[label]={'status':'INCOMPLETE','em':None,'f1':None}
                visible={(a.title,a.sentence_id) for s in arm.states if s.agent_id==arm.sink for a in s.visible_atoms}
                item['annotated_support_visible_to_sink']=set(target.supporting_locators)<=visible if target.supporting_locators else None
            arms[name]=item
        records.append({'question_key':run.question_key,'team_size':run.n_agents,'cohort':cohort or run.cohort,'mode':run.mode,'status':'SAVED','arms':arms,'timings':run.timings})
        if scientific:
            metric='em' if run.dataset=='2wiki' else 'execution_accuracy'
            paired.append({'question_key':run.question_key,'team_size':run.n_agents,'mode':run.mode,
                           'arms':{name:{'em':item['sink'].get(metric)} for name,item in arms.items()}})
    arm_names=tuple(expected_arms or ('nocomm','private','static','random','selected','instar','dense','full_union'))
    arm_names=tuple(dict.fromkeys(arm_names+tuple(name for run in question_runs for name in run.arms)))
    for key,n in sorted(set(expected)-set(saved)):
        arms={name:{'status':'NOT_RUN','sink':{'status':'INCOMPLETE','em':None,'f1':None,'execution_accuracy':None},
                    'cost':{'logical_calls':None},'data_deliveries':None,'control_frames':None} for name in arm_names}
        records.append({'question_key':key,'team_size':n,'cohort':cohort,'mode':mode,'status':'MISSING','reason':'assigned_cell_not_saved','arms':arms,'timings':{}})
        if scientific:paired.append({'question_key':key,'team_size':n,'mode':'REAL','arms':{name:{'em':None} for name in arm_names}})
    contrasts={};stopping_contrasts={}
    if scientific:
        for n in sorted({n for _,n in expected}):
            contrasts[str(n)]=paired_summary([r for r in paired if r['team_size']==n],('selected','random'))
            if {'selected_always_run','selected_observable_stop'}<=set(arm_names):
                stopping_contrasts[str(n)]=paired_summary([r for r in paired if r['team_size']==n],('selected_always_run','selected_observable_stop'))
    coverage={'expected':len(expected),'saved':len(saved),'missing':len(expected)-len(saved)}
    incomplete=coverage['missing'] or any(v['status']=='INCOMPLETE' for v in list(contrasts.values())+list(stopping_contrasts.values()))
    summary={'status':('INCOMPLETE SCIENTIFIC' if incomplete else 'SCIENTIFIC') if scientific else 'ENGINEERING — NO SCIENTIFIC SCORES',
             'dataset':dataset,'cohort':cohort,'coverage':coverage,'records':records,'selected_minus_random_by_n':contrasts,
             'stopping_always_minus_observable_by_n':stopping_contrasts,
             'routing_policies':list({digest(r.policy_metadata):r.policy_metadata for r in question_runs}.values()),
             'provenance':provenance,
             'cost_warning':'Cross-arm references overlap; use immutable physical ledgers for unique actual attempts. Currency cost unknown.',
             'source_support_warning':'Annotated support and valid citations do not establish semantic necessity or entailment.'}
    atomic_json(directory/'report.json',summary)
    lines=['# Web reasoning '+('scientific report' if scientific else 'engineering validation'),' ',summary['status'],'',
           f'Question/team-size records: {len(records)}. No estimated legacy points are included.','',summary['cost_warning'],'']
    lines+=['Coverage: '+json.dumps(coverage,sort_keys=True),'']
    for n in sorted({n for _,n in expected}):
        subset=[r for r in records if r['team_size']==n];lines+=['## Team size '+str(n),'']
        for name in subset[0]['arms']:
            values=[r['arms'][name] for r in subset if r['status']=='SAVED']
            lines.append(f'- {name}: {sum(x["cost"]["logical_calls"] for x in values)} logical calls; '
                         f'{sum(x["data_deliveries"] for x in values)} data/request deliveries; '
                         f'{sum(x["control_frames"] for x in values)} control frames; {len(subset)-len(values)} missing assigned cells.')
        if scientific:lines+=['',json.dumps(contrasts[str(n)],sort_keys=True)]
        if str(n) in stopping_contrasts:lines+=['','Stopping contrast (same initial states): '+json.dumps(stopping_contrasts[str(n)],sort_keys=True)]
        lines.append('')
    path=directory/'report.md';path.write_text('\n'.join(lines)+'\n');return path
