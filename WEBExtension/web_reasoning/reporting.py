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


def write_report(question_runs,targets,output_dir,scientific=False,publics=None,provenance=None):
    if not question_runs:raise ValueError('nonempty_run_required')
    if len({r.dataset for r in question_runs})!=1:raise ValueError('do_not_pool_benchmarks')
    if scientific and any(r.mode!='REAL' for r in question_runs):raise ValueError('mock_is_not_scientific_evidence')
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
                item['sink']=evaluator.score(run.question_key,arm.outcome,target,public=public)
                for label,rule in (('abstention_vote',abstention_vote),('original_vote_diagnostic',original_vote)):
                    try:
                        vote=rule(arm.states,tuple(s.agent_id for s in arm.states))
                        winner=next((s.outcome for s in arm.states if s.outcome.proposal and s.outcome.proposal.answer is not None and normalize(s.outcome.proposal.answer)==normalize(vote)),
                                    Outcome('valid',Proposal('no_evidence',None)))
                        item[label]=evaluator.score(run.question_key,winner,target,public=public)
                    except ValueError:item[label]={'status':'INCOMPLETE','em':None,'f1':None}
                visible={(a.title,a.sentence_id) for s in arm.states if s.agent_id==arm.sink for a in s.visible_atoms}
                item['annotated_support_visible_to_sink']=set(target.supporting_locators)<=visible if target.supporting_locators else None
            arms[name]=item
        records.append({'question_key':run.question_key,'team_size':run.n_agents,'mode':run.mode,'arms':arms,'timings':run.timings})
        if scientific:
            metric='em' if run.dataset=='2wiki' else 'execution_accuracy'
            paired.append({'question_key':run.question_key,'team_size':run.n_agents,'mode':run.mode,
                           'arms':{name:{'em':item['sink'].get(metric)} for name,item in arms.items()}})
    contrasts={}
    if scientific:
        for n in sorted({r.n_agents for r in question_runs}):
            contrasts[str(n)]=paired_summary([r for r in paired if r['team_size']==n],('selected','random'))
    summary={'status':'SCIENTIFIC' if scientific else 'ENGINEERING — NO SCIENTIFIC SCORES',
             'dataset':question_runs[0].dataset,'records':records,'selected_minus_random_by_n':contrasts,
             'routing_policies':list({digest(r.policy_metadata):r.policy_metadata for r in question_runs}.values()),
             'provenance':provenance,
             'cost_warning':'Cross-arm references overlap; use immutable physical ledgers for unique actual attempts. Currency cost unknown.',
             'source_support_warning':'Annotated support and valid citations do not establish semantic necessity or entailment.'}
    atomic_json(directory/'report.json',summary)
    lines=['# Web reasoning '+('scientific report' if scientific else 'engineering validation'),' ',summary['status'],'',
           f'Question/team-size records: {len(records)}. No estimated legacy points are included.','',summary['cost_warning'],'']
    for n in sorted({r.n_agents for r in question_runs}):
        subset=[r for r in records if r['team_size']==n];lines+=['## Team size '+str(n),'']
        for name in subset[0]['arms']:
            values=[r['arms'][name] for r in subset]
            lines.append(f'- {name}: {sum(x["cost"]["logical_calls"] for x in values)} logical calls; '
                         f'{sum(x["data_deliveries"] for x in values)} data/request deliveries; '
                         f'{sum(x["control_frames"] for x in values)} control frames.')
        if scientific:lines+=['',json.dumps(contrasts[str(n)],sort_keys=True)]
        lines.append('')
    path=directory/'report.md';path.write_text('\n'.join(lines)+'\n');return path
