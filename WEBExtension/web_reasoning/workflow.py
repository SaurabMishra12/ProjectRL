"""Run frozen public inputs; private targets are loaded only by reporting."""
from dataclasses import asdict
import json
from pathlib import Path
from .preparation import verify_run,make_tokenizer,file_sha,forecast
from .config import validate_config
from .contracts import AgentState
from .backend import DurableBackend
from .runner import run_w1,initialize,call,ArmResult,QuestionRun,contract_hash,policy_metadata
from .adaptive import run_adaptive
from .storage import atomic_json,digest
from .serialization import public_from_dict,run_from_dict


def mock_transport(request,attempt_id):
    payload=json.loads(request['messages'][-1]['content']);atoms=payload['visible_sources']
    body={'status':'partial' if atoms else 'no_evidence','answer':None,
          'facts':[{'fact_id':a['atom_id'],'document_id':a['document_id'],'sentence_id':a['sentence_id'],'quote':a['text'],'claim':'Engineering fixture fact'} for a in atoms[:4]],
          'entity_tags':[a['title'] for a in atoms[:8]],'needs':([{'entity':atoms[0]['title'],'relation':'requested attribute'}] if atoms else []),'derivation':[],'conflicts':[]}
    if 'program' in payload['output_schema']['properties']:body.update(program=[],operand_sources=[])
    if set(payload['output_schema']['properties'])=={'answer','citations'}:body={'answer':'noanswer','citations':[[a['title'],a['sentence_id']] for a in atoms[:4]]}
    # Fixture usage is an engineering byte count, explicitly not a provider observation.
    return 200,{}, {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(body)}}],
                    'usage':{'prompt_tokens':len(json.dumps(request['messages']).encode()),'completion_tokens':len(json.dumps(body).encode())}}


def w2(public,assignments,sink,backend,config):
    start=len(backend.references);initial=initialize(public,assignments,backend,config)
    arms={'nocomm':ArmResult('nocomm',initial,sink,initial[sink].outcome)}
    for policy in ('private','static','random','selected','instar','dense'):
        arms[policy]=run_adaptive(public,initial,sink,backend,config,policy,always_run=not config.stopping)
    offset=len(backend.references);atoms=tuple(a for source in public.sources for a in source.atoms)
    outcome=call(public,atoms,None,(),None,'initial',backend,config,{'arm':'full_union','stage':'initial','node':sink})
    state=AgentState(sink,atoms,outcome.proposal.facts if outcome.proposal else (),outcome)
    arms['full_union']=ArmResult('full_union',(state,),sink,outcome,tuple(backend.references[offset:]))
    return QuestionRun(public.key,public.dataset,len(initial),initial,arms,tuple(backend.references[start:]),
                       tuple(p for arm in arms.values() for p in arm.transmissions),config.mode,
                       contract_hash(public,assignments,sink,config,backend.tokenizer.identity),policy_metadata=policy_metadata(config))


def run_manifest(manifest_path,mode='mock',cohort='pilot',permit_path=None):
    manifest_path=Path(manifest_path).resolve();root=manifest_path.parent;manifest=json.loads(manifest_path.read_text())
    config=validate_config(manifest['config'])
    if mode=='live':
        from .live import require_live_permit
        permit=json.loads(Path(permit_path).read_text()) if permit_path else None
        require_live_permit(file_sha(manifest_path),permit,config)
        if config.mode!='REAL':raise ValueError('mock_manifest_cannot_dispatch_live')
    elif mode!='mock' or config.mode!='MOCK':raise ValueError('execution_mode_mismatch')
    verify_run(root);tokenizer=make_tokenizer(config,manifest['sources'])
    if mode=='live':
        from .live import live_transport
        transport=live_transport(manifest['sources'])
    else:transport=mock_transport
    backend=DurableBackend(root/'inference',config,transport,tokenizer)
    original_cohort='evaluation' if cohort=='scaling' else cohort
    if original_cohort not in ('pilot','development','evaluation'):raise ValueError('unknown_cohort')
    publics=tuple(public_from_dict(x) for x in json.loads((root/(original_cohort+'_inputs.json')).read_text()))
    keys=set(manifest['scaling_keys']) if cohort=='scaling' else {p.key for p in publics}
    ns=config.team_sizes if cohort=='scaling' else (3,)
    expected_calls=sum(len(keys)*(2*n+6 if config.experiment!='W2' else 5*n+4*max(1,n//2)+10) for n in ns)
    if expected_calls>config.max_completions:raise ValueError('cohort_logical_ceiling_exceeds_frozen_budget')
    runs=[]
    for n in ns:
        assignments=json.loads((root/f'{original_cohort}_N{n}_assignments.json').read_text())
        for public in publics:
            if public.key not in keys:continue
            lookup={a.atom_id:a for source in public.sources for a in source.atoms}
            assigned=tuple(tuple(lookup[key] for key in shard) for shard in assignments[public.key])
            sink=manifest['sinks'][original_cohort][str(n)][public.key]
            expected=contract_hash(public,assigned,sink,config,tokenizer.identity)
            path=root/'results'/f'{digest([original_cohort,public.key,n])}.json'
            if path.exists():
                saved=json.loads(path.read_text());run=run_from_dict(saved['run'])
                if run.contract_hash!=expected:raise ValueError('saved_run_contract_mismatch')
            else:
                run=(w2 if config.experiment=='W2' else run_w1)(public,assigned,sink,backend,config)
                atomic_json(path,{'run':asdict(run),'sha256':digest(asdict(run)),'manifest_sha256':file_sha(manifest_path)})
            runs.append(run)
    atomic_json(root/'usage_summary.json',backend.summary())
    return tuple(runs)
