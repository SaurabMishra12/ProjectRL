"""Run exactly the authorized gate, with persistent physical reuse and no tuning."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from WEBExtension.web_reasoning.backend import DurableBackend,Request,BudgetExceeded
from WEBExtension.web_reasoning.config import validate_config
from WEBExtension.web_reasoning.evaluators import load_reference
from WEBExtension.web_reasoning.preparation import make_tokenizer,file_sha
from WEBExtension.web_reasoning.serialization import public_from_dict
from WEBExtension.web_reasoning.storage import atomic_json,digest,Journal,canonical
from WEBExtension.web_reasoning.live import live_transport
from WEBExtension.causal_study.readout_reassessment.analysis import paired
from .core import ARMS,READOUTS,parse,messages,packet,choose,vote,gate,norm
from .prepare import ROOT


def verify():
    m=json.loads((ROOT/'manifest.json').read_text());assert file_sha(ROOT/'manifest.json')==(ROOT/'manifest.sha256').read_text().strip()
    for name,h in m['files'].items():assert file_sha(ROOT/name)==h,name
    for name,h in {**m['code'],**m['backend_code'],**m['source_hashes']}.items():assert file_sha(name)==h,name
    return m


def physical_audit(backend):
    summary=backend.summary();kinds=Counter();reasons=Counter();warnings=Counter()
    for p in (backend.directory/'responses').glob('*.json'):
        outcome=parse(json.loads(p.read_text())['raw']);kinds[outcome['kind']]+=1
        if outcome['reason']:reasons[outcome['reason']]+=1
        warnings.update(outcome['warnings'])
    return {**summary,'contract_failures':kinds['invalid']+kinds['refusal'],'relaxed_parse_kinds':dict(kinds),
            'parse_failure_reasons':dict(reasons),'nonfatal_metadata_warnings':dict(warnings)}


def evaluate_row(public,shards,target,backend,config,tokenizer,phase):
    start=len(backend.references);model={'model':config.model,'deployment_revision':config.deployment_revision,
        'provider':'azure_v1','endpoint':config.provider_base_url,'replicate_id':config.replicate_id}
    decoding={'dataset':'2wiki','prompt_variant':'enhanced','temperature':0.,'max_tokens':config.max_output_per_call}
    def call(atoms,previous=None,incoming=(),arm='initial',node=0):
        request=Request(messages(public.question,atoms,previous,incoming),model,tokenizer.identity,decoding)
        # DurableBackend owns physical accounting and returns its legacy strict parse.
        # The experimental contract intentionally parses the preserved raw response below.
        backend.complete(request,{'qid':public.key,'phase':phase,'arm':arm,'node':node})
        saved=json.loads((backend.directory/'responses'/(request.key+'.json')).read_text())
        outcome=parse(saved['raw'])
        return {**outcome,'call_key':request.key,'usage':saved['raw'].get('usage') or {}}
    initial=[call(shard,node=i) for i,shard in enumerate(shards)]
    packets=[packet(state,i,tokenizer,config.packet_cap) for i,state in enumerate(initial)]
    titles=[list(dict.fromkeys(a.title for a in shard)) for shard in shards]
    choices={kind:choose(kind,initial,titles,public.question,config.seed) for kind in ('random','heuristic','complementary')}
    metadata=[canonical({'agent':i,'titles':titles[i],'missing_entities':s['missing_entities'],'answer_available':bool(s['answer'])}) for i,s in enumerate(initial)]
    official=load_reference('2wiki');golds=[target['answer'],*target['aliases']]
    def score(answer):
        values=[official.eval_answer(answer or 'noanswer',g) for g in golds]
        return {'answer':answer or 'noanswer','em':100.*max(v[0] for v in values),'f1':100.*max(v[1] for v in values)}
    arms={}
    for name in ARMS:
        if name=='full_union':
            states=[call(tuple(a for shard in shards for a in shard),arm=name,node=1)];edges=[];calls=[states[0]]
        else:
            states=list(initial)
            senders=choices.get(name,())
            if name=='instar':edges=[(0,1),(2,1)]
            elif name=='dense':edges=[(s,t) for t in range(3) for s in range(3) if s!=t]
            else:edges=[(s,1) for s in senders]
            for recipient in sorted({t for _,t in edges}):
                incoming=[packets[s] for s,t in edges if t==recipient]
                states[recipient]=call(shards[recipient],initial[recipient],incoming,name,recipient)
            calls=list(initial)+[states[i] for i in sorted({t for _,t in edges})]
        sink=states[0] if name=='full_union' else states[1]
        arms[name]={'states':states,'edges':[list(e) for e in edges],
            'hub_request_key':sink['call_key'],'readouts':{'sink1':score(sink.get('answer') if sink['kind']=='valid' else None),
                                                       'abstention_aware':score(vote(states))},
            'logical_call_keys':[c['call_key'] for c in calls],
            'known_input_tokens':sum(c['usage'].get('prompt_tokens') or 0 for c in calls),
            'known_output_tokens':sum(c['usage'].get('completion_tokens') or 0 for c in calls),
            'missing_usage_fields':sum(c['usage'].get(k) is None for c in calls for k in ('prompt_tokens','completion_tokens')),
            'payload_tokens':sum(packets[s]['tokens'] for s,_ in edges),
            'omitted_facts':sum(packets[s]['omitted_facts'] for s,_ in edges),
            'metadata_tokens':sum(tokenizer.count(x) for x in metadata) if name in ('heuristic','complementary') else 0}
    assert arms['instar']['hub_request_key']==arms['dense']['hub_request_key']
    return {'question_key':public.key,'question':public.question,'phase':phase,'arms':arms,'initial':initial,
            'router_choices':choices,'packets':packets,'logical_refs':backend.references[start:]}


def write_report(rows,backend,manifest,stopped=''):
    audit=physical_audit(backend);pilot=[r for r in rows if r['phase']=='pilot'];decision=gate(pilot,audit)
    # The frozen gate evaluates pilot physical outputs only, not later expansion failures.
    existing=ROOT/'gate_decision.json'
    if existing.exists():decision=json.loads(existing.read_text())
    stats={};contrasts={}
    cohorts={'pilot':pilot,'expansion':[r for r in rows if r['phase']=='expansion'],'pooled':rows}
    for cohort,group in cohorts.items():
        if not group:continue
        expected=20 if cohort=='pilot' else 80 if cohort=='expansion' else 100 if decision['pass'] else 20
        n=expected;stats[cohort]={}
        for name in ARMS:
            stats[cohort][name]={'n':n,'completed_records':len(group),'readouts':{rule:{'correct':sum(r['arms'][name]['readouts'][rule]['em']==100 for r in group),
                'em':sum(r['arms'][name]['readouts'][rule]['em'] for r in group)/n,
                'f1':sum(r['arms'][name]['readouts'][rule]['f1'] for r in group)/n} for rule in READOUTS},
                **{metric:sum(r['arms'][name][metric] for r in group) for metric in ('known_input_tokens','known_output_tokens','missing_usage_fields','payload_tokens','metadata_tokens','omitted_facts')},
                'logical_calls':sum(len(r['arms'][name]['logical_call_keys']) for r in group)}
        if len(group)!=expected:continue
        for name,base in [('instar','nocomm'),('complementary','nocomm'),('complementary','random'),('complementary','heuristic'),('complementary','dense'),('instar','dense'),('full_union','dense')]:
            for rule in READOUTS:
                contrasts[cohort+':'+name+'-'+base+':'+rule]={m:paired({r['question_key']:r['arms'][name]['readouts'][rule][m] for r in group},
                    {r['question_key']:r['arms'][base]['readouts'][rule][m] for r in group},binary=m=='em') for m in ('em','f1')}
    result={'status':'STOPPED_'+stopped if stopped else 'COMPLETE_100' if len(rows)==100 else 'GATE_FAILED_STOPPED' if len(pilot)==20 and not decision['pass'] else 'RUNNING',
        'completed_questions':len(rows),'gate':decision,'stats':stats,'contrasts':contrasts,'physical_usage':audit,
        'router_k_distribution':dict(Counter(len(r['router_choices']['complementary']) for r in rows)),
        'manifest_sha256':file_sha(ROOT/'manifest.json'),'limitations':manifest['limitations'],
        'metrics':'Official alias-aware 2Wiki answer EM/F1 on operational predictions; unusable outputs count as zero, never dropped. No supporting-fact metric or semantic adjudication.',
        'cost_scope':'Per-arm logical references overlap through physical reuse. Payload includes complete transmitted packet headers. Deterministic router metadata counted separately; all-in provider currency unavailable.'}
    atomic_json(ROOT/'summary.json',result)
    lines=['# Revised 2Wiki: relaxed schema and complementary routing','',
           '**Status: '+result['status']+'**. Completed question records: '+str(len(rows))+'.','',
           'Frozen gate pass: **'+str(decision['pass'])+'**. Gate failures: '+canonical(decision['reasons'])+'.','',
           'Physical response contract failures (including refusals/truncation): '+str(audit['contract_failures'])+'/'+str(audit['physical_completions'])+'.','',
           result['metrics'],'',result['cost_scope'],'']
    for cohort,ss in stats.items():
        lines += ['## '+cohort,'','| Arm | AA EM count | AA F1 | Sink-1 EM count | Sink-1 F1 | Logical calls | Known model tokens | Packet tokens | Router metadata tokens |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name,s in ss.items():
            aa=s['readouts']['abstention_aware'];sink=s['readouts']['sink1']
            lines.append(f'| {name} | {aa["correct"]}/{s["n"]} | {aa["f1"]:.3f} | {sink["correct"]}/{s["n"]} | {sink["f1"]:.3f} | {s["logical_calls"]} | {s["known_input_tokens"]+s["known_output_tokens"]} | {s["payload_tokens"]} | {s["metadata_tokens"]} |')
    lines+=['','## Physical usage','','```json',json.dumps(audit,indent=2),'```','','## Scope','',
            '- No list-size or source-locator rejection is applied to otherwise usable answers. This removes those parser failure modes by design; it does not verify the truth of every fact.',
            '- Same system prompt and output contract in every arm. Full union is one call, read identically under both readouts.',
            '- A two-sender In-Star and Dense have the same one-round Sink-1 request by construction and reuse its physical response.',
            '- k=0 retains the original sink state. k=1/2 use frozen greedy coverage of unresolved entity names by public source-title metadata, falling back to two senders for unresolved unmatched needs.',
            '- The pilot gate is a prespecified operational expansion rule, not a significance or noninferiority claim. Expansion-only results must be shown separately from the conditionally selected pooled 100.',
            '- Original strict-schema pilot calls are not imported into this new prompt namespace. The old unresolved request is unchanged.',
            '- Explicit abstentions and operational failures remain in the denominator; refusal and parse diagnostics are separate from correctness.',
            '- Paired bootstrap intervals resample questions 10,000 times; all performance comparisons remain exploratory.',
            '', '[Frozen manifest](manifest.json) · [Complete numerical summary](summary.json) · [Saved records](results/)']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:raise PermissionError('explicit_execute_required')
    manifest=verify();config=validate_config(manifest['config']);tokenizer=make_tokenizer(config,manifest['descriptor'])
    publics=[public_from_dict(p) for p in json.loads((ROOT/'public_inputs.json').read_text())]
    targets=json.loads((ROOT/'targets.json').read_text());assign=json.loads((ROOT/'assignments.json').read_text())
    from dotenv import dotenv_values
    env=manifest['descriptor']['provider']['api_key_env']
    if not os.environ.get(env):
        key=dotenv_values('.env').get(env)
        if key:os.environ[env]=key
    send=live_transport(manifest['descriptor']);stage={'name':'pilot'};next_time=0.
    observations=Journal(ROOT/'transport_observations.jsonl')
    holder={}
    def transport(request,attempt_id):
        nonlocal next_time
        b=holder['backend'];events=b.journal.read();starts=[e for e in events if e['event']=='started']
        if stage['name']=='pilot':
            limits=manifest['pilot_ceiling'];closed={e['attempt_id']:e for e in events if e['event']=='success'}
            used_in=sum((e.get('usage') or {}).get('prompt_tokens') or 0 for e in closed.values())
            used_out=sum((e.get('usage') or {}).get('completion_tokens') or 0 for e in closed.values())
            # The current request has already been durably reserved; ceiling is checked before dispatch.
            if len(starts)>limits['max_attempts'] or len(closed)>=limits['max_completions'] or used_in+starts[-1]['reserved_input']>limits['max_input_tokens'] or used_out+starts[-1]['reserved_output']>limits['max_output_tokens']:
                raise BudgetExceeded('pilot_stage_ceiling_before_dispatch')
        while time.monotonic()<next_time:time.sleep(min(20.,next_time-time.monotonic()))
        started=time.monotonic();status,headers,raw=send(request,attempt_id)
        delay=2.
        if status==429:
            try:delay=max(15.,float(next((v for k,v in headers.items() if k.lower()=='retry-after'),15)))
            except (ValueError,TypeError):delay=15.
        next_time=time.monotonic()+delay
        event={'attempt_id':attempt_id,'status':status,'finish_reason':((raw.get('choices') or [{}])[0].get('finish_reason')) if isinstance(raw,dict) else None,
               'latency_seconds':time.monotonic()-started,'usage':raw.get('usage') if isinstance(raw,dict) else None}
        observations.append(event);print(canonical({'response':event}),flush=True)
        return status,headers,raw
    backend=DurableBackend(ROOT/'inference',config,transport,tokenizer);holder['backend']=backend
    rows=[]
    try:
        for index,public in enumerate(publics):
            phase='pilot' if index<20 else 'expansion'
            if index==20:
                gate_path=ROOT/'gate_decision.json'
                decision=json.loads(gate_path.read_text()) if gate_path.exists() else gate(rows,physical_audit(backend))
                if not gate_path.exists():atomic_json(gate_path,decision)
                print(canonical({'GATE':decision}),flush=True)
                if not decision['pass']:break
                stage['name']='expansion'
            path=ROOT/'results'/(public.key+'.json')
            if path.exists():
                saved=json.loads(path.read_text());assert saved['manifest_sha256']==file_sha(ROOT/'manifest.json') and digest(saved['result'])==saved['sha256'];row=saved['result']
            else:
                atoms={a.atom_id:a for source in public.sources for a in source.atoms}
                shards=[tuple(atoms[k] for k in shard) for shard in assign[public.key]]
                row=evaluate_row(public,shards,targets[public.key],backend,config,tokenizer,phase)
                atomic_json(path,{'result':row,'sha256':digest(row),'manifest_sha256':file_sha(ROOT/'manifest.json')})
            rows.append(row)
            print(canonical({'COMPLETE_QUESTION':len(rows),'phase':phase,'key':public.key,'complementary_k':len(row['router_choices']['complementary'])}),flush=True)
            write_report(rows,backend,manifest)
        result=write_report(rows,backend,manifest)
        atomic_json(ROOT/'execution_status.json',{'status':result['status'],'completed':len(rows),'gate_pass':result['gate']['pass']})
        print(canonical({'FINAL':result}),flush=True)
    except Exception as error:
        result=write_report(rows,backend,manifest,type(error).__name__)
        atomic_json(ROOT/'execution_status.json',{'status':result['status'],'completed':len(rows),'error_type':type(error).__name__})
        raise


if __name__=='__main__':main()
