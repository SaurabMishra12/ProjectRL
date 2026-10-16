"""Offline request reconstruction, official rescoring and scientific figures."""
import argparse
from collections import Counter
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import shutil

from WEBExtension.web_reasoning.backend import Request, DurableBackend
from WEBExtension.web_reasoning.config import validate_config
from WEBExtension.web_reasoning.evaluators import load_reference
from WEBExtension.web_reasoning.preparation import file_sha, make_tokenizer
from WEBExtension.web_reasoning.serialization import public_from_dict
from WEBExtension.web_reasoning.storage import digest, canonical, atomic_json
from WEBExtension.web_reasoning.accounting import valid_usage,cost_summary
from WEBExtension.causal_study.readout_reassessment.analysis import paired

driver=importlib.import_module('WEBExtension.pilots.2wiki_150_20261017.amendment')
core=driver.core
RUN=driver.ROOT
OUT=Path(__file__).resolve().parent


def verify_row(row, public, shards, target, config, tokenizer, inference):
    """Reconstruct all arm requests without calling or changing the backend."""
    model={'model':config.model,'deployment_revision':config.deployment_revision,
           'provider':'azure_v1','endpoint':config.provider_base_url,'replicate_id':config.replicate_id}
    decoding={'dataset':'2wiki','prompt_variant':'enhanced','temperature':0.,'max_tokens':config.max_output_per_call}
    checked=0
    def state(atoms,previous=None,incoming=()):
        request=Request(core.messages(public.question,atoms,previous,incoming),model,tokenizer.identity,decoding)
        saved=json.loads((Path(inference)/'responses'/(request.key+'.json')).read_text())
        assert saved['sha256']==digest({k:v for k,v in saved.items() if k!='sha256'})
        stored=json.loads((Path(inference)/'requests'/(request.key+'.json')).read_text())
        assert stored==json.loads(json.dumps(asdict(request)))
        assert digest(stored)==saved['request_sha256']==saved['key']==request.key
        return {**core.parse(saved['raw']),'call_key':request.key,'usage':saved['raw'].get('usage') or {}}
    initial=[state(shard) for shard in shards]
    assert row['initial']==initial
    packets=[core.packet(s,i,tokenizer,config.packet_cap) for i,s in enumerate(initial)]
    assert packets==row['packets']
    titles=[list(dict.fromkeys(a.title for a in shard)) for shard in shards]
    choices={kind:core.choose(kind,initial,titles,public.question,config.seed) for kind in ('random','heuristic','complementary')}
    assert {k:list(v) for k,v in choices.items()}=={k:list(v) for k,v in row['router_choices'].items()}
    metadata=[canonical({'agent':i,'titles':titles[i],'missing_entities':s['missing_entities'],
                         'answer_available':bool(s['answer'])}) for i,s in enumerate(initial)]
    evaluator=load_reference('2wiki'); golds=[target['answer'],*target['aliases']]
    assert set(row['arms'])==set(core.ARMS)
    for arm in core.ARMS:
        value=row['arms'][arm]
        if arm=='full_union':
            states=[state(tuple(a for shard in shards for a in shard))]; edges=[]; calls=states
        else:
            states=list(initial)
            if arm=='instar': edges=[(0,1),(2,1)]
            elif arm=='dense': edges=[(s,t) for t in range(3) for s in range(3) if s!=t]
            else: edges=[(s,1) for s in choices.get(arm,())]
            recipients=sorted({t for _,t in edges})
            for recipient in recipients:
                states[recipient]=state(shards[recipient],initial[recipient],
                                        [packets[s] for s,t in edges if t==recipient])
            calls=initial+[states[t] for t in recipients]
        assert value['states']==states
        assert value['edges']==[list(e) for e in edges]
        assert value['logical_call_keys']==[s['call_key'] for s in calls]
        sink=states[0] if arm=='full_union' else states[1]
        assert value['hub_request_key']==sink['call_key']
        predictions={'sink1':sink.get('answer') if sink['kind']=='valid' else None,
                     'abstention_aware':core.vote(states)}
        for rule,answer in predictions.items():
            actual=value['readouts'][rule];answer=answer or 'noanswer'
            scores=[evaluator.eval_answer(answer,g) for g in golds]
            assert actual['answer']==answer
            assert abs(actual['em']-100*max(x[0] for x in scores))<1e-9
            assert abs(actual['f1']-100*max(x[1] for x in scores))<1e-9
            checked+=1
        assert value['known_input_tokens']==sum(s['usage'].get('prompt_tokens') or 0 for s in calls)
        assert value['known_output_tokens']==sum(s['usage'].get('completion_tokens') or 0 for s in calls)
        assert value['missing_usage_fields']==sum(s['usage'].get(k) is None for s in calls for k in ('prompt_tokens','completion_tokens'))
        assert value['payload_tokens']==sum(packets[s]['tokens'] for s,_ in edges)
        assert value['omitted_facts']==sum(packets[s]['omitted_facts'] for s,_ in edges)
        assert value['metadata_tokens']==(sum(tokenizer.count(x) for x in metadata) if arm in ('heuristic','complementary') else 0)
    assert row['arms']['instar']['hub_request_key']==row['arms']['dense']['hub_request_key']
    return {'readouts_checked':checked,'sink_identity':True}


def verify_attempts(inference,config,tokenizer,partial=False):
    """Replay every reservation and terminal classification from saved artifacts."""
    inference=Path(inference)
    events=[json.loads(l) for l in (inference/'calls.jsonl').read_text().splitlines()]
    starts={};closed=set();keys=set();kinds=Counter();rejections=0;used_in=used_out=0
    for event in events:
        aid=event['attempt_id'];kind=event['event']
        if kind=='started':
            assert aid not in starts and set(starts)==closed
            request=json.loads((inference/'requests'/(event['key']+'.json')).read_text())
            assert digest(request)==event['key']==event['request_sha256']
            reserve_in=tokenizer.count(canonical(request['messages']))+config.framing_allowance
            reserve_out=request['decoding']['max_tokens']
            assert event['reserved_input']==reserve_in and event['reserved_output']==reserve_out
            assert type(reserve_out) is int and 0<reserve_out<=config.max_output_per_call
            assert len(starts)+1<=config.max_attempts and sum(kinds.values())+1<=config.max_completions
            assert used_in+reserve_in<=config.max_input_tokens and used_out+reserve_out<=config.max_output_tokens
            starts[aid]=event
        elif kind=='transport_error':
            assert aid in starts and aid not in closed
            if not partial:raise AssertionError('unresolved_transport_error')
        else:
            assert kind in ('success','rejected') and aid in starts and aid not in closed
            saved=json.loads((inference/'attempts'/(aid+'.json')).read_text())
            start=starts[aid]
            assert saved['sha256']==digest({k:v for k,v in saved.items() if k!='sha256'})
            assert saved['attempt_id']==aid and saved['key']==event['key']==start['key']
            assert saved['request_sha256']==start['request_sha256']
            assert saved['status']==event['status'] and saved['raw'].get('usage')==event['usage']
            raw=saved['raw'];status=saved['status']
            if kind=='rejected':
                assert status==429 and isinstance(raw.get('error'),dict) and not raw.get('choices')
                rejections+=1
            else:
                assert status in (200,201) or raw.get('choices')
                cache=inference/'responses'/(saved['key']+'.json')
                # A live success can precede cache publication by a few instructions.
                if cache.exists(): assert json.loads(cache.read_text())==saved
                else: assert partial
                assert saved['key'] not in keys
                keys.add(saved['key']);kinds[core.parse(raw)['kind']]+=1
                usage=event.get('usage') or {}
                used_in+=usage['prompt_tokens'] if valid_usage(usage.get('prompt_tokens')) else start['reserved_input']
                used_out+=usage['completion_tokens'] if valid_usage(usage.get('completion_tokens')) else start['reserved_output']
            closed.add(aid)
    if not partial:
        assert set(starts)==closed
        assert {p.stem for p in (inference/'responses').glob('*.json')}==keys
    assert used_in<=config.max_input_tokens and used_out<=config.max_output_tokens
    return {'verified_attempts':len(starts),'verified_rejections':rejections,
            'verified_physical_completions':sum(kinds.values()),'physical_parse_kinds':dict(kinds),
            'conservative_budget_input':used_in,'conservative_budget_output':used_out,
            'dispatch_reservations_recomputed':len(starts),'in_flight_attempts':len(starts)-len(closed)}


def main(partial=False):
    manifest=driver.verify(); config=validate_config(manifest['config'])
    tokenizer=make_tokenizer(config,manifest['descriptor'])
    publics=[public_from_dict(p) for p in json.loads((RUN/'public_inputs.json').read_text())]
    targets=json.loads((RUN/'targets.json').read_text()); assign=json.loads((RUN/'assignments.json').read_text())
    rows=[];checks=0;predictions=[]
    for i,public in enumerate(publics):
        path=RUN/('imported_results' if i<20 else 'results')/(public.key+'.json')
        if not path.exists():
            if partial:break
            raise ValueError('all_150_sealed_records_required')
        row=driver.load_result(path,manifest['historical_manifest_sha256'] if i<20 else file_sha(RUN/'manifest.json'))
        assert row['question_key']==public.key
        atoms={a.atom_id:a for source in public.sources for a in source.atoms}
        shards=[tuple(atoms[k] for k in shard) for shard in assign[public.key]]
        checks+=verify_row(row,public,shards,targets[public.key],config,tokenizer,RUN/'inference')['readouts_checked']
        rows.append(row)
        predictions.append({'question_key':public.key,'question':public.question,'gold':targets[public.key]['answer'],
            'cohort':'pilot' if i<20 else 'additional','k':len(row['router_choices']['complementary']),
            'readouts':{a:v['readouts'] for a,v in row['arms'].items()}})
    assert len({r['question_key'] for r in rows})==len(rows)
    # The original attempt and reference ledgers must be byte-identical prefixes.
    for name in ('calls.jsonl','logical_refs.jsonl'):
        assert (RUN/'inference'/name).read_bytes().startswith((driver.ORIGIN/'inference'/name).read_bytes())
    attempt_audit=verify_attempts(RUN/'inference',config,tokenizer,partial)
    kinds=attempt_audit['physical_parse_kinds']
    for key in manifest['cache_import']:
        if key.startswith(('responses/','requests/','attempts/')):
            assert file_sha(RUN/'inference'/key)==manifest['cache_import'][key]
    # An independently recomputed historical gate remains unchanged.
    origin_events=[json.loads(l) for l in (driver.ORIGIN/'inference/calls.jsonl').read_text().splitlines()]
    origin_refs=[json.loads(l) for l in (driver.ORIGIN/'inference/logical_refs.jsonl').read_text().splitlines()]
    pilot_usage=cost_summary(origin_events,origin_refs)
    pilot_usage['contract_failures']=sum(core.parse(json.loads((RUN/'inference'/name).read_text())['raw'])['kind']!='valid'
        for name in manifest['cache_import'] if name.startswith('responses/'))
    assert core.gate(rows[:20],pilot_usage)==manifest['historical_gate']
    audit={'status':'VERIFIED_PARTIAL_OFFLINE' if partial else 'VERIFIED_150_OFFLINE',
        'sealed_questions':len(rows),'readouts_checked':checks,**attempt_audit,
        'historical_gate_unchanged':True,'origin_hashes_verified':len(manifest['origin_files']),
        'sink_request_identities_verified':len(rows),'manifest_sha256':file_sha(RUN/'manifest.json'),
        'analysis_sha256':file_sha(__file__)}
    if partial:
        atomic_json(RUN/'partial_offline_audit.json',audit)
        print(canonical(audit));return
    backend=DurableBackend(RUN/'inference',config,None,tokenizer)
    summary=driver.report(rows,backend,manifest,'COMPLETE_150',final=True)
    for cohort,group in [('pilot',rows[:20]),('additional',rows[20:]),('pooled',rows)]:
        for rule in core.READOUTS:
            summary['contrasts'][cohort+':instar-random:'+rule]={metric:paired(
                {r['question_key']:r['arms']['instar']['readouts'][rule][metric] for r in group},
                {r['question_key']:r['arms']['random']['readouts'][rule][metric] for r in group},binary=metric=='em') for metric in ('em','f1')}
    summary['contrasts_note']='All paired comparisons exploratory; intervals are marginal and p-values unadjusted. In-Star/Dense Sink-1 equality follows request identity.'
    failures=[{'question_key':r['question_key'],'question':r['question'],'gold':targets[r['question_key']]['answer'],
               'needs':r['initial'][1]['missing_entities'],'choices':r['router_choices'],
               'sink_scores':{a:r['arms'][a]['readouts']['sink1'] for a in core.ARMS}}
              for r in rows if any(r['arms']['complementary']['readouts']['sink1']['em']<r['arms'][a]['readouts']['sink1']['em'] for a in ('heuristic','dense'))]
    for name,value in [('summary.json',summary),('audit.json',audit),('predictions.json',predictions),('failure_cases.json',failures)]:
        atomic_json(OUT/name,value)
    for name in ('manifest.json','authorization.json'):shutil.copyfile(RUN/name,OUT/name)
    plot(summary)
    lines=['# Measured 150-question 2Wiki evaluation','',
           '150 paired questions completed: 20 reused pilot questions and 130 additional questions. All seven arms and both readouts are retained.',
           '', '**The original 20-question gate failed.** The user subsequently authorized expansion to 150. No prompt, parser, router, assignment of the first 100, or model setting changed.',
           '', '![Additional 130 outcomes and logical token costs](additional_results.png)','',
           'The new 130 provide the primary extension results. Pooled 150 results and pilot results are separately available in summary.json. These publisher training-split results and all paired contrasts are exploratory.',
           '', '## Additional 130 questions','']
    for arm in core.ARMS:
        s=summary['stats']['additional'][arm]; aa=s['readouts']['abstention_aware']; sink=s['readouts']['sink1']
        lines.append(f'- {arm}: abstention-aware {aa["correct"]}/130 correct, F1 {aa["f1"]:.2f}; Sink-1 {sink["correct"]}/130 correct, F1 {sink["f1"]:.2f}.')
    lines += ['','## Audited scope','',
        f'- Reconstructed all arm requests and officially rescored {checks} readouts from saved responses; 150/150 In-Star and Dense Sink-1 requests match byte for byte.',
        '- Sink equality is guaranteed by the one-round incoming message identity and physical response reuse. It is not an independent noninferiority result.',
        f'- Physical parse outcomes: {dict(kinds)}. Refusal, malformed and unfinished outputs remain operational failures; explicit abstentions remain in all assigned denominators.',
        '- Per-arm logical token references overlap across physically reused requests. Payload, router metadata, and provider token usage are separately reported. Currency cost is unavailable.',
        '- This study evaluates deterministic routing and fixed-sink synthesis, not newly trained SparseComm controller weights. It does not establish asymptotic scaling or superiority of learned routing.',
        '- The relaxed common parser removes source-visibility and list-size schema rejection by design; it does not verify every generated fact.',
        '- Questions have gold-complete evidence in the union. Whole-article partitioning is independent of gold labels. Gold completeness does not establish that every example semantically requires communication.',
        '- Shared training provenance and previous pilot inspection limit confirmatory claims. No performance-based early stopping or post-outcome prompt tuning was used.',
        '', '[Full metrics and exploratory paired intervals](summary.json) · [All predictions](predictions.json) · [Offline audit](audit.json) · [Failure cases](failure_cases.json) · [Authorization amendment](authorization.json)']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(canonical(audit))


def plot(summary):
    os.environ['MPLCONFIGDIR']=str((OUT/'.mplcache').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    arms=list(core.ARMS); y=np.arange(len(arms));ss=summary['stats']['additional']
    labels=['Full union','No communication','Random k=1','Metadata k=1','Complementary k=0–2','Complete In-Star','Dense broadcast']
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for j,rule in enumerate(core.READOUTS):
        values=[ss[a]['readouts'][rule]['em'] for a in arms];pos=y+(j-.5)*.31
        axes[0].barh(pos,values,height=.29,color=('#237c72','#243b53')[j],label=('Abstention-aware','Fixed Sink 1')[j])
        for i,a in enumerate(arms):axes[0].text(values[i]+.5,pos[i],str(ss[a]['readouts'][rule]['correct'])+'/130',va='center',fontsize=8)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set(xlim=(0,105),xlabel='Operational answer EM (%)',ylabel='Evaluation arm',title='Additional 130: answer accuracy')
    axes[0].legend(fontsize=8,loc='lower right')
    inp=[ss[a]['known_input_tokens']/130/1000 for a in arms];out=[ss[a]['known_output_tokens']/130/1000 for a in arms]
    axes[1].barh(y,inp,color='#a8b4bf',label='Input');axes[1].barh(y,out,left=inp,color='#243b53',label='Output')
    axes[1].set(xlabel='Logical model tokens per question (thousands)',title='Additional 130: standalone token references')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('2Wiki · unchanged relaxed common prompt · N=3 partitioned agents',fontsize=12)
    fig.text(.01,.015,'Source: frozen 2wiki_150_20261017 saved responses. Exploratory training split; logical costs overlap through physical reuse.',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94))
    for ext in ('png','svg'):fig.savefig(OUT/('additional_results.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args();main(args.partial)
