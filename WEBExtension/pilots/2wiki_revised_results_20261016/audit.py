"""Independent offline audit and presentation of the frozen revised 2Wiki run."""
from collections import Counter
import gzip
import importlib
import json
import os
from pathlib import Path
import shutil
from WEBExtension.web_reasoning.storage import digest,atomic_json
from WEBExtension.web_reasoning.preparation import file_sha
from WEBExtension.web_reasoning.accounting import cost_summary
from WEBExtension.web_reasoning.evaluators import load_reference

OUT=Path(__file__).resolve().parent
RUN=Path('WEBExtension/web_reasoning/artifacts/2wiki_revised_20261016')


def main():
    driver=importlib.import_module('WEBExtension.pilots.2wiki_revised_20261016.run')
    core=importlib.import_module('WEBExtension.pilots.2wiki_revised_20261016.core')
    manifest=driver.verify();targets=json.loads((RUN/'targets.json').read_text());official=load_reference('2wiki')
    snapshot=json.loads((OUT/'preservation_snapshot.json').read_text());assert file_sha(snapshot['old_strict_ledger'])==snapshot['sha256']
    records=[]
    for p in (RUN/'results').glob('*.json'):
        saved=json.loads(p.read_text());assert saved['sha256']==digest(saved['result']) and saved['manifest_sha256']==file_sha(RUN/'manifest.json')
        records.append(saved['result'])
    order=manifest['pilot_keys']+manifest['expansion_keys'];records.sort(key=lambda r:order.index(r['question_key']))
    pilot=[r for r in records if r['phase']=='pilot'];assert len(pilot)==20
    assert [r['question_key'] for r in pilot]==manifest['pilot_keys']
    events=[json.loads(l) for l in (RUN/'inference/calls.jsonl').read_text().splitlines()]
    refs=[json.loads(l) for l in (RUN/'inference/logical_refs.jsonl').read_text().splitlines()]
    pilot_keys={r['key'] for r in refs if r['phase']=='pilot'}
    pilot_aids={e['attempt_id'] for e in events if e.get('key') in pilot_keys}
    pe=[e for e in events if e.get('attempt_id') in pilot_aids];pilot_usage=cost_summary(pe,[r for r in refs if r['phase']=='pilot'])
    counts=Counter();reasons=Counter();physical_keys=set();missing_usage=[]
    for path in (RUN/'inference/responses').glob('*.json'):
        saved=json.loads(path.read_text());assert saved['sha256']==digest({k:v for k,v in saved.items() if k!='sha256'})
        key=saved['key'];physical_keys.add(key)
        request=json.loads((RUN/'inference/requests'/(key+'.json')).read_text());assert digest(request)==saved['request_sha256']==key
        assert all('api_key' not in k.lower() for k in request)
        outcome=core.parse(saved['raw']);counts[outcome['kind']]+=1
        if key in pilot_keys and outcome['kind']!='valid':reasons[outcome['reason']]+=1
        usage=saved['raw'].get('usage') or {}
        if any(type(usage.get(k)) is not int or usage[k]<0 for k in ('prompt_tokens','completion_tokens')):missing_usage.append(key)
    pilot_usage['contract_failures']=sum(reasons.values())
    recomputed=core.gate(pilot,pilot_usage);saved_gate=json.loads((RUN/'gate_decision.json').read_text())
    assert saved_gate==recomputed
    if not saved_gate['pass']:assert len(records)==20 and not any(r['phase']=='expansion' for r in refs)
    checked=0;predictions=[];failure_cases=[]
    for row in records:
        qid=row['question_key'];golds=[targets[qid]['answer'],*targets[qid]['aliases']]
        assert row['arms']['instar']['hub_request_key']==row['arms']['dense']['hub_request_key']
        pred={'question_key':qid,'question':row['question'],'gold':targets[qid]['answer'],'phase':row['phase'],'k':len(row['router_choices']['complementary']),'arms':{}}
        for arm,value in row['arms'].items():
            states=value['states'];sink=states[0] if arm=='full_union' else states[1]
            expected={'sink1':sink.get('answer') if sink['kind']=='valid' else None,'abstention_aware':core.vote(states)}
            for rule,answer in expected.items():
                score=value['readouts'][rule];assert score['answer']==(answer or 'noanswer')
                scores=[official.eval_answer(answer or 'noanswer',g) for g in golds]
                assert abs(score['em']-100*max(v[0] for v in scores))<1e-8
                assert abs(score['f1']-100*max(v[1] for v in scores))<1e-8;checked+=1
            assert set(value['logical_call_keys'])<=physical_keys
            pred['arms'][arm]={'readouts':value['readouts'],'sink_kind':sink['kind'],'sink_reason':sink['reason']}
        predictions.append(pred)
        comp=row['arms']['complementary']['readouts']['sink1']
        heuristic=row['arms']['heuristic']['readouts']['sink1'];dense=row['arms']['dense']['readouts']['sink1']
        if comp['em']<heuristic['em'] or comp['em']<dense['em']:
            failure_cases.append({'question_key':qid,'question':row['question'],'gold':targets[qid]['answer'],
                'sink_missing_entities':row['initial'][1]['missing_entities'],'router_choices':row['router_choices'],
                'sink_results':{'complementary':comp,'heuristic':heuristic,'dense':dense},
                'initial_packets':row['packets']})
    summary=json.loads((RUN/'summary.json').read_text())
    assert len(records)==summary['completed_questions']
    assert summary['gate']==saved_gate
    audit={'status':'VERIFIED_OFFLINE','sealed_questions':len(records),'physical_parse_kinds':dict(counts),'pilot_usage':pilot_usage,
           'pilot_gate_recomputed_identically':True,'official_readout_scores_checked':checked,'old_ledger_unchanged':True,
           'missing_or_invalid_usage_keys':missing_usage,
           'review_notes':['Independent source review found no immediate gate bypass.',
                'Frozen pilot transport input ceiling counts missing usage as zero; global backend reserves conservatively. This conditional weakness was audited against actual usage. No missing or invalid usage means it was not exercised; fix in a future protocol rather than edit a running frozen implementation.',
                'Reporting paired-statistics dependency is absent from this new source map; its hash matches earlier committed Hotpot readout provenance. It does not affect gate or requests.'],
           'analysis_sha256':file_sha(__file__),'manifest_sha256':file_sha(RUN/'manifest.json')}
    assert not missing_usage,'Missing usage requires explicit spend-ceiling assessment; do not claim budget verification.'
    for name in ('max_completions','max_attempts'):
        metric='physical_completions' if name=='max_completions' else 'physical_attempts'
        assert pilot_usage[metric]<=manifest['pilot_ceiling'][name]
    assert pilot_usage['known_input_tokens_lower_bound']<=manifest['pilot_ceiling']['max_input_tokens']
    assert pilot_usage['known_output_tokens_lower_bound']<=manifest['pilot_ceiling']['max_output_tokens']
    from WEBExtension.causal_study.readout_reassessment import analysis as statistics
    previous=json.loads(gzip.decompress(Path('WEBExtension/pilots/hotpot_readout_20261016/results/provenance.json.gz').read_bytes()))
    assert file_sha(statistics.__file__)==previous['inputs_sha256'][str(Path(statistics.__file__).resolve())]
    audit['paired_statistics_sha256']=file_sha(statistics.__file__)
    atomic_json(OUT/'audit.json',audit);atomic_json(OUT/'predictions.json',predictions);atomic_json(OUT/'failure_cases.json',failure_cases)
    for name in ('summary.json','manifest.json','gate_decision.json'):shutil.copy2(RUN/name,OUT/name)
    os.environ['MPLCONFIGDIR']=str((OUT/'.mplcache').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    ss=summary['stats']['pilot'];arms=list(core.ARMS);y=np.arange(len(arms))
    labels=['Full union','No communication','Random, k=1','Metadata heuristic, k=1','Complementary, k=0..2','Complete In-Star','Dense broadcast']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for j,rule in enumerate(core.READOUTS):
        values=[ss[a]['readouts'][rule]['em'] for a in arms];pos=y+(j-.5)*.3
        axes[0].barh(pos,values,height=.28,color=('#237c72','#243b53')[j],label=('Abstention-aware','Fixed Sink 1')[j])
        for i,v in enumerate(values):axes[0].text(v+.8,pos[i],f'{int(round(v/5))}/20',va='center',fontsize=8)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set(xlim=(0,110),xlabel='Operational answer EM (%)',title='Frozen 20-question gate');axes[0].legend(loc='lower right',fontsize=8)
    inp=[ss[a]['known_input_tokens']/20/1000 for a in arms];out=[ss[a]['known_output_tokens']/20/1000 for a in arms]
    axes[1].barh(y,inp,color='#a8b4bf',label='Input');axes[1].barh(y,out,left=inp,color='#243b53',label='Output')
    axes[1].set(xlabel='Logical model tokens per question (thousands)',title='Known standalone token references');axes[1].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Revised 2Wiki pilot · relaxed common contract · N=3 partitioned agents',fontsize=12)
    fig.text(.01,.01,'Training-split exploratory pilot. Shared requests are physically reused; logical arm costs overlap. In-Star/Dense share the same sink request.',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.93))
    for ext in ('png','svg'):
        p=OUT/('pilot_results.'+ext);fig.savefig(p,dpi=180,bbox_inches='tight')
        if ext=='svg':p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n')
    plt.close(fig)
    lines=['# Revised 2Wiki: measured pilot decision','',
           '**Gate '+('PASSED' if saved_gate['pass'] else 'FAILED')+'**. '+str(len(records))+' complete question records; all assigned pilot questions retained.','',
           '**Outcome:** all 171 physical responses passed the relaxed contract (0% failures). NoComm met the <=15% ceiling under both readouts. In-Star and complementary exceeded NoComm and random routing, but did not beat the metadata one-sender control. Complementary also fell below Dense on Sink 1. The remaining 80 questions were not run.','',
           'Reasons: '+canonical_reasons(saved_gate['reasons'])+'.','',
           '![Observed pilot outcomes and token references](pilot_results.png)','',
           '| Arm | Abstention-aware correct / 20 | AA F1 | Sink-1 correct / 20 | Sink-1 F1 | Known model token references |',
           '|---|---:|---:|---:|---:|---:|']
    for arm,label in zip(arms,labels):
        v=ss[arm];aa=v['readouts']['abstention_aware'];sink=v['readouts']['sink1']
        lines.append(f'| {label} | {aa["correct"]}/20 | {aa["f1"]:.3f} | {sink["correct"]}/20 | {sink["f1"]:.3f} | {v["known_input_tokens"]+v["known_output_tokens"]:,} |')
    lines+=['','The gate uses both readouts and all frozen checks; no threshold or sample was changed after inference. '+('The frozen remaining 80 were run without tuning.' if len(records)==100 else 'No expansion was run because the gate did not pass.'),'',
            'Unique physical response parse kinds: '+json.dumps(dict(counts))+'. Contract failure fraction: '+str(saved_gate['physical_contract_failure_rate'])+'. All observed usage fields were present and valid. The original strict-schema ledger is unchanged.','',
            'Complementary sender-count distribution: '+json.dumps(summary['router_k_distribution'])+'.','',
            '## Interpretation','',
            '- Under matched abstention-aware voting, In-Star improves over NoComm by 45 EM percentage points (paired 95% bootstrap interval [25,65], unadjusted exact McNemar p=0.00390625). Complementary improves by 40 pp ([20,60], p=0.0078125). These are exploratory contrasts on previously inspected questions.',
            '- The metadata one-sender control is strong: 12/20 correct under abstention-aware voting and 11/20 at Sink 1. Thus the revised results do not support the broad claim that one sender is inherently insufficient.',
            '- Complete In-Star has the same Sink-1 results as Dense using 113,781 versus 177,203 known logical model-token references, a 35.79% reduction. This fixed-sink equality is a request-identity property.',
            '- The relaxed parser removes source-locator and list-limit rejection by design. This is not a semantic validation of generated facts.',
            '- Full-union single-agent and unchanged-prompt controls remain visible. NoComm was not constrained or filtered to force low accuracy.',
            '- In-Star/Dense Sink-1 equality is structural cache reuse, not an independent equivalence test.',
            '- This evaluates a deterministic complementary router, not learned SparseComm weights. Prompts, parser, packet cap, output cap, and fixed-sink protocol differ from the older pilot; their individual effects are not isolated.',
            '- These are twenty previously inspected publisher training questions with the original assignments, not a new held-out test. Paired intervals in summary.json are exploratory.',
            '- A reported code-review ceiling weakness only arises with missing input usage. Actual usage was complete and independently checked against the pilot limits; future reusable execution should fix that fallback. Frozen executed sources were not changed.',
            '', '## Saved failure anatomy','',
            '- For the two high schools\' country comparison, complementary selected sender 0 and abstained; the heuristic selected sender 2 and answered correctly. Broad overlap of entity words is not a reliable certificate that a need is covered.',
            '- For the birthday of the director of Billy the Kid in Santa Fe, complementary selected sender 0 and abstained; Dense combined both packets and recovered the full date. A relation description can require a further bridge even when it matches a source title.',
            '- For the nationality of the Chelsea Girls composer, the two-sender condition (complementary and Dense) abstained while the heuristic\'s one-sender condition answered American. More incoming packets did not always help in these saved outputs; no semantic judge or further inference was used to attribute the cause.',
            '', '[Exact diagnostic inputs and results](failure_cases.json)',
            '', '[All predictions](predictions.json) · [Full metrics and paired intervals](summary.json) · [Independent audit](audit.json) · [Frozen manifest](manifest.json) · [Gate decision](gate_decision.json)',
            '', 'Raw saved requests, provider responses, ledger and complete records remain in `WEBExtension/web_reasoning/artifacts/2wiki_revised_20261016`. Reproduce this audit with `python -m WEBExtension.pilots.2wiki_revised_results_20261016.audit`.']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'gate':saved_gate,'physical':summary['physical_usage'],'audit':audit},indent=2))


def canonical_reasons(reasons):return '; '.join(reasons) if reasons else 'none'


if __name__=='__main__':main()
