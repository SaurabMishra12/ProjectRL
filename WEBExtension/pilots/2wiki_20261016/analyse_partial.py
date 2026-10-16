"""Read-only recovery analysis: keep unknown attempts unresolved and never replay them."""
from collections import Counter
import importlib
import json
import os
from pathlib import Path
from WEBExtension.web_reasoning.accounting import cost_summary
from WEBExtension.web_reasoning.preparation import file_sha,PACKAGE
from WEBExtension.web_reasoning.serialization import run_from_dict,public_from_dict,targets_from_dict
from WEBExtension.web_reasoning.reporting import write_report
from WEBExtension.web_reasoning.statistics import paired_summary
from WEBExtension.web_reasoning.storage import digest,atomic_json
from WEBExtension.web_reasoning.contracts import Atom
from WEBExtension.web_reasoning.parsing import parse_outcome

M=importlib.import_module('WEBExtension.pilots.2wiki_20261016.analyse_pilot')
ROOT=M.ROOT;RUN=M.RUN


def main():
    manifest=json.loads((RUN/'manifest.json').read_text());manifest_hash=file_sha(RUN/'manifest.json')
    assert manifest_hash==(RUN/'manifest.sha256').read_text().strip()
    for name,sha in manifest['frozen_files'].items():assert file_sha(RUN/name)==sha,name
    for name,sha in manifest['code_hashes'].items():assert file_sha(PACKAGE/name)==sha,name
    events=[json.loads(s) for s in (RUN/'inference/calls.jsonl').read_text().splitlines()]
    refs=[json.loads(s) for s in (RUN/'inference/logical_refs.jsonl').read_text().splitlines()]
    starts={e['attempt_id']:e for e in events if e['event']=='started'}
    terminals={e['attempt_id']:e for e in events if e['event'] in ('success','rejected')}
    unknown=[s for aid,s in starts.items() if aid not in terminals]
    assert len(unknown)==1
    parse_kinds=Counter();parse_errors=Counter();finish=Counter();list_limits=Counter()
    for aid,start in starts.items():
        request=json.loads((RUN/'inference/requests'/(start['key']+'.json')).read_text())
        assert digest(request)==start['request_sha256']==start['key']
        if aid not in terminals:continue
        if terminals[aid]['event']=='rejected':continue
        saved=json.loads((RUN/'inference/attempts'/(aid+'.json')).read_text())
        assert saved['sha256']==digest({k:v for k,v in saved.items() if k!='sha256'})
        assert saved['key']==start['key'] and saved['request_sha256']==start['request_sha256'] and saved['attempt_id']==aid
        cache=json.loads((RUN/'inference/responses'/(start['key']+'.json')).read_text());assert cache['sha256']==saved['sha256']
        payload=json.loads(request['messages'][-1]['content'])
        atoms=tuple(Atom(**{**a,'context':tuple(a.get('context',{}).items())}) for a in payload['visible_sources'])
        outcome=parse_outcome(saved['raw'],atoms,'2wiki');parse_kinds[outcome.kind]+=1
        for choice in saved['raw'].get('choices',[]):finish[choice.get('finish_reason')]+=1
        if outcome.kind=='invalid':
            parse_errors[outcome.reason]+=1
            if outcome.reason=='list_limit':
                try:
                    obj=json.loads(saved['raw']['choices'][0]['message']['content'])
                    for field,maximum in [('facts',4),('entity_tags',8),('needs',2),('derivation',3),('conflicts',2)]:
                        if isinstance(obj.get(field),list) and len(obj[field])>maximum:list_limits[field]+=1
                except (ValueError,KeyError,TypeError):pass
    publics={p['key']:public_from_dict(p) for p in json.loads((RUN/'pilot_inputs.json').read_text())}
    targets={k:targets_from_dict(t) for k,t in json.loads((RUN/'pilot_targets.json').read_text()).items()}
    runs=[]
    for p in (RUN/'results').glob('*.json'):
        saved=json.loads(p.read_text());assert saved['sha256']==digest(saved['run']) and saved['manifest_sha256']==manifest_hash
        runs.append(run_from_dict(saved['run']))
    physical=cost_summary(events,refs);physical.update(unknown_attempts=unknown,uncertainty='One request timed out. It may or may not have generated/billed tokens; no automatic retry or invented usage.')
    out=ROOT/'analysis_partial';out.mkdir(exist_ok=True)
    audit={'status':'FROZEN_HASHES_AND_SEALED_SUCCESSES_VERIFIED_READ_ONLY','backend_recovery_status':'BLOCKED_UNKNOWN_TRANSPORT; original namespace unchanged',
           'manifest_sha256':manifest_hash,'frozen_files':len(manifest['frozen_files']),'code_files':len(manifest['code_hashes']),
           'sealed_results':len(runs),'physical_usage':physical,'physical_parse_kinds':dict(parse_kinds),'physical_invalid_reasons':dict(parse_errors),
           'physical_list_limit_fields':dict(list_limits),'finish_reasons':dict(finish)}
    atomic_json(out/'recovery_audit.json',audit)
    report_path=write_report(runs,targets,RUN/'reports/partial_pilot',True,publics,audit,
                             expected_cells=[(key,3) for key in publics],cohort='pilot',dataset='2wiki',mode='REAL')
    report=json.loads(report_path.with_suffix('.json').read_text());records=report['records'];sealed=[r for r in records if r['status']=='SAVED']
    assert report['coverage']=={'expected':20,'saved':19,'missing':1}
    stats={}
    for name in M.NAMES:
        s=M.summary(sealed,name);old_n=s['n'];s['cost_question_records']=old_n;s['n']=20
        s['operational_em']*=old_n/20;s['operational_f1']*=old_n/20
        s['incomplete_endpoints']+=1
        s['official_em_identification_bounds']=[s['operational_em'],s['operational_em']+5*s['incomplete_endpoints']]
        for v in s['readout_diagnostics'].values():v['operational_em']*=old_n/20;v['incomplete']+=1
        stats[name]=s
    op=[{'question_key':r['question_key'],'mode':'REAL','arms':{name:{'em':r['arms'][name]['sink'].get('em') or 0,'f1':r['arms'][name]['sink'].get('f1') or 0} for name in M.NAMES}} for r in records]
    strict=[{'question_key':r['question_key'],'mode':'REAL','arms':{name:{'em':r['arms'][name]['sink'].get('em')} for name in M.NAMES}} for r in records]
    contrasts={}
    for a,b in [('selected','random'),('selected','nocomm'),('selected','private'),('selected','static'),('selected','dense'),('selected','instar'),('instar','dense'),('selected','full_union')]:
        contrasts[a+'_minus_'+b]={'operational_em':paired_summary(op,(a,b)),'operational_f1':paired_summary(op,(a,b),metric='f1'),'strict_em':paired_summary(strict,(a,b))}
    result={'status':'INCOMPLETE_EXPLORATORY_PILOT_FULL_RUN_NOT_JUSTIFIED','assigned_questions':20,'sealed_questions':19,'methods':stats,'contrasts':contrasts,
            'physical_usage':physical,'audit':audit,'manifest_sha256':manifest_hash,'analysis_sha256':file_sha(__file__),
            'decision':'Do not start the full study: selected routing ties random and is worse than the cheaper full-union control in observed operational success. The strict primary contrast remains incomplete with bounds including zero.',
            'operational_metric':'Correct among all 20 assigned questions, with invalid/unresolved outputs and the unsealed question counted as pipeline failures. This is not a complete official answer-accuracy estimate.',
            'cost_scope':'Known logical token/traffic totals cover 19 sealed questions; actual physical usage also includes completed work on the unfinished question. Unknown timeout usage is not fabricated.',
            'instar_dense_caveat':'In frozen one-round W1, In-Star and dense have byte-identical sink requests, reused physically. Equal sink outputs are structural, not independent empirical equivalence evidence.'}
    atomic_json(out/'summary.json',result)
    predictions=[]
    for record in records:
        key=record['question_key'];raw=next((r for r in runs if r.question_key==key),None)
        predictions.append({'question_key':key,'question':publics[key].question,'gold':targets[key].answer,'type':dict(targets[key].strata)['type'],
                            'result_status':record['status'],'arms':{name:{'kind':raw.arms[name].outcome.kind if raw else 'unresolved',
                             'answer':raw.arms[name].outcome.proposal.answer if raw and raw.arms[name].outcome.proposal else None,
                             'reason':raw.arms[name].outcome.reason if raw else 'question_not_sealed_after_timeout',**record['arms'][name]['sink']} for name in M.NAMES}})
    atomic_json(out/'predictions.json',predictions)
    os.environ['MPLCONFIGDIR']=str((out/'.mplcache').resolve());M.figures(stats,out)
    lines=['# 2Wiki live pilot: no full-run expansion','',result['decision'],'',
           '20 real publisher training questions were assigned. Nineteen question records completed. One initial-agent request on the final question timed out after 185 confirmed completed generations; the namespace remains blocked and that request was not retried.','',
           '## Measured outcomes','',
           '| Method | Correct / 20 assigned | Operational EM | Operational F1 | Unknown sink scores | Known LLM token references (19 sealed) | Known communication tokens (19 sealed) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in stats.items():lines.append(f'| {s["name"]} | {s["correct"]}/20 | {s["operational_em"]:.1f}% | {s["operational_f1"]:.2f}% | {s["incomplete_endpoints"]} | {s["input_token_reference_sum"]+s["output_token_reference_sum"]:,} | {s["payload_tokens"]+s["control_tokens"]:,} |')
    lines+=['',result['operational_metric'],'',result['cost_scope'],'','![Measured accuracy and costs](pilot_results.png)','',
            '## Paired inference','',
            'Contrasts use all 20 assigned questions. Operational failures are zero; strict incomplete-cell bounds are retained separately. Bootstrap intervals resample questions 10,000 times. Selected-minus-random is the primary routing contrast; other comparisons are exploratory.','']
    for label,c in contrasts.items():
        a=c['operational_em'];b=c['strict_em'];lines.append(f'- {label}: operational EM difference {a["effect"]:+.1f} pp; 95% CI [{a["ci95"][0]:.1f}, {a["ci95"][1]:.1f}]; exact McNemar p={a["p_value"]:.4g}; strict status {b["status"]}, strict bounds {b["bounds"]}.')
    lines+=['','## Readout checks','',
            '| Method | Sink operational EM | Abstention-aware vote operational EM | Original vote operational EM |',
            '|---|---:|---:|---:|']
    for name,s in stats.items():lines.append(f'| {s["name"]} | {s["operational_em"]:.1f}% | {s["readout_diagnostics"]["abstention_vote"]["operational_em"]:.1f}% | {s["readout_diagnostics"]["original_vote_diagnostic"]["operational_em"]:.1f}% |')
    lines+=['','## Failure and cost audit','',result['instar_dense_caveat'],'',
            'Physical response parses: '+json.dumps(dict(parse_kinds))+'; invalid reasons: '+json.dumps(dict(parse_errors))+'; exceeded-list fields: '+json.dumps(dict(list_limits))+'.','',
            'Provider finish reasons: '+json.dumps(dict(finish))+'. Content-filtered outputs remain refusals, with reported usage retained.','',
            'Actual known usage and the unresolved attempt are in [recovery_audit.json](recovery_audit.json). Source/input/code hashes and every saved successful response were verified read-only; live backend recovery remains intentionally blocked.','',
            '## Limits and next experiment','',
            '- This is an exploratory training-split pilot with one unfinished question, not a complete held-out benchmark result.',
            '- Duplicate-title records were excluded before sampling. The giant source-component constraint was amended before inference; exact provenance and both candidate samples remain preserved.',
            '- The selected router is a deterministic metadata heuristic. No learned checkpoint was trained or evaluated.',
            '- Common improved prompts do not isolate the effect of prompt changes. Valid citations do not prove semantic faithfulness.',
            '- The single-agent full-union control is both more accurate and cheaper in the observed results; do not omit it from the research story.',
            '- Fix schema-limit following and test evidence-coverage-driven routing in another small matched pilot before spending on a full run. Do not repair stored outputs and pretend their downstream prompts were rerun.',
            '', '[All assigned question outcomes](predictions.json) · [Strict adapter report](../representative_run/reports/partial_pilot/report.json) · [Sampling provenance](../source/representative_provenance.json)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'methods':stats,'contrasts':contrasts,'audit':audit},indent=2))


if __name__=='__main__':main()
