"""E3-v2: exploratory, gold-free, distinct evidence slices at equal token length.

Historical E3 and frozen scientific modules are never modified. New runs reuse
saved initial states, not new initial generations. No performance-directed tuning.
"""
from __future__ import annotations
import argparse, json, os, shutil, signal, sys, threading, time
from pathlib import Path
from protocol import (ROOT, canonical, digest, file_hash, stable_rng, ExactTokenizer,
                      evidence_lines, delivered_facts, score, paired_effect)
from analysis.durable_storage import durable_json, durable_bytes, Journal

SOURCE = ROOT/'artifacts/run_f6aa61bac2c3dfae'

def evidence_bank(state):
    # All sender-visible evidence, in frozen retrieval/document/sentence order.
    # Avoid making eligible content depend on successful model citation generation.
    return '\n'.join('Evidence: '+canonical(f) for f in evidence_lines(state['assignment'][0]))

def content_pair(text,cap,tokenizer,seed):
    if type(cap) is not int or cap<=0: raise ValueError('Positive integer cap required')
    ids=tokenizer.encode(text); target=min(cap,len(ids)//2)
    if target<1: raise ValueError('Source too short to support distinct slices')
    prefix=tokenizer.decode(ids[:target])
    if '\ufffd' in prefix or prefix not in text or tokenizer.count(prefix)!=target:
        raise ValueError('Invalid exact prefix; no padding or silent budget adjustment')
    starts=list(range(1,len(ids)-target+1))
    stable_rng('E3-v2-crop',seed,digest(text)).shuffle(starts)
    for start in starts:
        crop=tokenizer.decode(ids[start:start+target])
        if (crop!=prefix and '\ufffd' not in crop and crop in text
                and tokenizer.count(crop)==target):
            common={'tokens':target,'source_hash':digest(text),'source_tokens':len(ids)}
            return ({**common,'text':prefix,'start_token':0},
                    {**common,'text':crop,'start_token':start})
    raise ValueError('No distinct exact-length crop exists; preflight must stop')

def prepare(source=SOURCE):
    source=Path(source); old=json.loads((source/'frozen_manifest.json').read_text())
    cfg=old['plan']['config']
    for name,sha in old['plan']['sources'].items():
        if file_hash(ROOT/name)!=sha: raise ValueError('Historical frozen source changed: '+name)
    for name in ['E1','E3']:
        status=json.loads((source/(name+'_status.json')).read_text())
        if status['status']!='COMPLETE' or status['sha256']!=file_hash(source/(name+'.jsonl')):
            raise ValueError('Historical completion/hash gate failed: '+name)
    records=[json.loads(l) for l in (source/'E1.jsonl').read_text().splitlines()]
    inputs=json.loads((source/'sample_inputs.json').read_text())
    if [r['id'] for r in records]!=[r['id'] for r in inputs]: raise ValueError('Input order mismatch')
    tokenizer=ExactTokenizer(cfg['tokenizer'],cfg['tokenizer_revision'])
    oldtok=json.loads((source/'E3_status.json').read_text())['tokenizer']
    if tokenizer.identity!=oldtok: raise ValueError('Pinned tokenizer identity differs')
    payloads=[]
    for record in records:
        text=evidence_bank(record['states']['R'])
        if not text:
            payloads.append({'id':record['id'],'source_text':text,'eligible':False,'reason':'NO_SENDER_EVIDENCE_AVAILABLE'})
            continue
        a,b=content_pair(text,128,tokenizer,cfg['crop_seed'])
        if a['tokens']!=128: raise ValueError('Source cannot support fixed 128-token contrast')
        payloads.append({'id':record['id'],'source_text':text,'eligible':True,'prefix':a,'crop':b,
            'arm_order':stable_rng('E3-v2-arm-order',record['id'],23).sample(['prefix','crop'],2)})
    paths=['e3_corrected.py','protocol.py','experiments.py','backend.py','machine_connection.py',
           'analysis/durable_runtime.py','analysis/durable_storage.py','analysis/adaptive_pacing.py',
           'controller_study.py','learned_controller.py']
    plan={'version':'E3-v2-evidence-bank-distinct-slices-1',
          'status_scope':'EXPLORATORY_FOLLOWUP_AFTER_INSPECTION_OF_ORIGINAL_HELDOUT_RESULTS',
          'source_run':str(source.resolve()),'source_files':{n:file_hash(source/n) for n in
              ['frozen_manifest.json','E1.jsonl','E3.jsonl','sample_inputs.json','sample_gold.json']},
          'config':cfg,'tokenizer':tokenizer.identity,'sources':{p:file_hash(ROOT/p) for p in paths},
          'question_ids':[r['id'] for r in records],'payload_plan_sha256':digest(payloads),
          'sender':0,'receiver':1,'regime':'R','cap':128,'short_source_rule':'Require 128 tokens; empty evidence retained as ineligible, never fabricated',
          'primary':'mean receiver F1(prefix) minus receiver F1(distinct seeded crop)',
          'source':'ALL sender-visible Wikipedia sentences, frozen BM25 rank then sentence order',
          'gold_usage':'scoring/post-hoc support and strata only',
          'endpoint':'receiver 1; no ensemble dilution',
          'matching':'one directed edge, same sender/receiver/initial states, equal actual payload tokens, one receiver update per arm',
          'limitations':['R receiver already sees same evidence; estimates redundant-evidence presentation, not novel fact transfer',
                        'Contiguous slices may cut sentences/JSON; report complete-fact delivery rather than claim entailment',
                        'Equal payload tokens do not imply identical provider prompt or generated tokens',
                        'Same inspected held-out questions; follow-up is exploratory, no noninferiority claim'],
          'maximum_new_logical_completions':600,'no_new_initial_generations':True}
    directory=ROOT/'artifacts'/('e3_corrected_'+digest(plan)[:16]);directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'frozen_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['plan']!=plan: raise ValueError('Manifest changed')
    if not manifest.exists():
        from analysis.durable_runtime import utc
        durable_json(manifest,{'plan':plan,'utc':utc()})
        durable_json(directory/'payload_plan.json',payloads)
        for name in paths: durable_bytes(directory/'source_snapshot'/name,(ROOT/name).read_bytes())
    if digest(json.loads((directory/'payload_plan.json').read_text()))!=plan['payload_plan_sha256']:
        raise ValueError('Saved payload plan differs')
    eligible=[p for p in payloads if p['eligible']]
    preflight={'questions':len(payloads),'eligible_questions':len(eligible),
        'ineligible':[{k:p[k] for k in ['id','reason']} for p in payloads if not p['eligible']],
        'distinct_payload_pairs':sum(p['prefix']['text']!=p['crop']['text'] for p in eligible),
        'equal_actual_token_pairs':sum(p['prefix']['tokens']==p['crop']['tokens']==128 for p in eligible),
        'fixed_payload_tokens_per_arm':128,
        'prefix_complete_facts':sum(len(delivered_facts(p['prefix']['text'])) for p in eligible),
        'crop_complete_facts':sum(len(delivered_facts(p['crop']['text'])) for p in eligible),
        'run_dir':str(directory),'maximum_new_completions':2*len(eligible),'inference_calls_in_preflight':0}
    if preflight['questions']!=300 or preflight['distinct_payload_pairs']!=len(eligible) or preflight['equal_actual_token_pairs']!=len(eligible):
        raise ValueError('All-eligible-question intervention fidelity gate failed')
    durable_json(directory/'preflight.json',preflight)
    return directory,cfg,tokenizer,records,inputs,payloads

def receiver_condition(row,state,revised,deliveries):
    from experiments import condition,measured_cost
    out=condition(row,state,revised,deliveries)
    out['ensemble_answer']=out['answer'];out['answer']=revised[1]['answer'];out.update(score(out['answer'],row['answer']))
    gold=set(map(tuple,row['supporting_facts']));support=set(map(tuple,revised[1]['citations']))
    out['answer_support_recall']=len(gold&support)/max(1,len(gold))
    out['answer_support_precision']=len(gold&support)/max(1,len(support))
    out['invalid_citation_count']=len(revised[1]['invalid_citations'])
    out['provider_refusal_count']=int(bool(revised[1].get('provider_refusal')))
    out['cost']=measured_cost(state['responses']+[revised[1]])
    return out

def report(directory):
    import numpy as np
    path=directory/'E3.jsonl'
    rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    if len(rows)!=300 or len({r['id'] for r in rows})!=300: raise ValueError('Cannot finalize incomplete results')
    all_rows=rows
    rows=[r for r in rows if r['eligible']]
    expected=json.loads((directory/'payload_plan.json').read_text())
    if {r['id'] for r in all_rows}!={p['id'] for p in expected}: raise ValueError('Question identity gate failed')
    if {r['id'] for r in rows}!={p['id'] for p in expected if p['eligible']}: raise ValueError('Eligibility gate failed')
    summaries={}
    for label in ['R_prefix','R_crop','R_NoComm_receiver','R_SelfUpdate_receiver']:
        arms=[r['conditions'][label] for r in rows]
        summaries[label]={'n':len(arms),**{k:float(np.mean([a[k] for a in arms])) for k in
            ['f1','em','answer_support_recall','answer_support_precision','payload_tokens','invalid_citation_count','provider_refusal_count']}}
    contrasts={m:paired_effect({r['id']:r['conditions']['R_prefix'][m] for r in rows},
                               {r['id']:r['conditions']['R_crop'][m] for r in rows},seed=23)
               for m in ['f1','em','answer_support_recall']}
    diagnostics={'distinct_payload_pairs':sum(r['conditions']['R_prefix']['deliveries'][0]['text']!=r['conditions']['R_crop']['deliveries'][0]['text'] for r in rows),
        'equal_payload_token_pairs':sum(r['conditions']['R_prefix']['payload_tokens']==r['conditions']['R_crop']['payload_tokens'] for r in rows),
        'equal_receiver_calls':len(rows),'complete_fact_counts':{l:sum(len(r['conditions'][l]['deliveries'][0]['facts']) for r in rows) for l in ['R_prefix','R_crop']}}
    if diagnostics['distinct_payload_pairs']!=len(rows) or diagnostics['equal_payload_token_pairs']!=len(rows): raise ValueError('Completion fidelity failed')
    result={'status':'COMPLETE_EXPLORATORY','screened_questions':300,'eligible_paired_questions':len(rows),
            'ineligible_questions':[{'id':r['id'],'reason':r['reason']} for r in all_rows if not r['eligible']], 'summaries':summaries,'paired_prefix_minus_crop':contrasts,
            'primary':'f1 only; EM/support secondary descriptive, no separate confirmatory claims',
            'diagnostics':diagnostics,'result_sha256':file_hash(path)}
    durable_json(directory/'report.json',result)
    text='# Corrected E3-v2 results\n\nExploratory follow-up; original E3 preserved. No outcome was selected for paper support.\n\n'
    text+='```json\n'+json.dumps(result,indent=2)+'\n```\n\n'
    text+='Equal payload tokens and receiver calls isolate the presentation manipulation within this protocol. R recipients already have the same documents. A crop may cut structured evidence lines; this is not a semantic evidence-selector comparison or a novel-evidence transfer test. A nonsignificant contrast does not prove equivalence.\n'
    durable_bytes(directory/'REPORT.md',text.encode())
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4));labels=list(summaries)
    ax.bar(range(4),[summaries[l]['f1'] for l in labels]);ax.set_xticks(range(4),['Prefix','Distinct crop','NoComm receiver','SelfUpdate receiver'])
    ax.set_ylabel('Answer F1 (0–100)');ax.set_title('E3-v2: 299 eligible paired questions; exploratory');fig.tight_layout()
    fig.savefig(directory/'receiver_f1.png',dpi=160);fig.savefig(directory/'receiver_f1.pdf');plt.close(fig)
    return result

def execute(prepared):
    directory,cfg,tokenizer,records,inputs,payloads=prepared
    from filelock import FileLock
    from analysis.durable_runtime import Runtime,install,utc
    authorization={'attempts':[]}
    auth_pointer=directory/'retry_authorization_pointer.json'
    if auth_pointer.exists():
        pointer=json.loads(auth_pointer.read_text());auth_path=Path(pointer['path'])
        if auth_path.resolve().parent!=directory.resolve() or file_hash(auth_path)!=pointer['sha256']:
            raise ValueError('Retry authorization scope/checksum differs')
        authorization=json.loads(auth_path.read_text())
        ledger=[json.loads(l) for l in (directory/'calls.jsonl').read_text().splitlines()]
        starts={r['attempt_id']:r for r in ledger if r['event']=='started'}
        for entry in authorization['attempts']:
            old=starts[entry['attempt_id']]
            if entry['pipeline']!='e3_corrected' or Path(entry['run_dir']).resolve()!=directory.resolve():
                raise ValueError('Retry authorization belongs to another run')
            if any(entry[k]!=old[k] for k in ['key','request_sha256','utc']):
                raise ValueError('Retry request binding differs')
            if file_hash(old['request_file'])!=old['request_sha256']:
                raise ValueError('Original uncertain request changed')
            if sum(r['key']==old['key'] for r in starts.values())>=4 and not (directory/'responses'/(old['key']+'.json')).exists():
                raise ValueError('Three-replacement limit reached')
            if not any(r.get('event')=='uncertain_retry_authorized' and r.get('attempt_id')==entry['attempt_id'] for r in ledger):
                Journal(directory/'calls.jsonl').append({'event':'uncertain_retry_authorized','utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                    'attempt_id':entry['attempt_id'],'key':entry['key'],'authorization_file':str(auth_path),
                    'authorization_sha256':file_hash(auth_path),'original_outcome':'UNKNOWN_POTENTIALLY_BILLED','original_usage':None})
    runtime=Runtime(directory,authorization,'e3_corrected');runtime.phase='E3-v2'
    install(runtime)
    from backend import RealBackend
    from experiments import update
    from analysis.adaptive_pacing import AdaptivePacer
    pacer=AdaptivePacer(ROOT/'artifacts'/('azure_rate_'+digest([cfg['base_url'],cfg['model']])[:16]+'.json'))
    done={};path=directory/'E3.jsonl';journal=Journal(path)
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r['id'] in done: raise ValueError('Duplicate checkpoint')
            done[r['id']]=r
    if not set(done)<=set(r['id'] for r in records): raise ValueError('Foreign checkpoint')
    golds={r['id']:r for r in json.loads((SOURCE/'sample_gold.json').read_text())}
    def stop(signum,frame): runtime.stop_requested=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    thread=threading.Thread(target=runtime.monitor,daemon=True);thread.start();status='FAILED'
    try:
        with FileLock(str(directory/'inference.lock'),timeout=0):
            backend=RealBackend(cfg,tokenizer,directory)
            capture=backend.client.chat.completions.create
            def guidance(error):
                response=getattr(error,'response',None);headers=getattr(response,'headers',{}) or {}
                try: delay=max(1,float(headers.get('retry-after','60')))
                except (ValueError,TypeError): delay=60
                return delay,{}
            backend.client.chat.completions.create=lambda **params:pacer.call(capture,params,guidance,
                lambda data:durable_json(directory/'rate_limit_diagnostic.json',data))
            # No transparent timeouts/replacements: uncertain costs remain recorded.
            for record,row,p in zip(records,inputs,payloads):
                if record['id'] in done: continue
                if runtime.stop_requested: raise RuntimeError('Stop requested at saved question boundary')
                start=time.perf_counter();cpu=time.process_time();state=record['states']['R'];gold=golds[row['id']]
                out={'id':row['id'],'type':record['type'],'experiment':'E3-v2','conditions':{},'payload_plan_hash':digest(p),'eligible':p['eligible']}
                if not p['eligible']: out['reason']=p['reason']
                for mode in p.get('arm_order',[]):
                    revised,delivered=update(row,state,[(0,1)],[p[mode]],backend,'E3-v2_'+mode)
                    out['conditions']['R_'+mode]=receiver_condition(gold,state,revised,delivered)
                # Saved anchors, endpoint rescored; no fresh generation.
                base=receiver_condition(gold,state,state['responses'],[])
                base['cost']=__import__('experiments').measured_cost(state['responses'])
                out['conditions']['R_NoComm_receiver']=base
                out['conditions']['R_SelfUpdate_receiver']=receiver_condition(gold,state,record['conditions']['R_SelfUpdate']['responses'],[])
                out['question_wall_seconds']=time.perf_counter()-start;out['question_client_cpu_seconds']=time.process_time()-cpu
                journal.append(out);done[out['id']]=out
                durable_json(directory/'E3_status.json',{'status':'RUNNING','completed':len(done),'target':300,'utc':utc()})
                print('E3-v2:',len(done),'/300',flush=True)
            result=report(directory);status='COMPLETE'
            durable_json(directory/'E3_status.json',{'status':'COMPLETE','completed':300,'target':300,'sha256':file_hash(path),'utc':utc()})
            return result
    except BaseException as error:
        durable_json(directory/'E3_status.json',{'status':'STOPPED' if runtime.stop_requested else 'INCOMPLETE','completed':len(done),'target':300,'error_type':type(error).__name__,'utc':utc()})
        raise
    finally:
        runtime.close('STOPPED' if runtime.stop_requested else status);thread.join(timeout=5)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--report',type=Path);args=parser.parse_args()
    if args.report: print(json.dumps(report(args.report),indent=2));return
    prepared=prepare();print(json.dumps(json.loads((prepared[0]/'preflight.json').read_text()),indent=2),flush=True)
    if args.execute: execute(prepared)

if __name__=='__main__': main()
