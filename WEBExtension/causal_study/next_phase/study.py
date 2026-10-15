"""Prepare or mock the next study. Paid inference is unconditionally forbidden.

Commands: prepare, dry-run, status, paid (the last always raises PermissionError).
No credential helper or real transport is imported or initialized by this module.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from WEBExtension.causal_study.provider import BudgetBackend,atomic_json,digest
from WEBExtension.causal_study.run import tokenizer
from .binding_design import make_case,make_manifest,messages,validate_manifest,ARMS
from .deep_transitions import load_raw

ROOT=Path(__file__).resolve().parent
SOURCES=['binding_design.py','study.py','planning.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preservation_check():
    baseline=json.loads((ROOT/'PRESERVATION_BASELINE.json').read_text())
    previous=ROOT.parent
    if sha(previous/'paid_pilot/ledger.jsonl')!=baseline['paid_ledger_sha256']:
        raise ValueError('Previous paid ledger changed; stop')
    if sha(previous/'artifact_manifest.json')!=baseline['previous_artifact_manifest_sha256']:
        raise ValueError('Previous artifact manifest changed; stop')
    return baseline


def cached_cases():
    data,gold,paths,_,_=load_raw()
    cache=paths['E5'].parent/'inference/responses'
    cases=[];labels={};provenance=[]
    for line,row in enumerate(data['E5'],1):
        responses=row['conditions']['P_NoComm']['responses']
        facts=[];inputs=[]
        for response in responses:
            path=cache/(response['call_key']+'.json')
            saved=json.loads(path.read_text())
            if saved['content_hash']!=digest({k:v for k,v in saved.items() if k!='content_hash'}):
                raise ValueError('Historical response cache checksum mismatch')
            if saved['key']!=response['call_key']:raise ValueError('Historical cache identity mismatch')
            payload=json.loads(saved['request']['messages'][1]['content'])
            if payload['previous_answer'] is not None or payload['peer_messages']:
                raise ValueError('Expected independent initial state')
            if payload['question']!=gold[row['id']]['question']:raise ValueError('Question mispairing')
            facts.append(payload['evidence']);inputs.append({'call_key':response['call_key'],
                'response_path':str(path),'response_sha256':sha(path),'request_sha256':digest(saved['request'])})
        case=make_case(row['id'],gold[row['id']]['question'],responses,facts)
        cases.append(case);labels[row['id']]={'answer':gold[row['id']]['answer']}
        provenance.append({'question_id':row['id'],'raw_path':str(paths['E5']),'raw_line':line,
                           'initial_states':inputs,'eligible':case['eligible'],'reason':case.get('reason')})
    return cases,labels,provenance


def verify_prepared():
    preservation_check();directory=ROOT/'prepared'
    manifest=json.loads((directory/'manifest.json').read_text());validate_manifest(manifest)
    files=json.loads((directory/'file_hashes.json').read_text())
    for path,expected in files.items():
        if sha(directory/path)!=expected:raise ValueError('Prepared artifact changed: '+path)
    for name,expected in json.loads((directory/'execution_source_hashes.json').read_text()).items():
        if sha(ROOT/name)!=expected:raise ValueError('Prepared execution source changed: '+name)
    if json.loads((directory/'config.json').read_text())['manifest_hash']!=manifest['manifest_sha256']:
        raise ValueError('Config refers to another manifest')
    return manifest


def prepare():
    preservation_check();directory=ROOT/'prepared'
    if (directory/'manifest.json').exists():return verify_prepared()
    tok=tokenizer();cases,labels,provenance=cached_cases()
    manifest=make_manifest(cases,n=12,token_count=tok.count)
    previous=json.loads((ROOT.parent/'frozen/config.json').read_text())
    config={k:previous[k] for k in ['provider','model','model_revision','base_url','api_version','temperature','context_limit']}
    config.update({'max_tokens':2048,'max_completions':24,'max_attempts':30,
                   'max_retries_per_request':2,'min_interval_seconds':0.,'max_estimated_tokens':2000000,
                   'mock_only':True,'manifest_hash':manifest['manifest_sha256'],'tokenizer_identity':tok.identity})
    atomic_json(directory/'config.json',config)
    atomic_json(directory/'private_labels.json',{qid:labels[qid] for qid in [q['question_id'] for q in manifest['questions']]})
    atomic_json(directory/'screening.json',provenance)
    atomic_json(directory/'tokenizer.json',tok.identity)
    atomic_json(directory/'execution_source_hashes.json',{name:sha(ROOT/name) for name in SOURCES})
    for name in SOURCES:
        atomic_json(directory/'source_snapshot'/(name+'.json'),{'source':(ROOT/name).read_text(),'sha256':sha(ROOT/name)})
    payloads=[]
    for q in manifest['questions']:
        for arm in ARMS:
            payloads.append({'question_id':q['question_id'],'arm_internal_only':arm,'messages':messages(q,arm),
                             'prompt_sha256':digest(messages(q,arm)), 'prompt_content_tokens':q['prompt_tokens'][arm]})
    atomic_json(directory/'model_payloads.json',payloads)
    atomic_json(directory/'manifest.json',manifest)
    names=['config.json','private_labels.json','screening.json','tokenizer.json','model_payloads.json','manifest.json']
    atomic_json(directory/'file_hashes.json',{name:sha(directory/name) for name in names})
    verify_prepared()
    return manifest


class MockTransport:
    def __call__(self,request,request_id):
        payload=json.loads(request['messages'][1]['content'])
        if set(payload)!={'question','candidates','sources'}:raise ValueError('Unexpected model payload fields')
        return 200,{'x-request-id':'MOCK_'+request_id},{'id':'MOCK_'+request_id,'model':'MOCK_SELECTOR_ONLY',
          'choices':[{'finish_reason':'stop','message':{'content':'{"choice":"C0"}'}}],
          'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110}}


class MockOnlyBackend:
    mock_only=True
    def __init__(self,directory,config):
        if config.get('mock_only') is not True:raise PermissionError('Paid inference forbidden in this phase')
        self.backend=BudgetBackend(directory,config,transport=MockTransport(),sleep=lambda _:None)
    def complete(self,payload,metadata):return self.backend.complete(payload,metadata)
    def summary(self):return self.backend.summary()


def execute_paid(*args,**kwargs):
    raise PermissionError('NO_PAID_INFERENCE_AUTHORIZED: remaining E6 calls are not authorization for a new study')


def parse_choice(saved):
    raw=saved['raw'];choices=raw.get('choices') or []
    if not choices or choices[0].get('finish_reason')!='stop':raise ValueError('Incomplete selector response; preserve and stop')
    obj=json.loads(choices[0]['message']['content'])
    if not isinstance(obj,dict) or set(obj)!={'choice'} or obj['choice'] not in ['C0','C1','C2']:
        raise ValueError('Selector must return exactly one allowed candidate ID')
    return obj['choice']


def execute(manifest,labels,backend,checkpoints):
    validate_manifest(manifest)
    if type(backend) is not MockOnlyBackend:raise PermissionError('Only the offline mock backend is allowed')
    checkpoints=Path(checkpoints);checkpoints.mkdir(parents=True,exist_ok=True)
    for case in manifest['questions']:
        path=checkpoints/(case['question_id']+'.json')
        if path.exists():
            saved=json.loads(path.read_text())
            if saved.get('content_hash')!=digest({k:v for k,v in saved.items() if k!='content_hash'}):raise ValueError('Checkpoint checksum mismatch')
            if saved.get('manifest_hash')!=manifest['manifest_sha256']:raise ValueError('Checkpoint manifest mismatch')
            continue
        record={'question_id':case['question_id'],'manifest_hash':manifest['manifest_sha256'],
                'execution_mode':'MOCK_ONLY','arms':{}}
        for arm in case['arm_order']:
            payload=messages(case,arm)
            result=backend.complete(payload,{'question_id':case['question_id'],'stage':'finalizer','mapping':arm})
            choice=parse_choice(result);position=int(choice[1]);node=case['candidate_order'][position]
            record['arms'][arm]={'choice':choice,'selected_node':node,'selected_frozen_answer':case['candidates'][node]['answer'],
                                 'request_hash':result['request_hash'],'cache_hit':result['cache_hit'],
                                 'prompt_sha256':digest(payload),'response_sha256':digest(result['raw'])}
        # Private labels are deliberately never sent to the provider or scored as real outcomes.
        record['content_hash']=digest(record);atomic_json(path,record)
    usage=backend.summary()
    return {'status':'COMPLETE_MOCK_ONLY','questions':len(manifest['questions']),
            'mock_http_attempts':usage['http_attempts'],'mock_completions':usage['successful_completions'],
            'real_provider_calls':0,'scored_scientific_effects':False,'paid_authorized':False,
            'manifest_hash':manifest['manifest_sha256']}


def dry_run():
    manifest=prepare();config=json.loads((ROOT/'prepared/config.json').read_text())
    backend=MockOnlyBackend(ROOT/'mock/budget',config)
    labels=json.loads((ROOT/'prepared/private_labels.json').read_text())
    first=execute(manifest,labels,backend,ROOT/'mock/questions')
    second=execute(manifest,labels,MockOnlyBackend(ROOT/'mock/budget',config),ROOT/'mock/questions')
    if first['mock_http_attempts']!=24 or second['mock_http_attempts']!=24:raise ValueError('Projected mock/resume count mismatch')
    status={**second,'resume_additional_requests':second['mock_http_attempts']-first['mock_http_attempts'],
            'preserved_paid_ledger_sha256':preservation_check()['paid_ledger_sha256']}
    atomic_json(ROOT/'mock/validation.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','dry-run','status','paid'])
    args=parser.parse_args()
    if args.action=='paid':execute_paid()
    elif args.action=='prepare':
        m=prepare();print(json.dumps({'status':'PREPARED_NOT_AUTHORIZED','questions':len(m['questions']),
              'eligible_cached':m['mechanically_eligible_cases'],'future_finalizer_calls':m['projected_finalizer_calls']},indent=2))
    elif args.action=='dry-run':print(json.dumps(dry_run(),indent=2))
    else:
        p=ROOT/'mock/validation.json';print(p.read_text() if p.exists() else '{"status":"NOT_RUN","paid_authorized":false}')


if __name__=='__main__':main()
