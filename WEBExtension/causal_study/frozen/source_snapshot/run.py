"""Freeze, mock, execute and resume the single authorized controlled pilot.

Run from the repository root with ``python -m WEBExtension.causal_study.run``.
No action defaults to paid inference. Paid state always uses ONE fixed directory.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import time
from pathlib import Path

from filelock import FileLock
from .design import build_manifest, validate_manifest, initial_messages, revision_messages, majority, build_e7, e7_messages
from .provider import BudgetBackend, atomic_json, digest
from WEBExtension.protocol import load_official, score, ExactTokenizer

ROOT=Path(__file__).resolve().parent
WEB=ROOT.parent
STUDY_FILES=['design.py','provider.py','run.py','statistics.py']


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokenizer():
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    config=json.loads((WEB/'config.json').read_text())
    return ExactTokenizer(config['tokenizer'],config['tokenizer_revision'])


def exposure_rows():
    paths=set((WEB/'artifacts').glob('**/sample_gold.json'))
    paths.update((WEB/'artifacts').glob('controller_splits_*/*_gold.json'))
    paths={p for p in paths if 'source_snapshot' not in p.parts}
    rows=[]; sources=[]
    for p in sorted(paths):
        value=json.loads(p.read_text())
        if not isinstance(value,list): raise ValueError('Unexpected exposure artifact format: '+str(p))
        rows.extend(value)
        sources.append({'path':str(p.relative_to(WEB.parent)), 'sha256':sha256(p),'rows':len(value)})
    return rows,sources


def verify_frozen(directory,manifest,tok):
    if manifest.get('manifest_sha256')!=digest({k:v for k,v in manifest.items() if k!='manifest_sha256'}):
        raise ValueError('Frozen manifest checksum mismatch')
    validate_manifest(manifest)
    for filename,expected in json.loads((directory/'source_hashes.json').read_text()).items():
        if sha256(ROOT/filename)!=expected: raise ValueError('Frozen execution source changed: '+filename)
    if json.loads((directory/'tokenizer.json').read_text())!=tok.identity:
        raise ValueError('Frozen tokenizer identity changed')
    if json.loads((directory/'private_labels.json').read_text())!=manifest['private']:
        raise ValueError('Frozen private labels differ from manifest')
    if json.loads((directory/'question_inputs.json').read_text())!={'questions':manifest['questions']}:
        raise ValueError('Frozen public inputs differ from manifest')
    config=json.loads((directory/'config.json').read_text())
    if config.get('study_manifest_hash')!=manifest['manifest_sha256'] or config.get('tokenizer_identity')!=tok.identity:
        raise ValueError('Frozen config identity differs from manifest/tokenizer')
    return manifest


def freeze():
    directory=ROOT/'frozen'
    directory.mkdir(parents=True,exist_ok=True)
    tok=tokenizer()
    existing=directory/'manifest.json'
    if existing.exists():
        manifest=json.loads(existing.read_text())
        verify_frozen(directory,manifest,tok)
        return manifest,tok
    blocked,sources=exposure_rows()
    source={'dataset_path':str((WEB/'data/hotpot_dev_fullwiki_v1.json').relative_to(WEB.parent)),
            'dataset_sha256':sha256(WEB/'data/hotpot_dev_fullwiki_v1.json'),
            'known_exposure_sources':sources,'construction_time_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    manifest=build_manifest(load_official(),n=8,seed=20261015,blocked_rows=blocked,token_count=tok.count,source=source)
    validate_manifest(manifest)
    config=json.loads((WEB/'config.json').read_text())
    config={k:config[k] for k in ['provider','model','model_revision','base_url','api_version',
                                 'temperature','max_tokens','context_limit']}
    config.update({'max_completions':150,'max_attempts':180,'max_retries_per_request':2,
                   'min_interval_seconds':2., 'max_estimated_tokens':2000000,
                   'tokenizer_identity':tok.identity,'study_manifest_hash':manifest['manifest_sha256'],
                   'e7_paid_inference':False})
    for filename in STUDY_FILES:
        if not (ROOT/filename).is_file(): raise ValueError('Execution source missing: '+filename)
    hashes={f:sha256(ROOT/f) for f in STUDY_FILES}
    atomic_json(directory/'source_hashes.json',hashes)
    for f in STUDY_FILES:
        (directory/'source_snapshot').mkdir(exist_ok=True)
        shutil.copyfile(ROOT/f,directory/'source_snapshot'/f)
    atomic_json(directory/'config.json',config)
    atomic_json(directory/'tokenizer.json',tok.identity)
    atomic_json(directory/'private_labels.json',manifest['private'])
    atomic_json(directory/'question_inputs.json',{'questions':manifest['questions']})
    atomic_json(directory/'environment.json',{'python':platform.python_version(),
                'packages':{p:importlib.metadata.version(p) for p in ['httpx','filelock','numpy','transformers']}})
    # Manifest is written last: its presence commits the complete frozen design.
    atomic_json(existing,manifest)
    verify_frozen(directory,manifest,tok)
    return manifest,tok


def parse_result(saved,messages):
    raw=saved.get('raw') or {}
    if not isinstance(raw,dict): raise ValueError('Non-object provider response; raw retained')
    choices=raw.get('choices') or []
    if not choices: raise ValueError('No provider completion; raw retained')
    choice=choices[0]; message=choice.get('message') or {}
    refusal=choice.get('finish_reason')=='content_filter' or bool(message.get('refusal'))
    if refusal:
        parsed={'answer':'noanswer','citations':[],'invalid_citations':[],
                'status':'PROVIDER_ABSTENTION','provider_refusal':True}
    else:
        if choice.get('finish_reason')!='stop':
            raise ValueError('Truncated/incomplete completion; never scored as prediction or regenerated')
        text=message.get('content')
        if not isinstance(text,str): raise ValueError('Missing textual completion')
        stripped=text.strip()
        if stripped.startswith('```'):
            import re
            stripped=re.sub(r'^```(?:json)?\s*|\s*```$','',stripped)
        obj=json.loads(stripped)
        if not isinstance(obj,dict) or not isinstance(obj.get('answer'),str) or not obj['answer'].strip():
            raise ValueError('Missing nonempty string answer; raw retained')
        if not isinstance(obj.get('citations'),list): raise ValueError('Missing citation list; raw retained')
        payload=json.loads(messages[1]['content'])
        facts=list(payload['evidence'])
        for peer in payload.get('peer_messages',[]): facts.extend(peer.get('evidence',[]))
        visible={(f['title'],f['sentence_id']) for f in facts}
        valid=[]; invalid=[]
        for citation in obj['citations']:
            if (isinstance(citation,list) and len(citation)==2 and isinstance(citation[0],str)
                    and type(citation[1]) is int and tuple(citation) in visible):
                if citation not in valid: valid.append(citation)
            else: invalid.append(citation)
        parsed={'answer':obj['answer'].strip(),'citations':valid,'invalid_citations':invalid,
                'status':'VALID','provider_refusal':False}
    return {**parsed,'request_hash':saved.get('request_hash'), 'cache_hit':saved.get('cache_hit'),
            'usage':raw.get('usage'), 'returned_model':raw.get('model')}


def run_question(q,label,backend,token_count=None):
    record={'question_id':q['id'],'cells':{},'initial':{},'diagnostics':{}}
    for placement in q['execution_order']:
        initial=[]
        for node in range(3):
            messages=initial_messages(q,placement,node)
            saved=backend.complete(messages,{'question_id':q['id'],'placement':placement,'stage':'initial','node':node})
            initial.append(parse_result(saved,messages))
        record['initial'][placement]=initial
        record['diagnostics'][placement]={'initial_majority':majority(initial),
                 'initial_majority_score':score(majority(initial),label['answer'])}
        for mode in q['revision_order']:
            revised=[]; deliveries=[]; prompt_lengths=[]
            for node in range(3):
                messages=revision_messages(q,placement,node,initial,mode)
                payload=json.loads(messages[1]['content'])
                for peer in payload['peer_messages']:
                    deliveries.append({'sender':peer['sender'],'recipient':node,'answer':peer['answer'],
                                       'facts':peer['evidence'],
                                       'tokens':token_count(json.dumps(peer,ensure_ascii=False)) if token_count else None})
                prompt_lengths.append(sum(token_count(m['content']) for m in messages) if token_count else None)
                saved=backend.complete(messages,{'question_id':q['id'],'placement':placement,'stage':mode,'node':node})
                revised.append(parse_result(saved,messages))
            answer=majority(revised)
            cell={'answer':answer,**score(answer,label['answer']),'responses':revised,
                  'initial_request_hashes':[r['request_hash'] for r in initial], 'logical_calls':6,
                  'deliveries':deliveries,'prompt_content_tokens_by_node':prompt_lengths}
            record['cells'][placement+('_comm' if mode=='broadcast' else '_nocomm')]=cell
    # Gold is first used after generating every response for this question.
    for name,cell in record['cells'].items():
        placement=name.split('_')[0]; initial=record['initial'][placement]
        support=set(map(tuple,label['supporting_facts']))
        delivered=[set() for _ in range(3)]
        for delivery in cell['deliveries']:
            delivered[delivery['recipient']].update((f['title'],f['sentence_id']) for f in delivery['facts'])
        own=[{(d['title'],i) for d in docs for i in range(len(d['sentences']))}
             for docs in q['placements'][placement]['assignments']]
        cell['new_annotated_support_delivered_by_node']=[len((facts & support)-local) for facts,local in zip(delivered,own)]
        cell['all_support_accessible_by_node']=[support <= (local | facts) for local,facts in zip(own,delivered)]
        cell['initial_agent_scores']=[score(r['answer'],label['answer']) for r in initial]
        cell['revised_agent_scores']=[score(r['answer'],label['answer']) for r in cell['responses']]
        cell['any_initial_exact']=any(s['em']==100 for s in cell['initial_agent_scores'])
        cell['any_revised_exact']=any(s['em']==100 for s in cell['revised_agent_scores'])
        cell['correct_to_wrong_nodes']=[i for i,(a,b) in enumerate(zip(cell['initial_agent_scores'],cell['revised_agent_scores']))
                                        if a['em']==100 and b['em']!=100]
        cell['wrong_to_correct_nodes']=[i for i,(a,b) in enumerate(zip(cell['initial_agent_scores'],cell['revised_agent_scores']))
                                        if a['em']!=100 and b['em']==100]
        cell['aggregator_rejected_correct']=cell['any_revised_exact'] and cell['em']!=100
    record['statistics_row']={'question_id':q['id'],**{name+'_f1':cell['f1']/100 for name,cell in record['cells'].items()}}
    return record


def mock_transport(request,request_id):
    payload=json.loads(request['messages'][1]['content'])
    facts=payload['evidence']
    # Source-index fixtures exercise actual valid evidence delivery to all peers.
    citations=[[f['title'],f['sentence_id']] for f in facts]
    return 200,{'x-request-id':'MOCK_'+request_id},{'id':'MOCK_'+request_id,'model':'MOCK_ONLY',
        'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'answer':'noanswer','citations':citations})}}],
        'usage':{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150}}


def dry_run():
    manifest,tok=freeze(); config=json.loads((ROOT/'frozen/config.json').read_text())
    config={**config,'min_interval_seconds':0.,'mock_only':True}
    directory=ROOT/'dry_run'
    backend=BudgetBackend(directory,config,transport=mock_transport,token_count=tok.count,sleep=lambda _:None)
    questions=[]; e7=[]
    for q in manifest['questions']:
        label=manifest['private']['labels'][q['id']]
        questions.append(run_question(q,label,backend,tok.count))
        initial=questions[-1]['initial']['distributed']
        plan=build_e7(q,label,initial,token_count=tok.count)
        if plan['eligible']:
            messages=[e7_messages(q,plan,initial[plan['receiver']],arm) for arm in ['relevant','redundant','distractor']]
            if len({digest(m) for m in messages})!=3: raise ValueError('E7 interventions identical')
        e7.append(plan)
    first=backend.summary()
    for q in manifest['questions']:
        run_question(q,manifest['private']['labels'][q['id']],backend,tok.count)
    second=backend.summary()
    if first['http_attempts']!=144 or second['http_attempts']!=144:
        raise ValueError('Dry run/resume call count mismatch')
    atomic_json(directory/'mock_question_outputs.json',questions)
    atomic_json(directory/'e7_distinctness.json',e7)
    atomic_json(directory/'validation.json',{'status':'PASS_MOCK_ONLY','physical_mock_calls':first['http_attempts'],
        'resume_additional_mock_calls':second['http_attempts']-first['http_attempts'],
        'e6_unique_questions':len(questions),'e6_cells_per_question':4,
        'e7_provisionally_distinct_triples':sum(p['eligible'] for p in e7),
        'e7_paid_ready':False,'real_provider_calls':0,'frozen_manifest_sha256':manifest['manifest_sha256'],
        'frozen_config_sha256':sha256(ROOT/'frozen/config.json')})
    return json.loads((directory/'validation.json').read_text())


def pilot():
    manifest,tok=freeze()
    gate=ROOT/'dry_run/validation.json'
    if not gate.is_file() or json.loads(gate.read_text()).get('status')!='PASS_MOCK_ONLY':
        raise ValueError('Local dry run is required before paid inference')
    if json.loads(gate.read_text())['frozen_manifest_sha256']!=manifest['manifest_sha256']:
        raise ValueError('Dry run covered another frozen manifest')
    if json.loads(gate.read_text())['frozen_config_sha256']!=sha256(ROOT/'frozen/config.json'):
        raise ValueError('Frozen configuration changed after dry run')
    for report in ['00_REPOSITORY_AUDIT.md','01_CAUSAL_DESIGN_AND_PREREGISTRATION.md',
                   '02_API_COST_AND_CALL_BUDGET.md','03_DRY_RUN_VALIDATION.md']:
        if not (ROOT/'reports'/report).is_file(): raise ValueError('Pre-inference report missing: '+report)
    config=json.loads((ROOT/'frozen/config.json').read_text())
    directory=ROOT/'paid_pilot'
    directory.mkdir(exist_ok=True)
    with FileLock(str(directory/'study.lock'),timeout=0):
        backend=BudgetBackend(directory,config,token_count=tok.count)
        start=time.time(); status={'status':'RUNNING','completed_questions':0,'target_questions':8,
                                 'start_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(directory/'status.json',status)
        try:
            for q in manifest['questions']:
                checkpoint=directory/'questions'/(q['id']+'.json')
                if checkpoint.exists():
                    result=json.loads(checkpoint.read_text())
                    expected=result.get('content_hash')
                    if expected!=digest({k:v for k,v in result.items() if k!='content_hash'}):
                        raise ValueError('Question checkpoint checksum mismatch')
                else:
                    result=run_question(q,manifest['private']['labels'][q['id']],backend,tok.count)
                    result['content_hash']=digest(result)
                    atomic_json(checkpoint,result)
                status['completed_questions']+=1
                status.update({'usage':backend.summary(),'runtime_this_session_seconds':time.time()-start})
                atomic_json(directory/'status.json',status)
                print(json.dumps({'completed_questions':status['completed_questions'],**backend.summary()}),flush=True)
            status['status']='COMPLETE'
        except Exception as exc:
            status.update({'status':'INCOMPLETE','error_type':type(exc).__name__,
                           'error_message':str(exc) if isinstance(exc,ValueError) else
                           'Inference stopped safely; inspect saved attempt statuses. No blind retry.'})
            raise
        finally:
            status.update({'usage':backend.summary(),'runtime_this_session_seconds':time.time()-start,
                           'end_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
            atomic_json(directory/'status.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','dry-run','pilot','status'])
    args=parser.parse_args()
    if args.action=='freeze':
        manifest,_=freeze(); result={'status':'FROZEN_NOT_EXECUTED','manifest_sha256':manifest['manifest_sha256'],
                                   'selection':manifest['selection'],'protocol':manifest['protocol']}
    elif args.action=='dry-run': result=dry_run()
    elif args.action=='pilot': result=pilot()
    else:
        p=ROOT/'paid_pilot/status.json'; result=json.loads(p.read_text()) if p.exists() else {'status':'NOT_RUN'}
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
