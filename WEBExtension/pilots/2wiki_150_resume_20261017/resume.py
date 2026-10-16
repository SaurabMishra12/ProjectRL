"""One explicitly approved replacement; immutable original timeout and budgets."""
import argparse
from dataclasses import replace
import importlib
import json
from pathlib import Path
import shutil

from WEBExtension.web_reasoning.backend import DurableBackend,ProviderRejected
from WEBExtension.web_reasoning.config import validate_config
from WEBExtension.web_reasoning.preparation import file_sha,make_tokenizer
from WEBExtension.web_reasoning.storage import atomic_json,canonical,digest
from WEBExtension.web_reasoning.accounting import valid_usage

base=importlib.import_module('WEBExtension.pilots.2wiki_150_20261017.amendment')
PARENT=base.ROOT
ROOT=Path('WEBExtension/web_reasoning/artifacts/2wiki_150_resume_20261017')
CODE=Path(__file__).resolve().parent
REPLACEMENT_KEY=''
original_report=base.report


def reserve_unknown(config,start):
    return replace(config,max_completions=config.max_completions-1,max_attempts=config.max_attempts-1,
        max_input_tokens=config.max_input_tokens-start['reserved_input'],
        max_output_tokens=config.max_output_tokens-start['reserved_output'])


def charged_usage(events):
    starts={e['attempt_id']:e for e in events if e['event']=='started'}
    terminal={e['attempt_id']:e for e in events if e['event'] in ('success','rejected')}
    used=[0,0]
    for aid,start in starts.items():
        end=terminal.get(aid)
        if end and end['event']=='rejected':continue
        usage=end.get('usage') or {} if end else {}
        for i,(field,reserve) in enumerate((('prompt_tokens','reserved_input'),('completion_tokens','reserved_output'))):
            value=usage.get(field)
            used[i]+=value if valid_usage(value) else start[reserve]
    return tuple(used)


class OneReplacementBackend(DurableBackend):
    def _reserve(self,request):
        if request.key==REPLACEMENT_KEY and any(e['event']=='started' and e['key']==REPLACEMENT_KEY for e in self._read()):
            raise ProviderRejected('one_authorized_replacement_attempt_used')
        return super()._reserve(request)


def prepare():
    if (ROOT/'manifest.json').exists():raise ValueError('prepared_resume_exists')
    m=base.verify();parent_sha=file_sha(PARENT/'manifest.json')
    events=[json.loads(l) for l in (PARENT/'inference/calls.jsonl').read_text().splitlines()]
    closed={e['attempt_id'] for e in events if e['event'] in ('success','rejected')}
    unknown=[e for e in events if e['event']=='started' and e['attempt_id'] not in closed]
    assert len(unknown)==1
    start=unknown[0];aid=start['attempt_id']
    assert any(e['event']=='transport_error' and e['attempt_id']==aid and e['error_type']=='APITimeoutError' for e in events)
    assert not (PARENT/'inference/attempts'/(aid+'.json')).exists()
    request=json.loads((PARENT/'inference/requests'/(start['key']+'.json')).read_text())
    assert digest(request)==start['key']
    config=reserve_unknown(validate_config(m['config']),start)
    ROOT.mkdir(parents=True,exist_ok=True)
    for name in ('public_inputs.json','targets.json','assignments.json'):
        shutil.copyfile(PARENT/name,ROOT/name)
    for path in (PARENT/'imported_results').glob('*.json'):
        dest=ROOT/'imported_results'/path.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(path,dest)
    cache={str(p.relative_to(PARENT/'inference')):file_sha(p) for sub in ('requests','responses','attempts')
           for p in (PARENT/'inference'/sub).glob('*.json')}
    cache['logical_refs.jsonl']=file_sha(PARENT/'inference/logical_refs.jsonl')
    base.import_cache(PARENT/'inference',ROOT/'inference',cache)
    # This is a NEW resolved-lineage ledger, not an edit to the stopped ledger.
    resolved=[e for e in events if e['attempt_id']!=aid]
    (ROOT/'inference/calls.jsonl').write_text(''.join(canonical(e)+'\n' for e in resolved))
    shutil.copyfile(PARENT/'inference/calls.jsonl',ROOT/'parent_calls_full.jsonl')
    auth={**m['authorization'],'replacement_authorization':{
        'question':'May I make one replacement attempt for question 55 within the existing total budget?',
        'user_answer':'Yes, allow one replacement attempt','attempt_id':aid,'request_key':start['key'],
        'rule':'At most one new physical attempt for this key, including confirmed rejections. Original uncertain attempt remains unresolved and reserved at maximum usage.'}}
    atomic_json(ROOT/'authorization.json',auth)
    atomic_json(ROOT/'quarantined_attempt.json',{'original_start':start,'original_ledger_sha256':file_sha(PARENT/'inference/calls.jsonl'),
        'original_ledger':str(PARENT/'inference/calls.jsonl'),'original_status':'UNKNOWN_PROVIDER_GENERATION_AND_BILLING',
        'reservation_counts_as_one_completion':True,'human_authorized_replacement':True})
    parent_files=[p for p in PARENT.rglob('*') if p.is_file() and (p.suffix in ('.json','.jsonl') or p.name=='manifest.sha256')]
    m['source_hashes']={**m['source_hashes'],**{str(p):file_sha(p) for p in parent_files}}
    m['files']={name:file_sha(ROOT/name) for name in ('public_inputs.json','targets.json','assignments.json',
        'authorization.json','quarantined_attempt.json','parent_calls_full.jsonl')}
    m['files'].update({str(p.relative_to(ROOT)):file_sha(p) for p in (ROOT/'imported_results').glob('*.json')})
    m['code'].update({str(p):file_sha(p) for p in CODE.glob('*.py')})
    m.update({'version':'2wiki_150_one_approved_timeout_replacement_v1','config':config.to_dict(),
        'authorization':auth,'parent_manifest_sha256':parent_sha,'parent_root':str(PARENT),
        'quarantined_start':start,'cache_import':cache,'combined_ceiling':json.loads((PARENT/'manifest.json').read_text())['config']})
    m['limitations']+=['An Azure Round-0 timeout occurred on question 55 after 54 complete questions. The user explicitly authorized one replacement attempt; two other saved Round-0 responses for that question are reused.',
        'Original timeout remains unresolved in the unchanged parent ledger. The new working ledger imports only resolved events; a full parent-ledger snapshot and hashes preserve the excluded unknown event. Its maximum reservation is subtracted from every working ceiling.']
    atomic_json(ROOT/'manifest.json',m);sha=file_sha(ROOT/'manifest.json');(ROOT/'manifest.sha256').write_text(sha+'\n')
    for path in (PARENT/'results').glob('*.json'):
        row=base.load_result(path,parent_sha)
        atomic_json(ROOT/'results'/path.name,{'result':row,'sha256':digest(row),'manifest_sha256':sha,
            'origin_file':str(path),'origin_file_sha256':file_sha(path),'origin_manifest_sha256':parent_sha})
    tokenizer=make_tokenizer(config,m['descriptor'])
    backend=DurableBackend(ROOT/'inference',config,None,tokenizer)
    assert backend.summary()['physical_completions']==464
    print(canonical({'reused_complete_questions':54,'remaining_questions':96,'reused_successful_responses':464,
        'quarantined_unknown_attempts':1,'working_ceiling':config.to_dict(),'manifest_sha256':sha}))


def report(*args,**kwargs):
    summary=original_report(*args,**kwargs)
    m=args[2];start=m['quarantined_start'];physical=summary['physical_usage']
    physical['physical_attempts']+=1;physical['uncertain_attempts']+=1
    physical['quarantined_prior_uncertain_attempts']=1
    physical['possible_completions_including_unknown']=physical['physical_completions']+physical['uncertain_attempts']
    charged_in,charged_out=charged_usage(args[1].journal.read())
    physical['conservative_total_input_including_unknown']=charged_in+start['reserved_input']
    physical['conservative_total_output_including_unknown']=charged_out+start['reserved_output']
    summary['additional_physical_usage']['physical_attempts']+=1
    summary['quarantined_prior_attempt']={'attempt_id':start['attempt_id'],'key':start['key'],
        'reserved_input':start['reserved_input'],'reserved_output':start['reserved_output'],'human_authorized_replacement':True}
    limit=m['combined_ceiling']
    assert physical['physical_attempts']<=limit['max_attempts']
    assert physical['possible_completions_including_unknown']<=limit['max_completions']
    assert physical['conservative_total_input_including_unknown']<=limit['max_input_tokens']
    assert physical['conservative_total_output_including_unknown']<=limit['max_output_tokens']
    atomic_json(ROOT/'summary.json',summary)
    with (ROOT/'REPORT.md').open('a') as f:
        f.write('\nOne historical Azure timeout remains unresolved in the unchanged parent ledger. The user approved one replacement attempt. Its maximum reserved usage remains charged to the combined ceiling; physical attempt totals above include it.\n')
    return summary


def activate():
    global REPLACEMENT_KEY
    m=json.loads((ROOT/'manifest.json').read_text());REPLACEMENT_KEY=m['quarantined_start']['key']
    base.ROOT=ROOT;base.DurableBackend=OneReplacementBackend;base.report=report


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--verify',action='store_true');g.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.prepare:prepare()
    else:
        activate()
        if args.verify:print(canonical({'status':'PASS','questions':base.verify()['question_count']}))
        else:base.run()
