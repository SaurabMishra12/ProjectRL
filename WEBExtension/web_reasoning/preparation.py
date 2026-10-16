"""Freeze local verified inputs before any provider access."""
from collections import Counter
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
from .config import validate_config
from .data import load_verified,allocate
from .splits import freeze_splits,balanced_sinks,normalize
from .storage import atomic_json,digest
from .tokenization import UTF8FixtureTokenizer,PinnedTokenizer
from .serialization import public_from_dict
from .ablations import compile_variant

PACKAGE=Path(__file__).resolve().parent
COHORTS=('pilot','development','evaluation')


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_tokenizer(config,descriptor):
    if config.mode=='MOCK':return UTF8FixtureTokenizer()
    spec=descriptor.get('tokenizer',{})
    if spec.get('revision')!=config.tokenizer_revision:raise ValueError('tokenizer_revision_mismatch')
    return PinnedTokenizer(spec['path'],spec['revision'],spec['sha256'])


def read_aliases(descriptor):
    if descriptor is None:return None
    data=Path(descriptor['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=descriptor['sha256']:raise ValueError('alias_hash_mismatch')
    try:obj=json.loads(data)
    except ValueError:obj=[json.loads(line) for line in data.splitlines()]
    if isinstance(obj,list):obj={row['Q_id']:row for row in obj}
    if not isinstance(obj,dict):raise ValueError('invalid_alias_file')
    return obj


def nested_keys(examples,n,seed):
    if n>len(examples):raise ValueError('scaling_cohort_exceeds_evaluation')
    groups={}
    for e in examples:groups.setdefault(dict(e.targets.strata).get('type','unknown'),[]).append(e)
    quotas={k:int(n*len(v)/max(1,len(examples))) for k,v in groups.items()}
    priority=sorted(groups,key=lambda k:(-(n*len(groups[k])/max(1,len(examples))-quotas[k]),k))
    for k in priority[:n-sum(quotas.values())]:quotas[k]+=1
    return sorted(e.public.key for k,rows in groups.items() for e in sorted(rows,key=lambda e:digest([seed,e.public.key]))[:quotas[k]])


def prepare_run(config_path,sources,output_dir):
    config=validate_config(json.loads(Path(config_path).read_text()))
    descriptor=json.loads(Path(sources).read_text()) if not isinstance(sources,dict) else sources
    allowed={'dataset','provenance','train','evaluation','aliases','sample_sizes','tokenizer','provider','placement','source_url','release_id'}
    if set(descriptor)-allowed:raise ValueError('unknown_source_descriptor_field')
    if 'provider' in descriptor:
        from .live import validate_provider
        validate_provider(descriptor['provider'])
        endpoint=descriptor['provider']['base_url']
        if config.provider_base_url and config.provider_base_url!=endpoint:raise ValueError('provider_identity_mismatch')
        config=replace(config,provider_base_url=endpoint)
    if descriptor.get('dataset') not in ('2wiki','finqa') or descriptor.get('provenance') not in ('publisher','synthetic'):
        raise ValueError('dataset_and_provenance_required')
    if config.mode=='REAL' and descriptor['provenance']!='publisher':raise ValueError('synthetic_is_not_real_data')
    if config.mode=='REAL' and descriptor['dataset']=='2wiki' and 'aliases' not in descriptor:raise ValueError('official_alias_file_required')
    tokenizer=make_tokenizer(config,descriptor)
    alias=read_aliases(descriptor.get('aliases'))
    def load(name):
        if name not in descriptor:return ()
        file=descriptor[name]
        return load_verified(file['path'],file['sha256'],descriptor['dataset'],alias)
    train=load('train');evaluation=load('evaluation')
    sizes={'pilot':20,'development':40,'evaluation':200,'scaling':50,**descriptor.get('sample_sizes',{})}
    if set(sizes)!={'pilot','development','evaluation','scaling'} or any(type(n) is not int or n<0 for n in sizes.values()):raise ValueError('invalid_sample_sizes')
    partitions=freeze_splits(train,{k:sizes[k] for k in ('pilot','development')},config.seed)
    partitions.update(freeze_splits(evaluation,{'evaluation':sizes['evaluation']},config.seed))
    # Cross-source grouping is validated across publisher splits too, not merely IDs.
    seen={}
    for cohort,rows in partitions.items():
        for e in rows:
            markers=[('question',normalize(e.public.question))]
            if e.public.dataset=='2wiki':markers.extend(('source',normalize(s.title)) for s in e.public.sources)
            else:markers.append(('report',dict(e.targets.strata)['report']))
            for marker in markers:
                if marker in seen and seen[marker]!=cohort:raise ValueError('cross_partition_question_or_source_overlap')
                seen[marker]=cohort
    scaling=nested_keys(partitions['evaluation'],sizes['scaling'],config.seed)
    directory=Path(output_dir).resolve();directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():
        old=json.loads((directory/'manifest.json').read_text())
        if old['config']!=json.loads(json.dumps(config.to_dict())) or old['sources']!=descriptor:raise ValueError('prepared_run_is_immutable')
        verify_run(directory);return directory/'manifest.json'
    files={};frozen_sinks={};counts={}
    for cohort,rows in partitions.items():
        public_path=directory/(cohort+'_inputs.json');target_path=directory/(cohort+'_targets.json')
        atomic_json(public_path,[asdict(e.public) for e in rows]);atomic_json(target_path,{e.public.key:asdict(e.targets) for e in rows})
        files[public_path.name]=file_sha(public_path);files[target_path.name]=file_sha(target_path)
        counts[cohort]=len(rows);frozen_sinks[cohort]={}
        for n in sorted(set(config.team_sizes)|{3}):
            frozen_sinks[cohort][str(n)]=balanced_sinks([e.public.key for e in rows],n,config.seed)
            assignments={}
            for e in rows:
                placement=descriptor.get('placement','natural')
                compiled=compile_variant(e,config,config,n,placement,tokenizer.count)
                assignments[e.public.key]=[[a.atom_id for a in shard] for shard in compiled['assignments']]
            path=directory/f'{cohort}_N{n}_assignments.json';atomic_json(path,assignments);files[path.name]=file_sha(path)
    code={str(p.relative_to(PACKAGE)):file_sha(p) for p in PACKAGE.rglob('*.py') if 'tests' not in p.parts}
    code['references/sources.json']=file_sha(PACKAGE/'references/sources.json')
    router_sha=file_sha(config.router_checkpoint) if config.router_checkpoint else None
    manifest={'version':'web_reasoning_run_v1','config':config.to_dict(),'sources':descriptor,'tokenizer':tokenizer.identity,
              'frozen_files':files,'code_hashes':code,'counts':counts,'scaling_keys':scaling,'sinks':frozen_sinks,
              'router_sha256':router_sha,'score_status':'NOT_RUN','currency_cost':None}
    atomic_json(directory/'manifest.json',manifest)
    (directory/'manifest.sha256').write_text(file_sha(directory/'manifest.json')+'\n')
    return directory/'manifest.json'


def verify_run(run_dir):
    directory=Path(run_dir);manifest_path=directory/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if file_sha(manifest_path)!=(directory/'manifest.sha256').read_text().strip():raise ValueError('manifest_hash_mismatch')
    for name,expected in manifest['frozen_files'].items():
        if file_sha(directory/name)!=expected:raise ValueError('frozen_input_hash_mismatch:'+name)
    for name,expected in manifest['code_hashes'].items():
        if file_sha(PACKAGE/name)!=expected:raise ValueError('code_hash_mismatch:'+name)
    checkpoint=manifest['config'].get('router_checkpoint')
    if checkpoint and file_sha(checkpoint)!=manifest['router_sha256']:raise ValueError('router_checkpoint_changed')
    for path in (directory/'results').glob('*.json'):
        saved=json.loads(path.read_text())
        if saved.get('sha256')!=digest(saved['run']) or saved.get('manifest_sha256')!=file_sha(manifest_path):raise ValueError('result_hash_mismatch')
    journal=directory/'inference'/'calls.jsonl'
    if journal.exists():
        from .backend import DurableBackend
        from .accounting import cost_summary
        tokenizer=make_tokenizer(validate_config(manifest['config']),manifest['sources'])
        backend=DurableBackend(directory/'inference',manifest['config'],None,tokenizer)
        summary=backend.summary()
    else:summary={'physical_attempts':0,'physical_completions':0}
    return {'status':'PASS','frozen_files_checked':len(manifest['frozen_files']),'code_files_checked':len(manifest['code_hashes']),
            'provider_mode':manifest['config']['mode'],'physical_usage':summary}


def forecast(manifest):
    c=validate_config(manifest['config']);counts=manifest['counts']
    def per(n):return 2*n+6 if c.experiment!='W2' else 5*n+4*max(1,n//2)+10
    return {'logical_completion_ceilings':{**{k:counts[k]*per(3) for k in COHORTS},
              'scaling_before_reuse':len(manifest['scaling_keys'])*sum(per(n) for n in c.team_sizes)},
            'nested_n3_logical_reuse':len(manifest['scaling_keys'])*per(3) if 3 in c.team_sizes else 0,
            'unique_physical_requests':'unknown_before_initial_states','input_tokens':'compiled_per_request_at_dispatch',
            'max_output_per_call':c.max_output_per_call,'currency_cost':None,'mode':c.mode}
