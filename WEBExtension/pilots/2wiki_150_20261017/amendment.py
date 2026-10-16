"""Explicit user-authorized 150-question extension; preserve the failed pilot."""
import argparse
from collections import Counter
from dataclasses import asdict
import heapq
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

from WEBExtension.web_reasoning.backend import DurableBackend
from WEBExtension.web_reasoning.config import validate_config
from WEBExtension.web_reasoning.data import from_2wiki, allocate
from WEBExtension.web_reasoning.preparation import make_tokenizer, file_sha, read_aliases
from WEBExtension.web_reasoning.serialization import public_from_dict
from WEBExtension.web_reasoning.storage import atomic_json, digest, canonical, Journal
from WEBExtension.web_reasoning.splits import normalize
from WEBExtension.web_reasoning.live import live_transport
from WEBExtension.causal_study.readout_reassessment.analysis import paired

prior = importlib.import_module('WEBExtension.pilots.2wiki_revised_20261016.run')
prep = importlib.import_module('WEBExtension.pilots.2wiki_revised_20261016.prepare')
core = importlib.import_module('WEBExtension.pilots.2wiki_revised_20261016.core')
ROOT = Path('WEBExtension/web_reasoning/artifacts/2wiki_150_20261017')
ORIGIN = prep.ROOT
CODE = Path(__file__).resolve().parent
COST_METRICS = ('known_input_tokens','known_output_tokens','missing_usage_fields',
                'payload_tokens','metadata_tokens','omitted_facts')


def import_cache(source, destination, expected):
    """Copy verified complete physical lineage, never rewrite the origin ledger."""
    source, destination = Path(source), Path(destination)
    for name, sha in expected.items():
        if file_sha(source/name) != sha:
            raise ValueError('origin_hash_mismatch:'+name)
        if (destination/name).exists() and file_sha(destination/name) != sha:
            raise ValueError('destination_cache_already_changed:'+name)
    for name in expected:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source/name, target)


def load_result(path, manifest_sha):
    saved = json.loads(Path(path).read_text())
    if saved.get('manifest_sha256') != manifest_sha or digest(saved['result']) != saved.get('sha256'):
        raise ValueError('result_identity_mismatch:'+str(path))
    return saved['result']


def authorization_record(gate, origin_sha):
    return {'user_request':'increase the 20 to 150','user_authorized_total':150,
            'historical_manifest_sha256':origin_sha,'historical_gate':gate,
            'expansion_despite_failed_gate':not gate['pass'],
            'interpretation':'Explicit post-pilot sample-size amendment; no prompt, parser, router, readout or model change. Exploratory results; retain the failed gate.'}


def select_extra(stream, previous, n):
    used = {('question',normalize(p['question'])) for p in previous}
    used.update(('title',normalize(s['title'])) for p in previous for s in p['sources'])
    heaps = {}; counts = Counter(); excluded = Counter()
    for row in stream:
        titles = [t for t,_ in row['context']]
        if len(set(titles)) != len(titles):
            excluded['duplicate_title'] += 1; continue
        if any(not t or any(not isinstance(s,str) or not s for s in ss) for t,ss in row['context']):
            excluded['invalid_text'] += 1; continue
        if not prep.complete_support(row):
            excluded['missing_or_invalid_annotated_support_locator'] += 1; continue
        kind = row['type']; counts[kind] += 1
        markers = {('question',normalize(row['question']))} | {('title',normalize(t)) for t in titles}
        if markers & used: continue
        score = int(digest(['2wiki-150-extension',20261017,row['_id']]),16)
        heap = heaps.setdefault(kind,[]); item = (-score,str(row['_id']),row)
        if len(heap) < 1000: heapq.heappush(heap,item)
        elif score < -heap[0][0]: heapq.heapreplace(heap,item)
    quotas = {k:int(n*v/sum(counts.values())) for k,v in counts.items()}
    while sum(quotas.values()) < n:
        k = max(counts,key=lambda k:(n*counts[k]/sum(counts.values())-quotas[k],k))
        quotas[k] += 1
    selected = []
    for kind in sorted(quotas):
        chosen = 0
        if not quotas[kind]: continue
        for _,_,row in sorted(heaps[kind],key=lambda x:(-x[0],x[1])):
            markers = {('question',normalize(row['question']))} | {('title',normalize(t)) for t,_ in row['context']}
            if markers & used: continue
            selected.append(row); used.update(markers); chosen += 1
            if chosen == quotas[kind]: break
        if chosen != quotas[kind]: raise ValueError('insufficient_source_disjoint_candidates:'+kind)
    if len(selected) != n: raise ValueError('selection_count')
    return selected, dict(quotas), dict(excluded)


def prepare():
    if (ROOT/'manifest.json').exists(): raise ValueError('prepared_run_exists_do_not_replace')
    old = prior.verify(); origin_sha = file_sha(ORIGIN/'manifest.json')
    gate = json.loads((ORIGIN/'gate_decision.json').read_text())
    publics = json.loads((ORIGIN/'public_inputs.json').read_text())
    targets = json.loads((ORIGIN/'targets.json').read_text())
    assignments = json.loads((ORIGIN/'assignments.json').read_text())
    assert len(publics) == 100
    config = validate_config({**old['config'],'max_completions':1471,'max_attempts':1531,
                             'max_input_tokens':13000000,'max_output_tokens':6100000})
    tokenizer = make_tokenizer(config,old['descriptor'])
    source = prep.OLD/'source/data_ids_april7.zip'
    arrays = importlib.import_module('WEBExtension.pilots.2wiki_20261016.prepare_pilot').array_rows
    with zipfile.ZipFile(source) as archive:
        with archive.open('train.json') as stream:
            extra, quotas, excluded = select_extra(arrays(io.TextIOWrapper(stream,encoding='utf-8')),publics,50)
    aliases = read_aliases(old['descriptor']['aliases'])
    for row in extra:
        ex = from_2wiki(row,aliases); public = ex.public
        shards = allocate(public,3,'P',config.seed,tokenizer.count)
        publics.append(asdict(public)); targets[public.key] = asdict(ex.targets)
        assignments[public.key] = [[a.atom_id for a in shard] for shard in shards]
    assert len(publics) == len(targets) == len(assignments) == 150
    placement = Counter()
    for index, public in enumerate(publics):
        atoms = {a['atom_id']:a for s in public['sources'] for a in s['atoms']}
        shards = assignments[public['key']]
        assert sorted(k for shard in shards for k in shard) == sorted(atoms)
        gold = {tuple(x) for x in targets[public['key']]['supporting_locators']}
        assert gold and gold <= {(a['title'],a['sentence_id']) for a in atoms.values()}
        colocated = any(gold <= {(atoms[k]['title'],atoms[k]['sentence_id']) for k in shard} for shard in shards)
        cohort = 'pilot' if index < 20 else 'additional'
        placement[cohort+(' co_located' if colocated else ' disjoint')] += 1
    origin_files = {'manifest.json','manifest.sha256','gate_decision.json','summary.json',
                    'public_inputs.json','targets.json','assignments.json','inference/config.json',
                    'inference/calls.jsonl','inference/logical_refs.jsonl'}
    origin_files.update(str(p.relative_to(ORIGIN)) for sub in ('requests','responses','attempts')
                        for p in (ORIGIN/'inference'/sub).glob('*.json'))
    origin_files.update('results/'+qid+'.json' for qid in old['pilot_keys'])
    origin_hashes = {name:file_sha(ORIGIN/name) for name in sorted(origin_files)}
    cache_hashes = {name.removeprefix('inference/'):sha for name,sha in origin_hashes.items()
                    if name.startswith('inference/') and name != 'inference/config.json'}
    ROOT.mkdir(parents=True,exist_ok=True)
    import_cache(ORIGIN/'inference',ROOT/'inference',cache_hashes)
    for qid in old['pilot_keys']:
        path = ORIGIN/'results'/(qid+'.json'); load_result(path,origin_sha)
        target = ROOT/'imported_results'/path.name; target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path,target)
    files = {'public_inputs.json':publics,'targets.json':targets,'assignments.json':assignments,
             'authorization.json':authorization_record(gate,origin_sha)}
    for name,value in files.items(): atomic_json(ROOT/name,value)
    imports = {str(p.relative_to(ROOT)):file_sha(p) for p in (ROOT/'imported_results').glob('*.json')}
    code = dict(old['code'])
    code.update({str(p):file_sha(p) for p in CODE.glob('*.py')})
    dependency = 'WEBExtension/causal_study/readout_reassessment/analysis.py'
    code[dependency] = file_sha(dependency)
    manifest = {'version':'2wiki_150_authorized_amendment_v1','config':config.to_dict(),
        'descriptor':old['descriptor'],'tokenizer':tokenizer.identity,'code':code,
        'backend_code':old['backend_code'],'source_hashes':old['source_hashes'],
        'files':{**{name:file_sha(ROOT/name) for name in files},**imports},
        'origin_root':str(ORIGIN),'origin_files':origin_hashes,'cache_import':cache_hashes,
        'historical_gate':gate,'historical_manifest_sha256':origin_sha,
        'pilot_keys':old['pilot_keys'],'additional_keys':[p['key'] for p in publics[20:]],
        'previously_frozen_unrun_keys':old['expansion_keys'],'newly_selected_keys':[p['key'] for p in publics[100:]],
        'additional_50_types':quotas,'publisher_eligibility_exclusions':excluded,'placement':dict(placement),
        'prior_physical_usage':json.loads((ORIGIN/'summary.json').read_text())['physical_usage'],
        'authorization':authorization_record(gate,origin_sha),'question_count':150,
        'new_physical_completion_ceiling':1300,'new_attempt_ceiling':1360,
        'limitations':old['limitations']+[
            'User explicitly authorized 150 after the 20-question gate failed. Historical gate remains false.',
            'Original 100 questions and assignments preserved; 50 added by seeded type-stratified source-disjoint sampling before outcomes.',
            'Report original 20, additional 130, and pooled 150 separately; no performance-based stopping or tuning.',
            'Imports 171 physical completions with complete prior ledger lineage. Budget totals include imported usage; currency unavailable.']}
    atomic_json(ROOT/'manifest.json',manifest)
    (ROOT/'manifest.sha256').write_text(file_sha(ROOT/'manifest.json')+'\n')
    backend = DurableBackend(ROOT/'inference',config,None,tokenizer)
    assert backend.summary()['physical_completions'] == manifest['prior_physical_usage']['physical_completions'] == 171
    print(canonical({'prepared':150,'reused_questions':20,'additional_questions':130,
                     'historical_gate_pass':gate['pass'],'placement':dict(placement),
                     'manifest_sha256':file_sha(ROOT/'manifest.json'),'physical_budget':config.max_completions}),flush=True)


def verify():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    assert file_sha(ROOT/'manifest.json') == (ROOT/'manifest.sha256').read_text().strip()
    for name,sha in manifest['files'].items(): assert file_sha(ROOT/name) == sha, name
    for name,sha in {**manifest['code'],**manifest['backend_code'],**manifest['source_hashes']}.items():
        assert file_sha(name) == sha, name
    for name,sha in manifest['origin_files'].items(): assert file_sha(ORIGIN/name) == sha, name
    return manifest


def arm_stats(rows, expected):
    stats = {}
    for arm in core.ARMS:
        stats[arm] = {'n':expected,'completed_records':len(rows),
            'readouts':{rule:{'correct':sum(r['arms'][arm]['readouts'][rule]['em']==100 for r in rows),
                **{metric:sum(r['arms'][arm]['readouts'][rule][metric] for r in rows)/expected
                   for metric in ('em','f1')}} for rule in core.READOUTS},
            **{metric:sum(r['arms'][arm][metric] for r in rows) for metric in COST_METRICS},
            'logical_calls':sum(len(r['arms'][arm]['logical_call_keys']) for r in rows)}
    return stats


def report(rows, backend, manifest, status='RUNNING', final=False):
    audit = prior.physical_audit(backend)
    cohorts = {'pilot':(rows[:20],20),'additional':(rows[20:],130),'pooled':(rows,150)}
    stats = {name:arm_stats(group,n) for name,(group,n) in cohorts.items()}
    contrasts = {}
    if final:
        comparisons = [('instar','nocomm'),('instar','heuristic'),('complementary','nocomm'),
                       ('complementary','random'),('complementary','heuristic'),
                       ('complementary','dense'),('instar','dense'),('full_union','dense')]
        for cohort,(group,n) in cohorts.items():
            if len(group) != n: continue
            for arm,base in comparisons:
                for rule in core.READOUTS:
                    contrasts[cohort+':'+arm+'-'+base+':'+rule] = {
                        metric:paired({r['question_key']:r['arms'][arm]['readouts'][rule][metric] for r in group},
                                      {r['question_key']:r['arms'][base]['readouts'][rule][metric] for r in group},binary=metric=='em')
                        for metric in ('em','f1')}
    prior_usage = manifest['prior_physical_usage']
    new_usage = {key:audit[key]-prior_usage[key] for key in ('physical_completions','physical_attempts',
        'known_input_tokens_lower_bound','known_output_tokens_lower_bound','rejected_attempts')}
    summary = {'status':status,'assigned_questions':150,'completed_questions':len(rows),
        'historical_gate':manifest['historical_gate'],'authorization':manifest['authorization'],
        'stats':stats,'contrasts':contrasts,'physical_usage':audit,'additional_physical_usage':new_usage,
        'router_k_distribution':dict(Counter(len(r['router_choices']['complementary']) for r in rows)),
        'sink_request_matches':sum(r['arms']['instar']['hub_request_key']==r['arms']['dense']['hub_request_key'] for r in rows),
        'manifest_sha256':file_sha(ROOT/'manifest.json'),'limitations':manifest['limitations']}
    atomic_json(ROOT/'summary.json',summary)
    lines=['# 2Wiki 150-question authorized extension','',f'Status: **{status}**. Completed {len(rows)}/150.',
        '', 'The original 20-question gate failed. The user explicitly authorized expansion to 150; the gate has not been changed.',
        '', 'Prompts, parser, routing, decoding and both readouts are unchanged. The 20 completed questions reuse their saved responses.',
        '', 'All assigned questions remain in operational denominators; missing records are zero in interim scores. Paired intervals are computed only for complete cohorts.',
        '', 'In-Star and Dense share an identical Sink-1 request by construction; this is a request-reuse identity, not a statistical equivalence discovery.',
        '', 'Per-arm logical model tokens overlap through cache reuse. Provider token usage is reported separately; currency cost is unavailable.','']
    for cohort,values in stats.items():
        lines += ['## '+cohort,'','| Arm | AA correct | AA F1 | Sink correct | Sink F1 | Logical model tokens | Payload | Metadata |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm,s in values.items():
            aa=s['readouts']['abstention_aware']; sink=s['readouts']['sink1']
            lines.append(f'| {arm} | {aa["correct"]}/{s["n"]} | {aa["f1"]:.3f} | {sink["correct"]}/{s["n"]} | {sink["f1"]:.3f} | {s["known_input_tokens"]+s["known_output_tokens"]} | {s["payload_tokens"]} | {s["metadata_tokens"]} |')
    lines += ['',f'Physical completions: {audit["physical_completions"]} total, including {prior_usage["physical_completions"]} imported; {new_usage["physical_completions"]} new.',
              f'Contract failures: {audit["contract_failures"]}/{audit["physical_completions"]}. Sink request matches: {summary["sink_request_matches"]}/{len(rows)}.',
              '', '[Summary with paired intervals](summary.json) · [Frozen manifest](manifest.json) · [Authorization amendment](authorization.json)',
              '', '## Limitations','']+['- '+item for item in manifest['limitations']]
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(ROOT/'execution_status.json',{'status':status,'completed':len(rows),'assigned':150,
                'historical_gate_pass':manifest['historical_gate']['pass'],'new_completions':new_usage['physical_completions']})
    return summary


def run():
    manifest=verify(); config=validate_config(manifest['config'])
    tokenizer=make_tokenizer(config,manifest['descriptor'])
    publics=[public_from_dict(p) for p in json.loads((ROOT/'public_inputs.json').read_text())]
    targets=json.loads((ROOT/'targets.json').read_text()); assignments=json.loads((ROOT/'assignments.json').read_text())
    from dotenv import dotenv_values
    env=manifest['descriptor']['provider']['api_key_env']
    if not os.environ.get(env):
        value=dotenv_values('.env').get(env)
        if value: os.environ[env]=value
    send=live_transport(manifest['descriptor']); next_time=0.
    observations=Journal(ROOT/'transport_observations.jsonl')
    def transport(request,attempt_id):
        nonlocal next_time
        while time.monotonic()<next_time: time.sleep(min(20.,next_time-time.monotonic()))
        started=time.monotonic(); status,headers,raw=send(request,attempt_id)
        delay=2.
        if status==429:
            try: delay=max(15.,float(next((v for k,v in headers.items() if k.lower()=='retry-after'),15)))
            except (TypeError,ValueError): delay=15.
        next_time=time.monotonic()+delay
        event={'attempt_id':attempt_id,'status':status,'latency_seconds':time.monotonic()-started,
               'finish_reason':((raw.get('choices') or [{}])[0].get('finish_reason')) if isinstance(raw,dict) else None,
               'usage':raw.get('usage') if isinstance(raw,dict) else None}
        observations.append(event); print(canonical({'response':event}),flush=True)
        return status,headers,raw
    backend=DurableBackend(ROOT/'inference',config,transport,tokenizer)
    rows=[]
    try:
        for index,public in enumerate(publics):
            path=ROOT/'results'/(public.key+'.json')
            if index<20:
                row=load_result(ROOT/'imported_results'/path.name,manifest['historical_manifest_sha256'])
            elif path.exists(): row=load_result(path,file_sha(ROOT/'manifest.json'))
            else:
                atoms={a.atom_id:a for s in public.sources for a in s.atoms}
                shards=[tuple(atoms[k] for k in shard) for shard in assignments[public.key]]
                row=prior.evaluate_row(public,shards,targets[public.key],backend,config,tokenizer,'additional')
                atomic_json(path,{'result':row,'sha256':digest(row),'manifest_sha256':file_sha(ROOT/'manifest.json')})
            assert row['question_key']==public.key
            rows.append(row)
            if index>=19:
                report(rows,backend,manifest)
                print(canonical({'COMPLETE_QUESTION':len(rows),'assigned':150,
                    'new_completions':backend.summary()['physical_completions']-171}),flush=True)
        final=report(rows,backend,manifest,'COMPLETE_150',final=True)
        verify()
        print(canonical({'FINAL_STATUS':final['status'],'completed':len(rows),'physical_usage':final['physical_usage']}),flush=True)
    except Exception as error:
        report(rows,backend,manifest,'STOPPED_'+type(error).__name__)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true'); group.add_argument('--verify',action='store_true')
    group.add_argument('--execute',action='store_true'); args=parser.parse_args()
    if args.prepare: prepare()
    elif args.verify: print(canonical({'status':'PASS','questions':verify()['question_count']}))
    else: run()
