"""Freeze 20 existing publisher questions plus 80 outcome-blind expansion rows."""
from collections import Counter
from dataclasses import asdict
import heapq
import importlib
import io
import json
from pathlib import Path
import zipfile
from WEBExtension.web_reasoning.data import from_2wiki,allocate
from WEBExtension.web_reasoning.preparation import read_aliases,make_tokenizer,file_sha,PACKAGE
from WEBExtension.web_reasoning.config import validate_config
from WEBExtension.web_reasoning.splits import normalize
from WEBExtension.web_reasoning.storage import atomic_json,digest

CODE=Path(__file__).resolve().parent
OLD=Path('WEBExtension/web_reasoning/artifacts/2wiki_pilot_20261016')
ROOT=Path('WEBExtension/web_reasoning/artifacts/2wiki_revised_20261016')


def complete_support(row):
    locators={(title,i) for title,sentences in row['context'] for i in range(len(sentences))}
    support=row.get('supporting_facts',[])
    return bool(support) and all(isinstance(pair,(list,tuple)) and len(pair)==2 and isinstance(pair[0],str) and type(pair[1]) is int and tuple(pair) in locators for pair in support)


def main():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'manifest.json').exists():raise ValueError('prepared_run_exists_do_not_replace')
    old_manifest=json.loads((OLD/'representative_run/manifest.json').read_text())
    for name,sha in old_manifest['frozen_files'].items():assert file_sha(OLD/'representative_run'/name)==sha
    descriptor=json.loads((OLD/'sources_representative.json').read_text())
    assert file_sha(OLD/'source/data_ids_april7.zip')==descriptor['release_id']
    original=json.loads((OLD/'source/representative_pilot_training_rows.json').read_text())
    lookup={str(r['_id']):r for r in original}
    first_ids=[r['key'] for r in json.loads((OLD/'representative_run/pilot_inputs.json').read_text())]
    first=[lookup[q] for q in first_ids];assert len(first)==20
    used={('question',normalize(r['question'])) for r in first}|{('title',normalize(t)) for r in first for t,_ in r['context']}
    arrays=importlib.import_module('WEBExtension.pilots.2wiki_20261016.prepare_pilot').array_rows
    heaps={};counts=Counter();excluded=Counter()
    with zipfile.ZipFile(OLD/'source/data_ids_april7.zip') as archive:
        with archive.open('train.json') as stream:
            for row in arrays(io.TextIOWrapper(stream,encoding='utf-8')):
                titles=[t for t,_ in row['context']]
                if len(set(titles))!=len(titles):excluded['duplicate_title']+=1;continue
                if any(not t or any(not isinstance(s,str) or not s for s in sentences) for t,sentences in row['context']):excluded['invalid_text']+=1;continue
                if not complete_support(row):excluded['missing_or_invalid_annotated_support_locator']+=1;continue
                kind=row['type'];counts[kind]+=1
                markers={('question',normalize(row['question']))}|{('title',normalize(t)) for t in titles}
                if markers&used:continue
                score=int(digest(['2wiki-revised-expansion',20261016,row['_id']]),16)
                heap=heaps.setdefault(kind,[]);item=(-score,str(row['_id']),row)
                if len(heap)<1000:heapq.heappush(heap,item)
                elif score < -heap[0][0]:heapq.heapreplace(heap,item)
    quotas={k:max(1,int(80*v/sum(counts.values()))) for k,v in counts.items()}
    while sum(quotas.values())<80:
        k=max(counts,key=lambda k:(80*counts[k]/sum(counts.values())-quotas[k],k));quotas[k]+=1
    rows=list(first)
    for kind in sorted(heaps):
        n=0
        for _,_,row in sorted(heaps[kind],key=lambda x:(-x[0],x[1])):
            markers={('question',normalize(row['question']))}|{('title',normalize(t)) for t,_ in row['context']}
            if markers&used:continue
            used.update(markers);rows.append(row);n+=1
            if n==quotas[kind]:break
        assert n==quotas[kind]
    assert len(rows)==100
    aliases=read_aliases(descriptor['aliases'])
    config=validate_config({**json.loads((OLD/'config.json').read_text()),'prompt_version':'relaxed_2wiki_v1',
        'schema_version':'relaxed_2wiki_v1','max_completions':1000,'max_attempts':1050,'max_input_tokens':10000000,
        'max_output_tokens':4100000,'max_output_per_call':4096,'packet_cap':512,
        'replicate_id':'2wiki-revised-gated-20261016','provider_base_url':descriptor['provider']['base_url']})
    tokenizer=make_tokenizer(config,descriptor);publics=[];targets={};assignments={};placement=Counter()
    old_assignment=json.loads((OLD/'representative_run/pilot_N3_assignments.json').read_text())
    for index,row in enumerate(rows):
        ex=from_2wiki(row,aliases);pub=ex.public;lookup={a.atom_id:a for s in pub.sources for a in s.atoms}
        shard_ids=old_assignment[pub.key] if index<20 else [[a.atom_id for a in shard] for shard in allocate(pub,3,'P',config.seed,tokenizer.count)]
        shards=[[lookup[k] for k in shard] for shard in shard_ids]
        assert sorted(a.atom_id for shard in shards for a in shard)==sorted(lookup)
        available={(a.title,a.sentence_id) for a in lookup.values()};gold=set(ex.targets.supporting_locators)
        assert gold<=available and gold
        placement[('pilot:' if index<20 else 'expansion:')+('co_located' if any(gold<={(a.title,a.sentence_id) for a in shard} for shard in shards) else 'disjoint')]+=1
        publics.append(asdict(pub));targets[pub.key]=asdict(ex.targets);assignments[pub.key]=shard_ids
    files={'public_inputs.json':publics,'targets.json':targets,'assignments.json':assignments}
    for name,value in files.items():atomic_json(ROOT/name,value)
    source_hashes={str(OLD/'source/data_ids_april7.zip'):file_sha(OLD/'source/data_ids_april7.zip'),
                   str(OLD/'source/representative_pilot_training_rows.json'):file_sha(OLD/'source/representative_pilot_training_rows.json'),
                   descriptor['aliases']['path']:descriptor['aliases']['sha256']}
    frozen={'version':'2wiki_revised_gated_v1','config':config.to_dict(),'descriptor':descriptor,'tokenizer':tokenizer.identity,
        'files':{name:file_sha(ROOT/name) for name in files},'code':{str(p):file_sha(p) for p in CODE.glob('*.py')},
        'backend_code':{str(p):file_sha(p) for p in PACKAGE.rglob('*.py') if 'tests' not in p.parts},'source_hashes':source_hashes,
        'pilot_keys':first_ids,'expansion_keys':[p['key'] for p in publics[20:]],'placement':dict(placement),
        'publisher_eligibility_exclusions':dict(excluded),
        'expansion_types':dict(Counter(r['type'] for r in rows[20:])),
        'gate':'All 20 complete; physical invalid/refusal/truncated output fraction <2%; NoComm EM <=15% under both readouts; instar and complementary strictly exceed NoComm, random, heuristic in EM and F1 under both readouts; neither below Dense sink EM or F1; exact instar/dense sink request keys.',
        'pilot_ceiling':{'max_completions':200,'max_attempts':220,'max_input_tokens':2000000,'max_output_tokens':820000},
        'authorization':'Direct user request to design and run revised N=3 2Wiki 20-question gate, expand to 100 only if it passes. No further dataset or post-gate tuning authorized.',
        'limitations':['Exploratory publisher training split; original 20 previously inspected; remaining 80 frozen before new outcomes.',
            'Whole source articles partitioned without gold-based placement; all annotated support available in union.',
            'Retains original 20 assignment exactly; no filtering based on NoComm performance.',
            'New common prompts invalidate reuse of the older strict-schema model calls; their uncertain attempt remains untouched.',
            'Source-disjoint selection within these 100 does not establish independence from all earlier training or controller-development data.']}
    atomic_json(ROOT/'manifest.json',frozen);(ROOT/'manifest.sha256').write_text(file_sha(ROOT/'manifest.json')+'\n')
    print(json.dumps({'prepared_questions':100,'pilot_questions':20,'placement':dict(placement),'gate':frozen['gate'],'max_new_pilot_calls':200},indent=2),flush=True)


if __name__=='__main__':main()
