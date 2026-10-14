"""Publisher-authenticated, question/document-disjoint controller study partitions."""
import heapq
import itertools
import json
import os
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from protocol import ROOT, load_official, normalize_row, normalize, digest, file_hash, select_sample, write_json
REVISION='1908d6afbbead072334abe2965f91bd2709910ab'
TRAIN_HASHES=['76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68',
              '713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f']

def document_keys(rows):
    return {' '.join(unicodedata.normalize('NFKC',d['title']).casefold().split()) for r in rows for d in r['documents']}

def inference_inputs(rows):
    return [{k:r[k] for k in ['id','question','documents']} for r in rows]

def assert_disjoint(parts):
    summary={}
    for name,rows in parts.items():
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate ID within '+name)
        if len({normalize(r['question']) for r in rows})!=len(rows):raise ValueError('Duplicate normalized question within '+name)
    for a,b in itertools.combinations(parts,2):
        ids={r['id'] for r in parts[a]} & {r['id'] for r in parts[b]}
        questions={normalize(r['question']) for r in parts[a]} & {normalize(r['question']) for r in parts[b]}
        docs=document_keys(parts[a]) & document_keys(parts[b])
        summary[a+'__'+b]={'id_overlap':len(ids),'normalized_question_overlap':len(questions),'candidate_title_overlap':len(docs)}
        if ids or questions or docs:raise ValueError('Leakage across '+a+' and '+b+': '+str(summary[a+'__'+b]))
    return summary

def training_sources():
    paths=[]
    cached=Path.home()/'.cache/huggingface/hub/datasets--hotpot_qa/snapshots'/REVISION/'fullwiki'
    for i,expected in enumerate(TRAIN_HASHES):
        name=f'train-{i:05d}-of-00002.parquet'
        source=cached/name
        if not source.exists():
            source=ROOT/'data/controller_training_sources'/name
            if not source.exists():
                source.parent.mkdir(parents=True,exist_ok=True)
                temporary=source.with_suffix('.download')
                url=f'https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{REVISION}/fullwiki/{name}'
                with urllib.request.urlopen(url,timeout=60) as response,open(temporary,'wb') as f:
                    while block:=response.read(1<<20):f.write(block)
                temporary.replace(source)
        if file_hash(source)!=expected:raise ValueError('Pinned publisher training hash mismatch')
        paths.append(source)
    return paths

def choose(rows,n,seed,blocked,weights):
    docs=document_keys(blocked);ids={r['id'] for r in blocked};qs={normalize(r['question']) for r in blocked}
    eligible=[r for r in rows if r['id'] not in ids and normalize(r['question']) not in qs and not(document_keys([r])&docs)]
    total=sum(weights.values());quotas={t:int(n*w/total) for t,w in weights.items()}
    order=sorted(weights,key=lambda t:(-(n*weights[t]/total-quotas[t]),t))
    for t in order[:n-sum(quotas.values())]:quotas[t]+=1
    result=[];seen=set()
    for t,count in quotas.items():
        group=sorted((r for r in eligible if r['type']==t),key=lambda r:digest([seed,r['id']]))
        kept=[]
        for r in group:
            q=normalize(r['question'])
            if q in seen:continue
            kept.append(r);seen.add(q)
            if len(kept)==count:break
        if len(kept)!=count:raise ValueError('Insufficient disjoint real '+t+' questions; increase deterministic candidate scan pool')
        result+=kept
    return sorted(result,key=lambda r:digest(['order',seed,r['id']]))

def prepare_splits(config):
    """Freeze before any training or held-out prediction; annotation-free assignment."""
    source={'publisher':'hotpotqa/hotpot_qa','revision':REVISION,'train_sha256':TRAIN_HASHES,
            'dev_sha256':json.loads((ROOT/'data/hotpot_dev_fullwiki_v1.provenance.json').read_text())['sha256']}
    spec={'version':'title_disjoint_v1','config':config,'source':source}
    directory=ROOT/'artifacts'/('controller_splits_'+digest(spec)[:16])
    manifest_path=directory/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['spec']!=spec:raise ValueError('Split spec changed')
        for name,checksum in manifest['files'].items():
            if file_hash(directory/name)!=checksum:raise ValueError('Frozen split file changed: '+name)
        return directory
    dev=load_official()
    legacy_path=ROOT.parent/'web_eval_subset_100.json'
    known_ids={r.get('id',r.get('_id')) for r in json.loads(legacy_path.read_text())} if legacy_path.exists() else set()
    weights=Counter(r['type'] for r in dev)
    heldout=choose([r for r in dev if r['id'] not in known_ids],config['test_questions'],config['split_seed'],[],weights)
    heldout_docs=document_keys(heldout);heldout_qs={normalize(r['question']) for r in heldout}
    heaps={t:[] for t in weights};eligible_counts=Counter();source_count=0
    import pyarrow.parquet as pq
    paths=training_sources()
    for source_path in paths:
        for batch in pq.ParquetFile(source_path).iter_batches(batch_size=512):
            for raw in batch.to_pylist():
                source_count+=1;r=normalize_row(raw)
                if normalize(r['question']) in heldout_qs or document_keys([r])&heldout_docs:continue
                eligible_counts[r['type']]+=1
                priority=int(digest([config['split_seed'],'train_pool',r['id']]),16)
                item=(-priority,r['id'],r);heap=heaps[r['type']]
                if len(heap)<config.get('split_pool_per_type',5000):heapq.heappush(heap,item)
                elif item>heap[0]:heapq.heapreplace(heap,item)
    if source_count!=90447:raise ValueError('Incomplete publisher training split')
    pool=[item[2] for heap in heaps.values() for item in heap]
    validation=choose(pool,config['validation_questions'],config['split_seed']+1,heldout,weights)
    train=choose(pool,config['train_questions'],config['split_seed']+2,heldout+validation,weights)
    pilot=choose(pool,config['pilot_questions'],config['split_seed']+3,heldout+validation+train,weights)
    parts={'train':train,'validation':validation,'heldout':heldout,'pilot':pilot}
    leakage=assert_disjoint(parts)
    directory.mkdir(parents=True,exist_ok=True);files={}
    for name,rows in parts.items():
        for suffix,value in [('gold',rows),('inputs',inference_inputs(rows))]:
            path=directory/(name+'_'+suffix+'.json');write_json(path,value);files[path.name]=file_hash(path)
    manifest={'spec':spec,'files':files,'known_legacy_eval_ids_excluded':sorted(known_ids),
              'publisher_training_rows_verified':source_count,'eligible_after_heldout_title_filter':dict(eligible_counts),
              'partitions':{name:{'n':len(rows),'types':dict(Counter(r['type'] for r in rows)),
                                  'ids':[r['id'] for r in rows],'candidate_titles':len(document_keys(rows))} for name,rows in parts.items()},
              'leakage_checks':leakage,
              'limitations':['Title-disjoint selection changes representativeness; not the standard official split protocol',
                             'Titles/question normalization do not detect all aliases, paraphrases or backbone pretraining contamination',
                             'Known legacy exclusion covers supplied 100-item Web sample; older exposure cannot be exhaustively established',
                             'HotpotQA train contexts are supplied candidates; dev fullwiki candidates have a different retrieval distribution']}
    write_json(manifest_path,manifest)
    return directory

def load_partition(directory,name,gold=False):
    if name not in ['train','validation','heldout','pilot']:raise ValueError(name)
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    filename=name+('_gold.json' if gold else '_inputs.json')
    if file_hash(directory/filename)!=manifest['files'][filename]:raise ValueError('Partition checksum mismatch')
    return json.loads((directory/filename).read_text())
