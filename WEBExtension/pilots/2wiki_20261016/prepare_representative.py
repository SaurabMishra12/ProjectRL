"""Outcome-blind pilot amendment after observing a giant source component."""
from collections import Counter
import heapq
import hashlib
import importlib
import io
import json
from pathlib import Path
import zipfile
from WEBExtension.web_reasoning.data import from_2wiki
from WEBExtension.web_reasoning.splits import normalize
from WEBExtension.web_reasoning.preparation import prepare_run,forecast,read_aliases,file_sha
from WEBExtension.web_reasoning.storage import atomic_json

ROOT=Path('WEBExtension/web_reasoning/artifacts/2wiki_pilot_20261016')
SEED=20261016


def main():
    previous=json.loads((ROOT/'source/provenance.json').read_text());arrays=importlib.import_module('WEBExtension.pilots.2wiki_20261016.prepare_pilot').array_rows
    counts=Counter();excluded=Counter();candidates={};total=0
    with zipfile.ZipFile(ROOT/'source/data_ids_april7.zip') as z:
        with z.open('train.json') as raw:
            for row in arrays(io.TextIOWrapper(raw,encoding='utf-8')):
                total+=1;titles=[t for t,_ in row['context']]
                if len(titles)!=len(set(titles)):excluded['duplicate_source_title']+=1;continue
                if any(not isinstance(t,str) or not t or not isinstance(sentences,list) or any(not isinstance(s,str) or not s for s in sentences) for t,sentences in row['context']):excluded['invalid_source_text']+=1;continue
                kind=row.get('type','unknown');counts[kind]+=1
                score=int(hashlib.sha256(f'{SEED}:pilot-representative:{row["_id"]}'.encode()).hexdigest(),16)
                heap=candidates.setdefault(kind,[]);item=(-score,str(row['_id']),row)
                if len(heap)<200:heapq.heappush(heap,item)
                elif score < -heap[0][0]:heapq.heapreplace(heap,item)
    quotas={k:max(1,int(20*v/sum(counts.values()))) for k,v in counts.items()}
    while sum(quotas.values())<20:
        k=max(counts,key=lambda k:(20*counts[k]/sum(counts.values())-quotas[k],k));quotas[k]+=1
    assert sum(quotas.values())==20
    rows=[];used=set();skipped=Counter()
    for kind in sorted(counts):
        chosen=0
        for _,_,row in sorted(candidates[kind],key=lambda x:(-x[0],x[1])):
            markers={('question',normalize(row['question']))}|{('title',normalize(t)) for t,_ in row['context']}
            if markers&used:skipped[kind]+=1;continue
            rows.append(row);used.update(markers);chosen+=1
            if chosen==quotas[kind]:break
        if chosen!=quotas[kind]:raise ValueError('insufficient_disjoint_pilot_candidates')
    assert len(rows)==20
    source=ROOT/'source/representative_pilot_training_rows.json';atomic_json(source,rows)
    descriptor=json.loads((ROOT/'sources.json').read_text());descriptor['train']={'path':str(source),'sha256':file_sha(source)}
    aliases=read_aliases(descriptor['aliases'])
    for row in rows:
        if row.get('answer_id') and row['answer_id'] not in aliases:raise ValueError('selected_gold_alias_missing')
        from_2wiki(row,aliases)
    provenance={**previous,'selected_types':dict(Counter(row['type'] for row in rows)),'selected_ids':sorted(str(r['_id']) for r in rows),
                'selection_script_sha256':file_sha(__file__),'original_training_questions':total,'eligible_training_questions':sum(counts.values()),
                'public_source_exclusions':dict(excluded),'overlap_candidates_skipped':dict(skipped),
                'selection':'Seeded hash ranking within type on all source-valid publisher training questions; proportional quotas with at least one per type; selected questions share no normalized article titles or question strings.',
                'sampling_amendment':'Before all inference: whole-corpus connected-component sampling places 156704/157871 eligible questions into a single component. Replace that infeasible representative small-cohort constraint for this pilot only. Do not generalize to a held-out confirmatory cohort; future splits require their own source-overlap audit.',
                'original_grouped_candidate_preserved':'source/pilot_training_rows.json and source/provenance.json',
                'model_outcomes_used_for_selection':False}
    atomic_json(ROOT/'source/representative_provenance.json',provenance)
    atomic_json(ROOT/'sources_representative.json',descriptor)
    manifest=prepare_run(ROOT/'config.json',ROOT/'sources_representative.json',ROOT/'representative_run')
    atomic_json(ROOT/'forecast_representative.json',forecast(json.loads(manifest.read_text())))
    print(json.dumps({'manifest':str(manifest),'types':provenance['selected_types'],'sampling_amendment':provenance['sampling_amendment'],'forecast':forecast(json.loads(manifest.read_text()))},indent=2),flush=True)


if __name__=='__main__':main()
