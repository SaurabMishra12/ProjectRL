"""Select the approved grouped training pilot without loading all source atoms."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile
from WEBExtension.web_reasoning.data import seeded,from_2wiki
from WEBExtension.web_reasoning.splits import normalize
from WEBExtension.web_reasoning.preparation import prepare_run,forecast,read_aliases,file_sha
from WEBExtension.web_reasoning.storage import atomic_json

ROOT=Path('WEBExtension/web_reasoning/artifacts/2wiki_pilot_20261016')
SEED=20261016


def array_rows(stream):
    decoder=json.JSONDecoder();buffer='';position=0;started=False;eof=False
    while True:
        while position<len(buffer) and buffer[position].isspace():position+=1
        if position==len(buffer):
            if eof:raise ValueError('truncated_json_array')
            buffer='';position=0;chunk=stream.read(1024*1024);eof=not chunk;buffer+=chunk
            continue
        if not started:
            if buffer[position]!='[':raise ValueError('list_required')
            started=True;position+=1;continue
        if buffer[position]==']':return
        if buffer[position]==',':position+=1;continue
        try:row,end=decoder.raw_decode(buffer,position)
        except json.JSONDecodeError:
            if eof:raise
            chunk=stream.read(1024*1024);eof=not chunk;buffer=buffer[position:]+chunk;position=0
            continue
        yield row;position=end
        if position>1024*1024:buffer=buffer[position:];position=0


class HashReader(io.RawIOBase):
    def __init__(self,stream,hasher):self.stream=stream;self.hasher=hasher
    def readable(self):return True
    def readinto(self,buffer):
        chunk=self.stream.read(len(buffer));self.hasher.update(chunk);buffer[:len(chunk)]=chunk
        return len(chunk)


def main():
    archive=ROOT/'source/data_ids_april7.zip'
    parent=[];meta=[];seen={};population=Counter();train_hash=hashlib.sha256();excluded=Counter();original_count=0
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    with zipfile.ZipFile(archive) as z:
        with z.open('train.json') as raw:
            text=io.TextIOWrapper(io.BufferedReader(HashReader(raw,train_hash)),encoding='utf-8')
            for row in array_rows(text):
                original_count+=1
                titles=[t for t,_ in row['context']]
                if len(titles)!=len(set(titles)):
                    excluded['duplicate_source_title']+=1;continue
                if any(not isinstance(t,str) or not t or not isinstance(sentences,list) or any(not isinstance(s,str) or not s for s in sentences) for t,sentences in row['context']):
                    excluded['invalid_source_text']+=1;continue
                i=len(meta);parent.append(i);kind=row.get('type','unknown');population[kind]+=1
                meta.append((str(row['_id']),kind,i))
                markers=[('question',normalize(row['question']))]+[('title',normalize(t)) for t,_ in row['context']]
                for marker in markers:
                    if marker in seen:parent[root(i)]=root(seen[marker])
                    else:seen[marker]=i
                if len(meta)%25000==0:print('Indexed training questions',len(meta),flush=True)
        groups={}
        for item in sorted(meta):groups.setdefault(root(item[2]),[]).append(item)
        remaining=list(groups.values());seeded(SEED,'groups').shuffle(remaining)
        selected=[];counts=Counter()
        while len(selected)<20:
            eligible=[g for g in remaining if len(g)<=20-len(selected)]
            if not eligible:raise ValueError('full_training_source_groups_prevent_exact_20_question_pilot')
            def deficit(group):
                return sum(population[t]*20/len(meta)-counts[t] for _,t,_ in group)/len(group)
            group=max(eligible,key=deficit);remaining.remove(group);selected.extend(group);counts.update(t for _,t,_ in group)
        keys={key for key,_,_ in selected};rows=[]
        with z.open('train.json') as raw:
            for row in array_rows(io.TextIOWrapper(raw,encoding='utf-8')):
                if str(row['_id']) in keys:rows.append(row)
        assert len(rows)==20
        aliases=z.read('id_aliases.json');alias_path=ROOT/'source/id_aliases.json';alias_path.write_bytes(aliases)
    source=ROOT/'source/pilot_training_rows.json';atomic_json(source,rows)
    alias_descriptor={'path':str(alias_path),'sha256':file_sha(alias_path)}
    alias_map=read_aliases(alias_descriptor)
    for row in rows:
        if row.get('answer_id') and row['answer_id'] not in alias_map:raise ValueError('selected_gold_alias_missing')
        from_2wiki(row,alias_map)
    provenance={'source_url':'https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1',
                'publisher_repository':'https://github.com/Alab-NII/2wikimultihop',
                'release':'data_ids_april7.zip (corrected sentence segmentation)',
                'archive_sha256':file_sha(archive),'training_member_sha256':train_hash.hexdigest(),
                'original_training_questions':original_count,'eligible_training_questions':len(meta),'public_source_exclusions':dict(excluded),
                'training_types':dict(population),'selected_types':dict(counts),
                'source_groups':len(groups),'largest_source_group':max(map(len,groups.values())),
                'groups_with_at_most_20_questions':sum(len(g)<=20 for g in groups.values()),
                'selection':'Approved freeze_splits grouped algorithm on publisher training questions passing existing unambiguous-source validation; sources and normalized questions define components. No answers or model outcomes used for selection.',
                'eligibility_amendment':'Before any model calls, exclude duplicate source titles and invalid/empty source text rejected by the existing adapter. Preserve initial rejected candidate rows. This pilot is conditional on valid unambiguous sources, not the entire publisher population.',
                'selected_ids':sorted(keys),'selection_script_sha256':file_sha(__file__)}
    atomic_json(ROOT/'source/provenance.json',provenance)
    tokenizer_revision='b5c939de8f754692c1647ca79fbf85e8c1e70f8a'
    from dotenv import dotenv_values
    provider=dotenv_values('.env')['AZURE_OPENAI_BASE_URL']
    descriptor={'dataset':'2wiki','provenance':'publisher','train':{'path':str(source),'sha256':file_sha(source)},
                'aliases':alias_descriptor,'sample_sizes':{'pilot':20,'development':0,'evaluation':0,'scaling':0},
                'tokenizer':{'path':'WEBExtension/data/tokenizer_cache/models--openai--gpt-oss-120b/snapshots/'+tokenizer_revision,
                             'revision':tokenizer_revision,'sha256':'5d166cf76a735006c818d9bbd4e4052ec03011c243969a1b6d6697685bc77b70'},
                'provider':{'base_url':provider,'api_key_env':'AZURE_OPENAI_API_KEY'},
                'source_url':provenance['source_url'],'release_id':provenance['archive_sha256']}
    config={'mode':'REAL','experiment':'W1','team_sizes':[3],'seed':SEED,'model':'gpt-oss-120b',
            'deployment_revision':'deployment_alias:gpt-oss-120b; immutable provider snapshot not exposed',
            'tokenizer_revision':tokenizer_revision,'max_completions':240,'max_attempts':260,
            'max_input_tokens':2000000,'max_output_tokens':500000,'max_output_per_call':2048,
            'packet_cap':192,'metadata_cap':96,'replicate_id':'2wiki-pilot-20261016'}
    atomic_json(ROOT/'config.json',config);atomic_json(ROOT/'sources.json',descriptor)
    manifest=prepare_run(ROOT/'config.json',ROOT/'sources.json',ROOT/'run')
    atomic_json(ROOT/'forecast.json',forecast(json.loads(manifest.read_text())))
    print(json.dumps({'manifest':str(manifest),'source_provenance':provenance,'forecast':forecast(json.loads(manifest.read_text()))},indent=2),flush=True)


if __name__=='__main__':main()
