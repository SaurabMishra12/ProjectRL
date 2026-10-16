"""Publisher-shaped adapters; source text and evaluation targets take separate paths."""
import hashlib
import json
import random
from pathlib import Path
from .contracts import Atom, Source, PublicExample, PrivateTargets, Example


def seeded(seed, *parts):
    raw=json.dumps([seed,*parts],ensure_ascii=False,sort_keys=True).encode()
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest(),'big'))


def from_2wiki(row, aliases):
    if row.get('answer_id') and aliases is None:
        raise ValueError('compatible_alias_file_required')
    sources=[]; titles=set()
    for index,(title,sentences) in enumerate(row['context']):
        if not isinstance(title,str) or not title or title in titles or not isinstance(sentences,list):
            raise ValueError('ambiguous_source_title')
        titles.add(title); doc=f'd{index}'
        atoms=tuple(Atom(f'{doc}:{j}',doc,j,'text',text,title) for j,text in enumerate(sentences))
        sources.append(Source(doc,title,atoms))
    locators={(s.title,a.sentence_id) for s in sources for a in s.atoms}
    support=tuple(tuple(x) for x in row.get('supporting_facts',()))
    if any(len(x)!=2 or type(x[1]) is not int or x not in locators for x in support):
        raise ValueError('invalid_support_locator')
    alias=(aliases or {}).get(row.get('answer_id'),())
    if isinstance(alias,dict):alias=tuple(alias.get('aliases',()))+tuple(alias.get('demonyms',()))
    if not isinstance(alias,(list,tuple)) or any(not isinstance(x,str) for x in alias):
        raise ValueError('invalid_alias_record')
    return Example(PublicExample(str(row['_id']),'2wiki',row['question'],tuple(sources)),
                   PrivateTargets(str(row['answer']),tuple(alias),support,'',(('type',row.get('type','unknown')),)))


def from_finqa(row):
    qa=row['qa']; table=row['table']; sources=[]
    if not table or not isinstance(table,list) or any(not isinstance(r,list) or len(r)!=len(table[0]) for r in table):
        raise ValueError('invalid_table_shape')
    for name in ('pre_text','post_text'):
        texts=row.get(name,[])
        atoms=tuple(Atom(f'{name}:{j}',name,j,'text',text,name) for j,text in enumerate(texts) if text.strip())
        if atoms:sources.append(Source(name,name,atoms))
    # Unit/footnote interpretation is deliberately copied with every intact row.
    # This is public report text only; it contains no qa/retrieval annotations.
    unit_text='\n'.join(row.get('pre_text',[])+row.get('post_text',[]))
    header=table[0]; atoms=[]
    for i,values in enumerate(table[1:],1):
        text=json.dumps({'header':header,'row':values,'report_notes':unit_text},ensure_ascii=False,separators=(',',':'))
        atoms.append(Atom(f'table:{i}','table',i,'table_row',text,'table',
                          (('header',json.dumps(header)),('row',json.dumps(values)),('report_notes',unit_text))))
    sources.append(Source('table','table',tuple(atoms)))
    return Example(PublicExample(str(row['id']),'finqa',qa['question'],tuple(sources)),
                   PrivateTargets(str(qa.get('exe_ans','')),(),(),qa.get('program',''),(('report',str(row['id']).rsplit('-',1)[0]),)))


def load_verified(path, expected_sha256, dataset, aliases=None):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('dataset_hash_mismatch')
    rows=json.loads(raw)
    if not isinstance(rows,list) or dataset not in ('2wiki','finqa'):
        raise ValueError('unsupported_dataset')
    result=tuple(from_2wiki(r,aliases) if dataset=='2wiki' else from_finqa(r) for r in rows)
    if len({e.public.key for e in result})!=len(result):raise ValueError('duplicate_question_id')
    return result


def allocate(public,n_agents,regime,seed,token_count):
    if n_agents not in (2,3,4,5) or regime not in ('P','R'):raise ValueError('invalid_placement')
    if regime=='R':return tuple(tuple(a for s in public.sources for a in s.atoms) for _ in range(n_agents))
    order=list(range(len(public.sources))); tie=list(range(n_agents)); rng=seeded(seed,public.key,'allocation')
    rng.shuffle(order);rng.shuffle(tie);rank={node:i for i,node in enumerate(tie)}
    loads=[0]*n_agents; assigned=[[] for _ in loads]
    for index in order:
        node=min(range(n_agents),key=lambda n:(loads[n],rank[n]))
        assigned[node].append(index)
        loads[node]+=sum(token_count(a.text) for a in public.sources[index].atoms)
    return tuple(tuple(a for index in sorted(indices) for a in public.sources[index].atoms) for indices in assigned)
