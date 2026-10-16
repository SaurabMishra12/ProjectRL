import copy,hashlib,json
import pytest
from .fixtures import wiki_row,finqa_row

def test_public_inputs_exclude_gold():
    from WEBExtension.web_reasoning.data import from_2wiki,from_finqa
    from WEBExtension.web_reasoning.contracts import runtime_object
    r=wiki_row(); before=copy.deepcopy(r); e=from_2wiki(r,{})
    assert r==before and e.targets.answer=='Delta'
    assert set(runtime_object(e.public))=={'question','visible_sources'}
    f=from_finqa(finqa_row()); text=' '.join(a.text for s in f.public.sources for a in s.atoms)
    assert '2020' in text and 'millions' in text and f.targets.program
    assert 'gold_inds' not in str(runtime_object(f.public))

@pytest.mark.parametrize('bad',['duplicate','index','aliases'])
def test_invalid_source_identity(bad):
    from WEBExtension.web_reasoning.data import from_2wiki
    r=wiki_row(); aliases={}
    if bad=='duplicate':r['context']*=2
    if bad=='index':r['supporting_facts'][0][1]=20
    if bad=='aliases':r['answer_id']='Q1';aliases=None
    with pytest.raises(ValueError):from_2wiki(r,aliases)

@pytest.mark.parametrize('n',[2,3,4,5])
def test_allocate_preserves_order_and_union(n):
    from WEBExtension.web_reasoning.data import from_2wiki,allocate
    r=wiki_row();r['context'] += [[str(i),[f'Fact {i}.']] for i in range(8)]
    p=from_2wiki(r,{}).public
    x=allocate(p,n,'P',7,len);y=allocate(p,n,'R',7,len)
    original=tuple(a for s in p.sources for a in s.atoms)
    assert sorted(a.atom_id for xs in x for a in xs)==sorted(a.atom_id for a in original)
    assert all(xs==original for xs in y) and x==allocate(p,n,'P',7,len)

def test_hash_checked_before_load(tmp_path):
    from WEBExtension.web_reasoning.data import load_verified
    p=tmp_path/'data.json';p.write_text(json.dumps([wiki_row()]))
    with pytest.raises(ValueError):load_verified(p,'0'*64,'2wiki',{})
    assert len(load_verified(p,hashlib.sha256(p.read_bytes()).hexdigest(),'2wiki',{}))==1
