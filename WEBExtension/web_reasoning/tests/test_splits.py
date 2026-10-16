import pytest
from .fixtures import wiki_row

def test_splits_group_sources_and_are_input_order_independent():
    from WEBExtension.web_reasoning.data import from_2wiki
    from WEBExtension.web_reasoning.splits import freeze_splits,balanced_sinks
    rows=[]
    for i in range(8):
        r=wiki_row();r['_id']=str(i);r['question']=f'Question {i}?';r['context'][0][0]=f'Title {i}';r['supporting_facts'][0][0]=f'Title {i}'
        rows.append(from_2wiki(r,{}))
    a=freeze_splits(rows,{'train':4,'test':4},7)
    b=freeze_splits(list(reversed(rows)),{'train':4,'test':4},7)
    assert a==b and len({e.public.key for xs in a.values() for e in xs})==8
    sinks=balanced_sinks([e.public.key for e in rows],3,7)
    assert max(list(sinks.values()).count(i) for i in range(3))-min(list(sinks.values()).count(i) for i in range(3))<=1

def test_group_cannot_be_silently_split():
    from WEBExtension.web_reasoning.data import from_2wiki
    from WEBExtension.web_reasoning.splits import freeze_splits
    a=wiki_row();b=wiki_row();b['_id']='2';b['question']='Another question?'
    with pytest.raises(ValueError):freeze_splits([from_2wiki(a,{}),from_2wiki(b,{})],{'a':1,'b':1},1)
