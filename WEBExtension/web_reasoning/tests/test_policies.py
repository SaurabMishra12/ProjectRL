import json,pytest

@pytest.mark.parametrize('n',[2,3,4,5])
@pytest.mark.parametrize('kind',['static','random','heuristic'])
def test_routing_is_call_matched_and_deterministic(n,kind):
    from WEBExtension.web_reasoning.policies import Metadata,select_senders,one_round_schedule
    m=tuple(Metadata(i,'partial',1,('Mira',),(),('h'+str(i),)) for i in range(n))
    k=max(1,n//2);s=select_senders(kind,m,0,k,'Mira',7,set())
    assert len(s)==len(set(s))==k and 0 not in s
    assert s==select_senders(kind,m,0,k,'Mira',7,set())
    schedule=one_round_schedule(kind,n,0,s)
    assert schedule.recipients==(0,) and len(schedule.edges)==k

def test_dense_sink_matches_complete_instar():
    from WEBExtension.web_reasoning.policies import one_round_schedule
    for n in (2,3,4,5):
        for sink in range(n):
            dense=one_round_schedule('dense',n,sink,())
            star=one_round_schedule('instar',n,sink,())
            assert tuple(e for e in dense.edges if e[1]==sink)==star.edges
            assert len(dense.edges)==n*(n-1)

def test_metadata_is_really_bounded_before_routing():
    from WEBExtension.web_reasoning.policies import metadata_frame,Metadata
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    from .test_packets import state
    m,text=metadata_frame(state(),UTF8FixtureTokenizer())
    assert len(text.encode())<=96
    assert m.entity_tags==tuple(json.loads(text)['tags'])
    assert 'Mira was born' not in text
    with pytest.raises(TypeError):Metadata(0,'partial',1,(),(),(),correct=True)
