import json,pytest
from .fixtures import w1_case

@pytest.mark.parametrize('n',[2,3,4,5])
def test_actual_slots_and_frozen_initial_states(n,tmp_path):
    from WEBExtension.web_reasoning.runner import run_w1
    from WEBExtension.web_reasoning.storage import digest
    p,a,s,b,c=w1_case(n,tmp_path)
    r=run_w1(p,a,s,b,c)
    assert len(r.logical_refs)==2*n+6 and len(r.arms)==8 and r.mode=='MOCK'
    assert r.initial==r.arms['nocomm'].states
    for kind in ['static','random','selected','private','instar']:
        assert len(r.arms[kind].references)==1
    assert len(r.arms['dense'].references)==n
    assert r.arms['instar'].references[0]['key']==next(x['key'] for x in r.arms['dense'].references if x['node']==s)
    assert r.arms['private'].states[s].visible_atoms==r.initial[s].visible_atoms
    count=b.summary()['physical_attempts'];r2=run_w1(p,a,s,b,c)
    assert b.summary()['physical_attempts']==count
    assert digest(r.initial)==digest(r2.initial)

def test_one_copy_of_received_source_in_model_input(tmp_path):
    from WEBExtension.web_reasoning.runner import run_w1
    p,a,s,b,c=w1_case(3,tmp_path);r=run_w1(p,a,s,b,c)
    key=r.arms['instar'].references[0]['key'];request=json.loads((tmp_path/'requests'/(key+'.json')).read_text())
    payload=json.loads(request['messages'][-1]['content'])
    assert len(payload['visible_sources'])==len({x['atom_id'] for x in payload['visible_sources']})
    assert all('source' not in x for packet in payload['incoming_packets'] for x in packet['facts'])
