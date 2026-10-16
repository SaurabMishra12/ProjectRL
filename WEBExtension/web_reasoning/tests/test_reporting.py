import pytest
from .fixtures import w1_case

def test_engineering_report_cannot_be_scientific(tmp_path):
    from WEBExtension.web_reasoning.runner import run_w1
    from WEBExtension.web_reasoning.reporting import write_report
    p,a,s,b,c=w1_case(3,tmp_path/'inference');r=run_w1(p,a,s,b,c)
    report=write_report([r],{},tmp_path/'report',scientific=False)
    assert 'NO SCIENTIFIC SCORES' in report.read_text()
    with pytest.raises(ValueError):write_report([r],{},tmp_path/'report',scientific=True)

def test_learned_policy_overhead_reported_once_and_roundtrips(tmp_path):
    from dataclasses import replace
    import json
    from WEBExtension.web_reasoning.training import train_router
    from .test_training import records,settings
    from WEBExtension.web_reasoning.runner import run_w1
    from WEBExtension.web_reasoning.reporting import write_report
    from WEBExtension.web_reasoning.serialization import run_from_dict
    from dataclasses import asdict
    path=tmp_path/'router.npz';train_router(records('train'),records('validation'),settings()).save(path)
    p,a,s,b,c=w1_case(3,tmp_path/'inference');c=replace(c,router_checkpoint=str(path))
    run=run_w1(p,a,s,b,c)
    assert run_from_dict(asdict(run)).policy_metadata==run.policy_metadata
    report=write_report([run,replace(run,question_key='another')],{},tmp_path/'report')
    obj=json.loads(report.with_suffix('.json').read_text())
    assert obj['routing_policies'][0]['training_costs']=={'input_tokens':40,'output_tokens':12,'calls':4}
    assert len(obj['routing_policies'])==1
    assert obj['routing_policies'][0]['checkpoint_sha256']
