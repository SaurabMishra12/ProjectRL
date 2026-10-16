from .fixtures import w1_case

def test_scaling_preserves_union_and_verified_reuse(tmp_path):
    from WEBExtension.web_reasoning.scaling import run_scaling,theoretical_reference
    p,a,s,b,c=w1_case(3,tmp_path/'source')
    def factory(n,config):return w1_case(n,tmp_path/str(n))[3]
    runs=run_scaling([p],[p.key],factory,c)
    assert [r.n_agents for r in runs]==[2,3,4,5]
    assert all({a.atom_id for state in r.initial for a in state.visible_atoms}=={a.atom_id for source in p.sources for a in source.atoms} for r in runs)
    again=run_scaling([p],[p.key],factory,c,existing_runs={(r.question_key,r.n_agents):r for r in runs})
    assert runs==again
    assert theoretical_reference(5,3)['dense_deliveries']==60

def test_estimated_or_different_contract_cannot_be_reused(tmp_path):
    from dataclasses import replace
    import pytest
    from WEBExtension.web_reasoning.scaling import run_scaling
    p,a,s,b,c=w1_case(3,tmp_path/'source')
    c=replace(c,team_sizes=(3,))
    def factory(n,config):return w1_case(n,tmp_path/'run')[3]
    run=run_scaling([p],[p.key],factory,c)[0]
    with pytest.raises(ValueError):run_scaling([p],[p.key],factory,c,existing_runs={(p.key,3):replace(run,contract_hash='estimated')})
