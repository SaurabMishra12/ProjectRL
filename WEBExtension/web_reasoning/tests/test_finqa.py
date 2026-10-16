import pytest
from .fixtures import finqa_row

def test_allowlist_references_percent_and_errors():
    from WEBExtension.web_reasoning.finqa import execute_program
    assert not execute_program(['__import__(','os','0',')','EOF'],()).valid
    assert execute_program(['subtract(','12','10',')','EOF'],()).value==2
    assert execute_program(['divide(','25%','const_100',')','EOF'],()).value==.0025
    assert not execute_program(['divide(','1','0',')','EOF'],()).valid
    assert not execute_program(['add(','#0','1',')','EOF'],()).valid
    assert not execute_program(['exp(','100','1000',')','EOF'],()).valid
    assert not execute_program(['add(','nan','1',')','EOF'],()).valid

def test_table_operations_and_official_parity():
    from WEBExtension.web_reasoning.finqa import execute_program
    from WEBExtension.web_reasoning.data import from_finqa
    from WEBExtension.web_reasoning.evaluators import load_reference
    row=finqa_row();public=from_finqa(row).public
    atoms=tuple(a for s in public.sources for a in s.atoms)
    program=['table_sum(','Revenue','none',')','EOF']
    ours=execute_program(program,atoms)
    official=load_reference('finqa').eval_program(program,row['table'])
    assert ours.valid and official[0]==0 and ours.value==official[1]==22

def test_operand_provenance_and_program_scoring():
    from WEBExtension.web_reasoning.finqa import validate_operands
    from WEBExtension.web_reasoning.evaluators import OfficialEvaluator
    from WEBExtension.web_reasoning.data import from_finqa
    from WEBExtension.web_reasoning.contracts import Proposal,Fact,Outcome
    import json
    e=from_finqa(finqa_row());a=e.public.sources[-1].atoms[0];program=('subtract(','12','10',')','EOF')
    mapping=tuple(json.dumps({'operand':v,'fact_ids':[a.atom_id]}) for v in ('12','10'))
    validate_operands(program,mapping,(a,))
    with pytest.raises(ValueError):validate_operands(program,(),(a,))
    p=Proposal('supported','2',(Fact(a.atom_id,a.text,'Revenue increased'),),program=program,operand_sources=mapping)
    score=OfficialEvaluator('finqa').score(e.public.key,Outcome('valid',p),e.targets,public=e.public,visible_atoms=(a,))
    assert score['execution_accuracy']==100 and score['program_accuracy']==100

def test_missing_units_and_financial_negatives():
    from dataclasses import replace
    from WEBExtension.web_reasoning.data import from_finqa
    from WEBExtension.web_reasoning.finqa import validate_operands,execute_program
    import json
    row=finqa_row();row['table'][1][1]='(12)';a=from_finqa(row).public.sources[-1].atoms[0]
    validate_operands(['add(','-12','const_1',')','EOF'],[{'operand':'-12','fact_ids':[a.atom_id]}],(a,))
    with pytest.raises(ValueError):validate_operands(['add(','-12','const_1',')','EOF'],[{'operand':'-12','fact_ids':[a.atom_id]}],(replace(a,context=()),))
    p=['subtract(','1,200','200',')','divide(','#0','const_100',')','EOF']
    assert execute_program(p,()).value==10

@pytest.mark.parametrize('status,answer',[('supported','22'),('partial',None)])
def test_hidden_table_or_abstaining_program_cannot_score(status,answer):
    import json
    from .fixtures import proposal_body
    from WEBExtension.web_reasoning.data import from_finqa
    from WEBExtension.web_reasoning.parsing import parse_outcome
    from WEBExtension.web_reasoning.evaluators import OfficialEvaluator
    row=finqa_row();row['qa'].update(exe_ans=22,program='table_sum(Revenue, none)')
    e=from_finqa(row);visible=e.public.sources[0].atoms[:1]
    body=proposal_body(visible[0],status,answer)
    body.update(program=['table_sum(','Revenue','none',')','EOF'],operand_sources=[])
    outcome=parse_outcome({'choices':[{'finish_reason':'stop','message':{'content':json.dumps(body)}}]},visible,'finqa')
    assert outcome.kind=='invalid'
    assert OfficialEvaluator('finqa').score(e.public.key,outcome,e.targets,public=e.public)['status']=='INCOMPLETE'

def test_scoring_requires_endpoint_visibility_and_supported_status():
    from WEBExtension.web_reasoning.data import from_finqa
    from WEBExtension.web_reasoning.contracts import Proposal,Fact,Outcome
    from WEBExtension.web_reasoning.evaluators import OfficialEvaluator
    e=from_finqa(finqa_row());a=e.public.sources[-1].atoms[0]
    p=Proposal('supported','2',(Fact(a.atom_id,a.text,'Observed'),),program=('subtract(','12','10',')','EOF'))
    ev=OfficialEvaluator('finqa')
    assert ev.score(e.public.key,Outcome('valid',p),e.targets,public=e.public)['status']=='INCOMPLETE'
    partial=Proposal('partial',None,p.facts,program=p.program)
    assert ev.score(e.public.key,Outcome('valid',partial),e.targets,public=e.public)['execution_accuracy']==0

def test_mapped_visible_table_program_parses_and_scores():
    import json
    from .fixtures import proposal_body
    from WEBExtension.web_reasoning.data import from_finqa
    from WEBExtension.web_reasoning.parsing import parse_outcome
    from WEBExtension.web_reasoning.evaluators import OfficialEvaluator
    row=finqa_row();row['qa'].update(exe_ans=22,program='table_sum(Revenue, none)')
    e=from_finqa(row);atom=e.public.sources[-1].atoms[0]
    body=proposal_body(atom,'supported','22');body.update(program=['table_sum(','Revenue','none',')','EOF'],operand_sources=[{'operand':'Revenue','fact_ids':[atom.atom_id]}])
    raw={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(body)}}]}
    outcome=parse_outcome(raw,(atom,),'finqa');assert outcome.kind=='valid'
    score=OfficialEvaluator('finqa').score(e.public.key,outcome,e.targets,public=e.public,visible_atoms=(atom,))
    assert score['execution_accuracy']==score['program_accuracy']==100
    body.update(program=['subtract(','12','10',')','EOF'],operand_sources=[])
    raw['choices'][0]['message']['content']=json.dumps(body)
    assert parse_outcome(raw,(atom,),'finqa').kind=='invalid'
