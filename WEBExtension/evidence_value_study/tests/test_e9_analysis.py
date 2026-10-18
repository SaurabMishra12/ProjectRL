import importlib
import pytest

MOD='WEBExtension.evidence_value_study.analysis.statistics'

def module():
    assert importlib.util.find_spec(MOD) is not None, 'E9 scoring and analysis missing'
    return importlib.import_module(MOD)

def test_publisher_normalization_aliases_and_empty_answer():
    m=module()
    assert m.answer_score('The New-York!', ['NewYork'])=={'em':1.,'f1':1.}
    assert m.answer_score('NYC',['New York','NYC'])['em']==1.
    assert m.answer_score('', ['Paris'])['f1']==0
    assert m.answer_score('',[''])['f1']==1

def test_statistical_pairing_and_interaction_formula():
    m=module()
    rows=[{'question_id':str(i),'A':.2,'B':.3,'C':.1,'D':.5} for i in range(30)]
    r=m.factorial(rows)
    assert r['contrasts']['interaction']['mean']==pytest.approx(.3)
    assert r['contrasts']['complete_peer_effect']['mean']==pytest.approx(.1)
    assert r['contrasts']['missing_peer_effect']['mean']==pytest.approx(.4)
    with pytest.raises(ValueError):m.factorial(rows+[rows[0]])
    with pytest.raises(ValueError):m.factorial([{**rows[0],'D':None}])

def test_mock_results_rejected_as_scientific_outcomes():
    m=module()
    with pytest.raises(ValueError,match='MOCK'):m.analyze_records([{'mode':'MOCK'}],{},[])

def test_missing_cells_preserved_and_bounds_not_complete_case():
    m=module()
    rows=[{'question_id':'q','mode':'REAL','cells':{k:{'status':'MALFORMED','answer':None} for k in ['complete_private','complete_peer','missing_private','missing_peer']}}]
    result=m.summarize_cells(rows,{'q':['X']},['q'])
    assert result['status']=='INSUFFICIENT_EVIDENCE'
    assert result['interaction_identification_bounds']==[-2.,2.]
    assert result['primary_effect'] is None and result['assigned_questions']==1

def test_power_sensitivity_uses_assumptions_and_sample_units():
    m=module()
    grid=m.power_grid()
    assert {r['n_questions'] for r in grid}=={200,300,600}
    assert {r['difference_f1_points'] for r in grid}=={2,5,10}
    assert all(r['nominal_completions']==11*r['n_questions'] for r in grid)
    assert all(r['basis']=='ASSUMED_CONTRAST_SD_NO_E9_PILOT' for r in grid)

def test_scientific_export_requires_explicit_mode_and_qualified_manifest():
    m=module()
    with pytest.raises(ValueError,match='provenance'):
        m.analyze_records([{'question_id':'q'}],{'q':['X']},['q'])
    with pytest.raises(ValueError,match='qualification'):
        m.analyze_records([{'question_id':'q','mode':'REAL'}],{'q':['X']},['q'])

def test_four_cell_EM_and_full_evidence_baseline_are_reported():
    m=module()
    observed={'status':'VALID','answer':'X'}
    rows=[{'question_id':str(i),'cells':{k:dict(observed) for k in m.CELLS},'baseline':dict(observed)} for i in range(2)]
    r=m.summarize_cells(rows,{'0':['X'],'1':['X']},['0','1'])
    assert all(v=={'em':1.,'f1':1.} for v in r['four_cell_f1_em'].values())
    assert r['full_evidence_RAG']=={'status':'OBSERVED','n':2,'em':1.,'f1':1.}
