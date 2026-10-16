import pytest

def test_exact_test_and_question_pairing():
    from WEBExtension.web_reasoning.statistics import exact_mcnemar,paired_summary
    assert exact_mcnemar(5,0)==.0625 and exact_mcnemar(0,0)==1
    rows=[{'question_key':str(i),'mode':'REAL','arms':{'a':{'em':100},'b':{'em':0}}} for i in range(4)]
    s=paired_summary(rows,('a','b'),draws=100)
    assert s['effect']==100 and s['ci95']==[100,100] and s['n_questions']==4
    rows[0]['mode']='MOCK'
    with pytest.raises(ValueError):paired_summary(rows,('a','b'))

def test_missing_cells_have_bounds_not_fabricated_answers():
    from WEBExtension.web_reasoning.statistics import paired_summary
    r=paired_summary([{'question_key':'q','mode':'REAL','arms':{'a':{'em':100},'b':{'em':None}}}],('a','b'),draws=20)
    assert r['status']=='INCOMPLETE' and r['effect'] is None and r['bounds']==[0,100]
