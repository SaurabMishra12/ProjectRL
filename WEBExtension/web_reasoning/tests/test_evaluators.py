import pytest
from .fixtures import wiki_row

def test_official_answer_aliases_and_support_parity():
    from WEBExtension.web_reasoning.evaluators import OfficialEvaluator,load_reference
    from WEBExtension.web_reasoning.data import from_2wiki
    from WEBExtension.web_reasoning.contracts import Outcome,Proposal,Fact
    from dataclasses import replace
    e=from_2wiki(wiki_row(),{});a=e.public.sources[0].atoms[0]
    outcome=Outcome('valid',Proposal('supported','The Delta',(Fact(a.atom_id,a.text,'claim'),)))
    targets=replace(e.targets,aliases=('The Delta',))
    evaluator=OfficialEvaluator('2wiki');score=evaluator.score(e.public.key,outcome,targets,public=e.public)
    module=load_reference('2wiki')
    assert score['em']==100*module.exact_match_score('The Delta','Delta')==100
    assert score['support_f1']==100
    assert evaluator.score('q',Outcome('invalid'),targets)['em'] is None
    assert module.eval_answer('yes indeed','yes')[1]==0

def test_official_source_integrity_rechecked_after_module_cache(tmp_path):
    import json,shutil
    from WEBExtension.web_reasoning.evaluators import load_reference,REFERENCES
    manifest=tmp_path/'sources.json';shutil.copy2(REFERENCES/'sources.json',manifest)
    source=tmp_path/'2wiki_official.py';shutil.copy2(REFERENCES/'2wiki_official.py',source)
    load_reference('2wiki',str(manifest));source.write_text(source.read_text()+'\n# changed\n')
    with pytest.raises(ValueError):load_reference('2wiki',str(manifest))
