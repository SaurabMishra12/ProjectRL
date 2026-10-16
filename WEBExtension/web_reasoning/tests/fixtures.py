def wiki_row():
    return {"_id":"fixture-1","question":"Where was Mira born?","context":[["Mira",["Mira was born in Delta."]]],"answer":"Delta","supporting_facts":[["Mira",0]],"evidences":[["Mira","birthplace","Delta"]],"type":"inference"}

def finqa_row():
    return {"id":"fixture-report-1","pre_text":["Amounts are in millions."],"post_text":[],"table":[["","2020","2019"],["Revenue","12","10"]],"qa":{"question":"What was the revenue increase?","exe_ans":2,"program":"subtract(12, 10)","gold_inds":{"table_1":"Revenue"}}}

def proposal_body(atom=None,status='partial',answer=None):
    return {'status':status,'answer':answer,'facts':([] if atom is None else [{'fact_id':atom.atom_id,'document_id':atom.document_id,'sentence_id':atom.sentence_id,'quote':atom.text,'claim':'Observed fact'}]),'entity_tags':['Mira'],'needs':[],'derivation':[],'conflicts':[]}

def backend_config(**changes):
    from WEBExtension.web_reasoning.config import validate_config
    return validate_config(changes)
