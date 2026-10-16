def wiki_row():
    return {"_id":"fixture-1","question":"Where was Mira born?","context":[["Mira",["Mira was born in Delta."]]],"answer":"Delta","supporting_facts":[["Mira",0]],"evidences":[["Mira","birthplace","Delta"]],"type":"inference"}

def finqa_row():
    return {"id":"fixture-report-1","pre_text":["Amounts are in millions."],"post_text":[],"table":[["","2020","2019"],["Revenue","12","10"]],"qa":{"question":"What was the revenue increase?","exe_ans":2,"program":"subtract(12, 10)","gold_inds":{"table_1":"Revenue"}}}

def proposal_body(atom=None,status='partial',answer=None):
    return {'status':status,'answer':answer,'facts':([] if atom is None else [{'fact_id':atom.atom_id,'document_id':atom.document_id,'sentence_id':atom.sentence_id,'quote':atom.text,'claim':'Observed fact'}]),'entity_tags':['Mira'],'needs':[],'derivation':[],'conflicts':[]}

def backend_config(**changes):
    from WEBExtension.web_reasoning.config import validate_config
    return validate_config(changes)

def w1_case(n_agents,directory,with_needs=False):
    import json
    from WEBExtension.web_reasoning.data import from_2wiki,allocate
    from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer
    from WEBExtension.web_reasoning.backend import DurableBackend
    r=wiki_row();r['context'] += [[f'Doc{i}',[f'Mira has attribute {i}.']] for i in range(5)]
    p=from_2wiki(r,{}).public;t=UTF8FixtureTokenizer()
    def send(request,attempt):
        payload=json.loads(request['messages'][-1]['content']);atoms=payload['visible_sources']
        b=proposal_body(None,'partial' if atoms else 'no_evidence')
        b['needs']=[{'entity':'Mira','relation':'birthplace'}] if with_needs and atoms else []
        b['facts']=[{'fact_id':a['atom_id'],'document_id':a['document_id'],'sentence_id':a['sentence_id'],'quote':a['text'],'claim':'Observed fact'} for a in atoms[:4]]
        if payload['output_schema']['properties'].get('program') is not None:b.update(program=[],operand_sources=[])
        if set(payload['output_schema']['properties'])=={'answer','citations'}:b={'answer':'noanswer','citations':[[a['title'],a['sentence_id']] for a in atoms[:4]]}
        return 200,{}, {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(b)}}],'usage':{'prompt_tokens':t.count(json.dumps(request['messages'])),'completion_tokens':t.count(json.dumps(b))}}
    c=backend_config(packet_cap=1000,metadata_cap=96,max_completions=1000,max_attempts=1100,max_input_tokens=10000000,max_output_tokens=3000000)
    return p,allocate(p,n_agents,'P',c.seed,t.count),0,DurableBackend(directory,c,send,t),c
