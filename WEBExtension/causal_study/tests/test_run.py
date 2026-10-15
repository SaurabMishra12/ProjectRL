import copy
import json
from pathlib import Path

import pytest

from WEBExtension.causal_study.run import run_question, parse_result
from WEBExtension.causal_study.design import build_manifest


def fixture():
    return {'id':'fixture-q','question':'Where is the city in the source?', 'type':'bridge',
            'answer':'GOLD_MARKER', 'supporting_facts':[['A',0],['B',0]],
            'documents':[{'title':c,'sentences':[c+' whole source fact.']} for c in 'ABCDEF']}


class CountingProvider:
    def __init__(self): self.requests=[]
    def complete(self,messages,metadata=None):
        payload=json.loads(messages[1]['content'])
        assert 'GOLD_MARKER' not in json.dumps(messages)
        assert set(payload)=={'question','evidence','previous_response','peer_messages'}
        self.requests.append((messages,metadata))
        raw={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({
            'answer':'noanswer','citations':[[payload['evidence'][0]['title'],0]]})}}],
             'usage':{'prompt_tokens':10,'completion_tokens':5}}
        return {'raw':raw,'request_hash':str(len(self.requests)), 'cache_hit':False}


def test_crossed_run_is_eighteen_calls_and_equal_updates_without_gold():
    m=build_manifest([fixture()],n=1)
    q=m['questions'][0]; b=CountingProvider()
    output=run_question(q,m['private']['labels'][q['id']],b)
    assert len(b.requests)==18
    assert set(output['cells'])=={'colocated_nocomm','colocated_comm','distributed_nocomm','distributed_comm'}
    for p in ['colocated','distributed']:
        no=output['cells'][p+'_nocomm']; yes=output['cells'][p+'_comm']
        assert len(no['responses'])==len(yes['responses'])==3
        assert no['initial_request_hashes']==yes['initial_request_hashes']
        assert no['deliveries']==[]
        assert len(yes['deliveries'])==6
        assert no['logical_calls']==yes['logical_calls']==6
        assert no['em']==yes['em']==0
    assert output['question_id']=='fixture-q'


def test_parse_checks_exactly_visible_citations_and_keeps_invalid_separately():
    messages=[{'role':'system','content':'x'},{'role':'user','content':json.dumps({
        'evidence':[{'title':'A','sentence_id':0,'text':'source'}], 'peer_messages':[]})}]
    saved={'raw':{'choices':[{'finish_reason':'stop','message':{'content':json.dumps({
        'answer':'Paris','citations':[['A',0],['B',0],['A',999]]})}}]},'request_hash':'h','cache_hit':False}
    r=parse_result(saved,messages)
    assert r['citations']==[['A',0]]
    assert r['invalid_citations']==[['B',0],['A',999]]


def test_truncated_or_malformed_success_is_never_used_as_prediction():
    for choice in [{'finish_reason':'length','message':{'content':'{}'}},
                   {'finish_reason':'stop','message':{'content':'not json'}}]:
        with pytest.raises(ValueError): parse_result({'raw':{'choices':[choice]}},[])


def test_frozen_manifest_and_labels_are_revalidated_on_resume(tmp_path,monkeypatch):
    import types
    from WEBExtension.causal_study import run
    original=run.ROOT
    web=tmp_path/'WEBExtension'; study=web/'causal_study'; study.mkdir(parents=True)
    for name in run.STUDY_FILES: (study/name).write_bytes((original/name).read_bytes())
    (web/'data').mkdir(); (web/'data/hotpot_dev_fullwiki_v1.json').write_text('[]')
    (web/'config.json').write_text(json.dumps({
        'provider':'azure_v1','model':'frozen','model_revision':'frozen',
        'base_url':'https://example.invalid','api_version':'v1','temperature':0.,
        'max_tokens':8192,'context_limit':32768}))
    rows=[]
    for i in range(8):
        row=fixture();row['id']=str(i);row['question']+=' '+str(i);rows.append(row)
    monkeypatch.setattr(run,'ROOT',study);monkeypatch.setattr(run,'WEB',web)
    monkeypatch.setattr(run,'tokenizer',lambda:types.SimpleNamespace(identity={'revision':'fake'},count=len))
    monkeypatch.setattr(run,'load_official',lambda:rows)
    monkeypatch.setattr(run,'exposure_rows',lambda:([],[]))
    manifest,_=run.freeze()
    p=study/'frozen/manifest.json';bad=copy.deepcopy(manifest)
    bad['questions'][0]['question']='silently changed';p.write_text(json.dumps(bad))
    with pytest.raises(ValueError): run.freeze()
    p.write_text(json.dumps(manifest))
    labels=study/'frozen/private_labels.json';labels.write_text('{}')
    with pytest.raises(ValueError): run.freeze()
