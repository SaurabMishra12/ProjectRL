import hashlib,json
from pathlib import Path
import pytest
from .fixtures import wiki_row

def prepared(tmp_path):
    from WEBExtension.web_reasoning.cli import prepare_run
    data=tmp_path/'data.json';data.write_text(json.dumps([wiki_row()]))
    config=tmp_path/'config.json';config.write_text('{}')
    source=tmp_path/'source.json';source.write_text(json.dumps({'dataset':'2wiki','provenance':'synthetic','train':{'path':str(data),'sha256':hashlib.sha256(data.read_bytes()).hexdigest()},'sample_sizes':{'pilot':1,'development':0,'evaluation':0,'scaling':0}}))
    return prepare_run(config,source,tmp_path/'run')

def test_prepare_forecast_mock_resume_and_verify_without_network(tmp_path,monkeypatch):
    import socket
    def forbidden(*a,**k):raise AssertionError('network forbidden')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    from WEBExtension.web_reasoning.cli import main,verify_run
    manifest=prepared(tmp_path)
    assert main(['forecast','--manifest',str(manifest)])==0
    assert main(['run','--manifest',str(manifest),'--mode','mock'])==0
    ledger=manifest.parent/'inference'/'calls.jsonl';before=ledger.read_bytes()
    assert main(['run','--manifest',str(manifest),'--mode','mock'])==0 and ledger.read_bytes()==before
    assert verify_run(manifest.parent)['status']=='PASS'
    assert main(['report','--run',str(manifest.parent),'--cohort','pilot','--engineering'])==0
    with pytest.raises(ValueError):main(['report','--run',str(manifest.parent),'--cohort','pilot','--scientific'])

def test_live_requires_reviewed_exact_permit_before_sdk(tmp_path,monkeypatch):
    from WEBExtension.web_reasoning.cli import main
    manifest=prepared(tmp_path)
    with pytest.raises(PermissionError):main(['run','--manifest',str(manifest),'--mode','live'])

def test_modified_inputs_rejected(tmp_path):
    from WEBExtension.web_reasoning.cli import verify_run
    manifest=prepared(tmp_path)
    (manifest.parent/'pilot_inputs.json').write_text('[]')
    with pytest.raises(ValueError):verify_run(manifest.parent)

def test_resumed_usage_summary_preserves_reference_history(tmp_path):
    from WEBExtension.web_reasoning.cli import main
    manifest=prepared(tmp_path)
    main(['run','--manifest',str(manifest),'--mode','mock'])
    before=json.loads((manifest.parent/'usage_summary.json').read_text())
    main(['run','--manifest',str(manifest),'--mode','mock'])
    after=json.loads((manifest.parent/'usage_summary.json').read_text())
    assert before['logical_references']==after['logical_references']==12

def test_provider_secret_fields_rejected_before_manifest_write(tmp_path):
    from WEBExtension.web_reasoning.preparation import prepare_run
    data=tmp_path/'d.json';data.write_text(json.dumps([wiki_row()]))
    config=tmp_path/'c.json';config.write_text('{}')
    source={'dataset':'2wiki','provenance':'synthetic','provider':{'api_key':'NEVER_SAVE_THIS'},'train':{'path':str(data),'sha256':hashlib.sha256(data.read_bytes()).hexdigest()},'sample_sizes':{'pilot':1,'development':0,'evaluation':0,'scaling':0}}
    with pytest.raises(ValueError):prepare_run(config,source,tmp_path/'bad')
    assert not (tmp_path/'bad'/'manifest.json').exists()

@pytest.mark.parametrize('experiment',['W1','W2'])
def test_all_n_manifest_workflow_resumes_without_transport(experiment,tmp_path):
    from WEBExtension.web_reasoning.preparation import prepare_run,verify_run
    from WEBExtension.web_reasoning.workflow import run_manifest
    row=wiki_row();data=tmp_path/'evaluation.json';data.write_text(json.dumps([row]))
    config=tmp_path/'config.json';config.write_text(json.dumps({'experiment':experiment,'max_completions':300,'max_attempts':350,'max_input_tokens':10000000,'max_output_tokens':1000000}))
    descriptor={'dataset':'2wiki','provenance':'synthetic','evaluation':{'path':str(data),'sha256':hashlib.sha256(data.read_bytes()).hexdigest()},'sample_sizes':{'pilot':0,'development':0,'evaluation':1,'scaling':1}}
    path=prepare_run(config,descriptor,tmp_path/'run')
    runs=run_manifest(path,cohort='scaling')
    assert [r.n_agents for r in runs]==[2,3,4,5]
    ledger=path.parent/'inference/calls.jsonl';before=ledger.read_bytes()
    run_manifest(path,cohort='scaling')
    assert ledger.read_bytes()==before
    assert verify_run(path.parent)['status']=='PASS'

def test_reports_isolate_staged_cohorts_and_nested_scaling(tmp_path):
    from WEBExtension.web_reasoning.preparation import prepare_run
    from WEBExtension.web_reasoning.workflow import run_manifest
    from WEBExtension.web_reasoning.cli import main
    def data(name,start,count):
        rows=[]
        for i in range(start,start+count):
            row=wiki_row();row['_id']=str(i);row['question']=f'Where was person {i} born?';row['context']=[[f'Person {i}',[f'Person {i} was born in Delta.']]];row['supporting_facts']=[[f'Person {i}',0]];rows.append(row)
        path=tmp_path/name;path.write_text(json.dumps(rows))
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    descriptor={'dataset':'2wiki','provenance':'synthetic','train':data('train.json',0,2),'evaluation':data('eval.json',2,2),'sample_sizes':{'pilot':1,'development':1,'evaluation':2,'scaling':1}}
    config=tmp_path/'config.json';config.write_text(json.dumps({'max_completions':100,'max_attempts':120,'max_input_tokens':10000000,'max_output_tokens':1000000}))
    path=prepare_run(config,descriptor,tmp_path/'run')
    for cohort in ('pilot','development','evaluation','scaling'):run_manifest(path,cohort=cohort)
    for cohort,count in (('pilot',1),('development',1),('evaluation',2),('scaling',4)):
        main(['report','--run',str(path.parent),'--cohort',cohort,'--engineering'])
        report=json.loads((path.parent/'reports'/cohort/'report.json').read_text())
        assert len(report['records'])==count and report['cohort']==cohort
        assert report['coverage']['missing']==0
        if cohort=='scaling':
            assert {r['question_key'] for r in report['records']}==set(json.loads(path.read_text())['scaling_keys'])
            assert {r['team_size'] for r in report['records']}=={2,3,4,5}

def test_w4_stopping_variants_use_adaptive_protocol(tmp_path,monkeypatch):
    from WEBExtension.web_reasoning.ablations import ablation_variants
    from WEBExtension.web_reasoning.config import validate_config
    from WEBExtension.web_reasoning.preparation import prepare_run,forecast
    from WEBExtension.web_reasoning.workflow import run_manifest,mock_transport
    from WEBExtension.web_reasoning.adaptive import run_adaptive
    from WEBExtension.web_reasoning.storage import digest
    initial_signatures=[]
    def capture_initial(public,initial,*args,**kwargs):
        initial_signatures.append(digest(initial))
        return run_adaptive(public,initial,*args,**kwargs)
    monkeypatch.setattr('WEBExtension.web_reasoning.workflow.run_adaptive',capture_initial)
    variants=ablation_variants(validate_config({'regime':'R','max_completions':100,'max_attempts':120,'packet_cap':1000,'max_input_tokens':10000000}))
    variants=[c for c in variants if c.execution_protocol=='adaptive']
    assert {c.stopping for c in variants}=={True,False}
    def supported(request,attempt):
        status,headers,raw=mock_transport(request,attempt)
        body=json.loads(raw['choices'][0]['message']['content'])
        if body['facts']:body.update(status='supported',answer='Delta',needs=[])
        raw['choices'][0]['message']['content']=json.dumps(body)
        return status,headers,raw
    monkeypatch.setattr('WEBExtension.web_reasoning.workflow.mock_transport',supported)
    results={}
    for c in variants:
        directory=tmp_path/str(c.stopping);directory.mkdir()
        data=directory/'data.json';data.write_text(json.dumps([wiki_row()]))
        config=directory/'config.json';config.write_text(json.dumps(c.to_dict()))
        path=prepare_run(config,{'dataset':'2wiki','provenance':'synthetic','train':{'path':str(data),'sha256':hashlib.sha256(data.read_bytes()).hexdigest()},'sample_sizes':{'pilot':1,'development':0,'evaluation':0,'scaling':0}},directory/'run')
        assert forecast(json.loads(path.read_text()))['logical_completion_ceilings']['pilot']==32
        run=run_manifest(path)[0];results[c.stopping]=len(run.arms['selected'].references)
        assert len(run.arms['selected_observable_stop'].references)==0
        assert len(run.arms['selected_always_run'].references)==3
    assert results=={True:0,False:3}
    assert len(set(initial_signatures))==1

def test_report_before_any_execution_preserves_assigned_cells(tmp_path):
    from WEBExtension.web_reasoning.cli import main
    path=prepared(tmp_path)
    main(['report','--run',str(path.parent),'--cohort','pilot','--engineering'])
    report=json.loads((path.parent/'reports/pilot/report.json').read_text())
    assert report['coverage']=={'expected':1,'saved':0,'missing':1}
    assert report['records'][0]['status']=='MISSING'

def test_interrupted_two_question_cohort_keeps_missing_cell(tmp_path,monkeypatch):
    from WEBExtension.web_reasoning.preparation import prepare_run
    from WEBExtension.web_reasoning.workflow import run_manifest,mock_transport
    from WEBExtension.web_reasoning.backend import ProviderRejected
    from WEBExtension.web_reasoning.cli import main
    first=wiki_row();second=wiki_row();second.update(_id='second',question='What is the birthplace of Mira?')
    data=tmp_path/'data.json';data.write_text(json.dumps([first,second]))
    config=tmp_path/'config.json';config.write_text(json.dumps({'max_completions':24,'max_attempts':26,'max_retries':0}))
    path=prepare_run(config,{'dataset':'2wiki','provenance':'synthetic','train':{'path':str(data),'sha256':hashlib.sha256(data.read_bytes()).hexdigest()},'sample_sizes':{'pilot':2,'development':0,'evaluation':0,'scaling':0}},tmp_path/'run')
    def reject_second(request,attempt):
        if list((path.parent/'results').glob('*.json')):return 429,{}, {'error':{'message':'fixture explicit rejection'}}
        return mock_transport(request,attempt)
    monkeypatch.setattr('WEBExtension.web_reasoning.workflow.mock_transport',reject_second)
    with pytest.raises(ProviderRejected):run_manifest(path)
    main(['report','--run',str(path.parent),'--cohort','pilot','--engineering'])
    report=json.loads((path.parent/'reports/pilot/report.json').read_text())
    assert report['coverage']=={'expected':2,'saved':1,'missing':1}
    assert len(report['records'])==2
