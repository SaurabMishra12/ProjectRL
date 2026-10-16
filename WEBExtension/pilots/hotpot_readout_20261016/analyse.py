"""Zero-inference E1/E5 readouts and exact one-round Hub-1 request audit.

Run from repository root with python -m WEBExtension.pilots.hotpot_readout_20261016.analyse.
Uses existing pure readout functions; no provider, credentials, or tokenizer loads.
"""
from __future__ import annotations
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import gzip
import json
import os
from pathlib import Path

from WEBExtension.causal_study.readout_reassessment import analysis as prior

ROOT = Path(__file__).resolve().parent
WEB = ROOT.parents[1]
E5 = WEB / 'artifacts/controller_study_a0b3768357484b45'
E1 = WEB / 'artifacts/run_f6aa61bac2c3dfae'
RULES = ('original', 'abstention_aware', 'sink1')
INSTAR = [[0, 1], [2, 1]]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def pure_definitions(path, names, env):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)


def main():
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    protected = {}
    def protect(path):
        path = Path(path); value = prior.sha(path)
        if str(path) in protected: assert protected[str(path)] == value
        protected[str(path)] = value
        return path
    manifest = json.loads(protect(E5 / 'manifest.json').read_text())
    for name in ('protocol.py', 'experiments.py', 'controller_study.py', 'backend.py'):
        assert prior.sha(protect(WEB / name)) == manifest['sources'][name], name
    protect(Path(prior.__file__))
    for path in (E5/'inference/calls.jsonl', E1/'calls.jsonl'):
        protect(path)
    gold_path = protect(E1 / 'sample_gold.json')
    gold = {r['id']: r for r in json.loads(gold_path.read_text())}
    assert len(gold) == 300
    inputs = {}; raw_paths = {}
    for exp, folder in [('E5', E5), ('E1', E1)]:
        path = protect(folder / (exp+'.jsonl')); raw_paths[exp] = path
        status = json.loads(protect(folder/(exp+'_status.json')).read_text())
        assert status['status'] == 'COMPLETE' and status['sha256'] == prior.sha(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 300 and len({r['id'] for r in rows}) == 300
        assert {r['id'] for r in rows} == set(gold)
        inputs[exp] = rows
    official, evaluator_path = prior.official_evaluator(); protect(evaluator_path)
    cache = {}; counts = Counter()
    def saved_response(exp, response, node):
        key = response['call_key']; location = (E5/'inference/responses' if exp=='E5' else E1/'responses')/(key+'.json')
        if str(location) not in cache:
            saved = json.loads(protect(location).read_text())
            assert saved['content_hash'] == digest({k:v for k,v in saved.items() if k!='content_hash'})
            assert saved['key'] == digest({k:saved[k] for k in ('identity', 'request', 'tokenizer')}) == key
            cache[str(location)] = saved
        saved = cache[str(location)]
        assert saved['metadata']['node'] == node
        content = saved['response']['choices'][0]['message'].get('content') or ''
        assert content == response['raw_text']
        counts['response_references_verified'] += 1
        return saved
    def score(answer, reference):
        result = prior.score_answer(answer, reference)
        assert result['em'] == 100*official['exact_match_score'](answer, reference)
        assert abs(result['f1'] - 100*official['f1_score'](answer, reference)[0]) < 1e-8
        counts['official_answer_scores_checked'] += 1
        return result
    summaries = {}; question_rows = []; by_arm = {}
    def add_condition(exp, row, arm, responses, reference):
        outcomes = {}
        for rule in RULES:
            answer = responses[1]['answer'] if rule=='sink1' else prior.aggregate(responses, rule)
            outcomes[rule] = {'answer':answer, **score(answer, reference)}
            by_arm.setdefault((exp,arm,rule), {})[row['id']] = outcomes[rule]
        question_rows.append({'experiment':exp,'id':row['id'],'arm':arm,'readouts':outcomes,
                              'call_keys':[r['call_key'] for r in responses]})
    for exp, rows in inputs.items():
        arms = set(rows[0]['conditions'])
        for row in rows:
            reference = gold[row['id']]['answer']
            assert set(row['conditions']) == arms
            for arm, condition in row['conditions'].items():
                rs = condition['responses']; assert len(rs)==3
                for node,r in enumerate(rs): saved_response(exp,r,node)
                reconstructed = prior.aggregate(rs,'original')
                assert reconstructed == condition['answer']
                original = score(reconstructed,reference)
                assert all(abs(original[m]-condition[m])<1e-8 for m in ('em','f1'))
                add_condition(exp,row,arm,rs,reference)

    # Execute only the inspected, hash-matching pure prompt/update definitions.
    # The capture object has no network implementation and returns saved outputs.
    env = {'json':json, 'canonical':canonical}
    pure_definitions(WEB/'protocol.py', {'evidence_lines','build_prompt','delivered_facts'}, env)
    pure_definitions(WEB/'experiments.py', {'visible_documents','update'}, env)
    e1_by_id = {r['id']:r for r in inputs['E1']}
    hub_rows = []; projection_costs = {}; subset = Counter()
    for row in inputs['E5']:
        qid=row['id']; reference=gold[qid]['answer']
        for regime in ('P','R'):
            dense=row['conditions'][regime+'_Broadcast']; initial=row['conditions'][regime+'_NoComm']['responses']
            assert dense['action']['quota']==256
            if regime=='P':
                policy=row['conditions']['P_Learned_seed31']['action']
                assert sorted(policy['edges'])==INSTAR and policy['quota']==128
            state={**e1_by_id[qid]['states'][regime], 'responses':initial}
            for node,r in enumerate(initial):
                saved=saved_response('E5',r,node)
                assert saved['request']['messages']==env['build_prompt'](gold[qid]['question'],state['roles'][node],state['assignment'][node],None,[])
            selected={}
            for delivery in dense['deliveries']:
                sender=delivery['sender']; content={k:v for k,v in delivery.items() if k not in ('sender','recipient','facts')}
                if sender in selected: assert selected[sender]==content
                selected[sender]=content
            assert sorted(dense['action']['edges'])==[[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]]
            class Capture:
                def __init__(self): self.calls=[]
                def complete(self,messages,allowed_documents,metadata):
                    node=metadata['node'];self.calls.append((node,messages,allowed_documents))
                    return dense['responses'][node]
            capture=Capture()
            projected,deliveries=env['update']({'id':qid,'question':gold[qid]['question']},state,INSTAR,selected,capture,'offline_capture')
            assert len(capture.calls)==1 and capture.calls[0][0]==1
            hub_saved=saved_response('E5',dense['responses'][1],1)
            assert capture.calls[0][1]==hub_saved['request']['messages']
            key=digest({'identity':hub_saved['identity'],'request':{**hub_saved['request'],'messages':capture.calls[0][1]},'tokenizer':hub_saved['tokenizer']})
            assert key==dense['responses'][1]['call_key']
            # Local allowed-source sets also match the dense topology.
            dense_capture=Capture()
            env['update']({'id':qid,'question':gold[qid]['question']},state,dense['action']['edges'],selected,dense_capture,'offline_capture')
            assert next(c for c in dense_capture.calls if c[0]==1)==capture.calls[0]
            add_condition('E5_projection',row,regime+'_InStar256_reused',projected,reference)
            pc=projection_costs.setdefault(regime,{'dense_edges':0,'instar_edges':0,'dense_recipient_calls':0,'instar_recipient_calls':0,'dense_payload_tokens':0,'instar_payload_tokens':0,'dense_known_model_tokens':0,'instar_known_model_tokens':0,'dense_missing_usage_references':0,'instar_missing_usage_references':0})
            for prefix,responses,ds in [('dense',initial+dense['responses'],dense['deliveries']),('instar',initial+[projected[1]],deliveries)]:
                pc[prefix+'_edges']+=len(ds);pc[prefix+'_recipient_calls']+=3 if prefix=='dense' else 1
                pc[prefix+'_payload_tokens']+=sum(d['tokens'] for d in ds)
                for r in responses:
                    pc[prefix+'_known_model_tokens']+=sum(r[k] or 0 for k in ('input_tokens','output_tokens'))
                    pc[prefix+'_missing_usage_references']+=sum(r[k] is None for k in ('input_tokens','output_tokens'))
            actual={}
            for arm,c in row['conditions'].items():
                if not arm.startswith(regime+'_') or sorted(c['action']['edges'])!=INSTAR:continue
                same=c['responses'][1]['call_key']==key
                same_incoming=[d['text'] for d in c['deliveries'] if d['recipient']==1]==[d['text'] for d in deliveries]
                assert same==same_incoming
                subset[(arm,'instar')]+=1;subset[(arm,'identical_request')]+=same
                actual[arm]={'quota':c['action']['quota'],'same_hub_request':same,'hub_answer':c['responses'][1]['answer']}
            hub_rows.append({'id':qid,'regime':regime,'broadcast_hub_key':key,'broadcast_hub_answer':dense['responses'][1]['answer'],
                             'projection_exact_request':True,'actual_instar_arms':actual,'gold_answer':reference})
    for (exp,arm,rule), qs in by_arm.items():
        assert set(qs)==set(gold)
        summaries.setdefault(exp,{}).setdefault(arm,{})[rule]={'n':300,'correct':sum(r['em']==100 for r in qs.values()),
            'em':sum(r['em'] for r in qs.values())/300,'f1':sum(r['f1'] for r in qs.values())/300}
    contrasts={}
    for exp,arm,base in [('E5','P_Broadcast','P_NoComm'),('E5','P_Learned_seed31','P_Broadcast'),('E5','P_Learned_seed31','P_NoComm'),('E5','P_Learned_seed19','P_Broadcast'),('E1','P_CW','P_NoComm'),('E1','P_CCW','P_NoComm')]:
        for rule in RULES:
            contrasts[exp+':'+arm+'-'+base+':'+rule]={m:prior.paired({q:r[m] for q,r in by_arm[exp,arm,rule].items()},
                {q:r[m] for q,r in by_arm[exp,base,rule].items()},binary=m=='em') for m in ('em','f1')}
    for path, expected in protected.items(): assert prior.sha(path)==expected,path
    result={'status':'COMPLETE_OFFLINE_ONLY','new_provider_calls':0,'unique_questions':300,'cohorts':'E1 and E5 overlap exactly; not 600 independent questions',
            'summaries':summaries,'contrasts':contrasts,'projection_costs':projection_costs,
            'actual_instar_identity':{arm:{field:subset[arm,field] for field in ('instar','identical_request')} for arm in sorted({k[0] for k in subset})},
            'verification':dict(counts),'exact_projection_hub_requests':len(hub_rows),
            'interpretation':'InStar256 projection reuses saved Broadcast Hub-1 output under an exactly reconstructed identical request. This is structural one-round readout sufficiency, not a separately generated treatment, learned-policy advantage, or statistical equivalence test.',
            'decision':'Keep one-sender 2Wiki stopped. Complete the existing-data readout report first. No N=100 launch: complementary-router implementation and a separate frozen study remain necessary.'}
    prior.save(out/'summary.json',result)
    prior.jsonl(out/'question_readouts.jsonl',question_rows)
    prior.jsonl(out/'hub_request_audit.jsonl',hub_rows)
    prior.save(out/'provenance.json',{'inputs_sha256':protected,'analysis_sha256':prior.sha(__file__),'raw_inputs_unchanged':True,'new_provider_calls':0})
    for name in ('question_readouts.jsonl','provenance.json'):
        with (out/(name+'.gz')).open('wb') as stream:
            with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:
                compressed.write((out/name).read_bytes())
    print(json.dumps({k:result[k] for k in ('status','new_provider_calls','verification','actual_instar_identity','projection_costs')},indent=2))


if __name__=='__main__': main()
