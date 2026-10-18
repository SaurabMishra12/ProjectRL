"""Offline independent adapters, bounded trace audits, and benchmark inspection."""
from pathlib import Path
from collections import Counter, defaultdict
import ast, hashlib, importlib.util, json, re
import numpy as np
from .contracts import RULES,hotpot_proposal,hotpot_normalize,readout,gsm_proposal,metrics
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
WEB=REPO/'WEBExtension'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,data):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def jl(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def write_rows(name,rows):
    (ROOT/name).write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in rows))
def interval(values):
    a=np.array(values,dtype=float);rng=np.random.default_rng(20261016)
    means=np.concatenate([a[rng.integers(0,len(a),(250,len(a)))].mean(1) for _ in range(40)])
    return {'n_unique_questions':len(a),'mean_pp':100*float(a.mean()),'ci95_pp':(100*np.quantile(means,[.025,.975])).tolist()}


def ranked_question_units(rows):
    """Select question-text units before any outputs, retaining a full paired ID."""
    texts={}
    groups=defaultdict(set)
    for row in rows:
        key=' '.join(re.findall(r'\w+',row['question'].casefold()))
        if row['id'] in texts and texts[row['id']]!=key:raise ValueError('Inconsistent question text within ID')
        texts[row['id']]=key;groups[key].add(row['id'])
    rank=lambda q:hashlib.sha256(('external-validation-v1/'+q).encode()).hexdigest()
    ordered=[];seen=set()
    for q in sorted(texts,key=rank):
        if texts[q] not in seen:ordered.append(q);seen.add(texts[q])
    return ordered,[sorted(g) for _,g in sorted(groups.items()) if len(g)>1]


def historical():
    raw=WEB/'artifacts/controller_study_a0b3768357484b45/E5.jsonl'
    goldpath=WEB/'artifacts/run_f6aa61bac2c3dfae/sample_gold.json'
    gold={r['id']:r['answer'] for r in json.loads(goldpath.read_text())};rows=jl(raw)
    assert len(rows)==300 and set(gold)=={r['id'] for r in rows}
    out=[];proposals=defaultdict(Counter);perq={}
    for r in rows:
        perq[r['id']]={}
        for arm,state in sorted(r['conditions'].items()):
            ps=[hotpot_proposal(x) for x in state['responses']];proposals[arm]+=Counter(p.kind for p in ps)
            assert len(ps)==3 and all(p.kind in {'answer','abstain','refusal'} for p in ps)
            for rule in RULES:
                d=readout(ps,rule);correct=d.kind=='answer' and hotpot_normalize(d.text)==hotpot_normalize(gold[r['id']])
                item={'id':r['id'],'arm':arm,'rule':rule,'text':d.text,'kind':d.kind,'correct':bool(correct),'proposal_kinds':[p.kind for p in ps]}
                out.append(item);perq[r['id']][arm,rule]=bool(correct)
    metrics_rows=[]
    for arm in sorted(rows[0]['conditions']):
        for rule in RULES:
            subset=[x for x in out if x['arm']==arm and x['rule']==rule]
            metrics_rows.append({'arm':arm,'rule':rule,'n':300,'correct':sum(x['correct'] for x in subset),
                'EM_percent':sum(x['correct'] for x in subset)/3,'system_kind_counts':dict(Counter(x['kind'] for x in subset)),
                'expressed_proposal_kind_counts':dict(proposals[arm]),'unsupported_false_answer_rate':'NOT_ESTIMABLE_NO_SUFFICIENCY_LABELS'})
    contrasts=[]
    for regime in ['P','R']:
        for base in ['NoComm','SelfUpdate']:
            for rule in RULES:
                contrasts.append({'regime':regime,'baseline':base,'rule':rule,**interval([int(v[regime+'_Broadcast',rule])-int(v[regime+'_'+base,rule]) for v in perq.values()])})
            contrasts.append({'regime':regime,'baseline':base,'rule':'aware_minus_original_interaction',**interval([(int(v[regime+'_Broadcast','aware'])-int(v[regime+'_'+base,'aware']))-(int(v[regime+'_Broadcast','original'])-int(v[regime+'_'+base,'original'])) for v in perq.values()])})
    save('results/historical_four_readouts.json',{'same_300_not_replication':True,'raw_sha256':sha(raw),'gold_sha256':sha(goldpath),'metrics':metrics_rows,'contrasts':contrasts})
    write_rows('results/historical_question_readouts.jsonl',out)
    return metrics_rows


def public_sources():
    # Check pinned git blob hashes; public files remain local audit evidence.
    checks=[]
    for treepath in (ROOT/'sources').glob('*_tree.json'):
        d=json.loads(treepath.read_text());prefix=treepath.name.removesuffix('_tree.json')
        assert not d.get('truncated'),treepath
        for entry in d['tree']:
            p=ROOT/'sources'/prefix/entry['path']
            if entry['type']!='blob' or not p.exists():continue
            b=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
            assert blob==entry['sha'],p
            checks.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'git_blob_sha1':blob,'commit':d['sha']})
    save('results/public_blob_verification.json',{'verified_files':len(checks),'files':checks})


def external_gsm():
    source=ROOT/'sources/moozwen__sakana-debate'
    # grading.py is inspected pure regex/float code; never import run_debate.py.
    spec=importlib.util.spec_from_file_location('frozen_external_grading',source/'code/grading.py');grade=importlib.util.module_from_spec(spec);spec.loader.exec_module(grade)
    dataset={r['idx']:r for r in jl(source/'experiments/phase1b_1c_regime_budget/data/gsm8k_subset_regime.jsonl')}
    results=[];records=[];sets=[]
    for path in sorted(source.glob('experiments/phase1b_1c_regime_budget/results/*/debate_N3_R2_seed1.jsonl')):
        rows=jl(path);ids=[r['idx'] for r in rows];assert len(ids)==len(set(ids))==500
        assert set(ids)==set(dataset);sets.append(set(ids));model=path.parent.name
        parser_mismatches=0;counts=Counter();perq={}
        for row in rows:
            assert row['n']==3 and row['r']==2 and row['seed']==1 and len(row['rounds'])==3
            assert grade._norm(str(row['gold']))==grade._norm(str(dataset[row['idx']]['gold']))
            assert row['final_preds']==row['rounds'][-1]['preds']
            perq[row['idx']]={}
            for rnd,stage in enumerate(row['rounds']):
                assert len(stage['raw'])==len(stage['preds'])==3
                parser_mismatches+=sum(grade.extract(t)!=a for t,a in zip(stage['raw'],stage['preds']))
                counts['agent_states']+=3;counts['saved_numeric_parse_failures']+=sum(a is None for a in stage['preds'])
                counts['raw_empty']+=sum(not t.strip() for t in stage['raw'])
                # Diagnostic search only; not a semantic annotation or parser rule.
                counts['possible_abstention_phrase_mentions']+=sum(bool(re.search(r"\b(?:I (?:do not|don't) know|cannot (?:answer|determine)|unable to (?:answer|determine))\b",t,re.I)) for t in stage['raw'])
                ps=[gsm_proposal(a) for a in stage['preds']]
                for rule in RULES:
                    result=readout(ps,rule,normalize=str)
                    pred=float(result.text) if result.kind=='answer' else None
                    correct=bool(grade.grade(pred,row['gold']))
                    perq[row['idx']][rnd,rule]=correct
                    records.append({'model_directory':model,'question_index':row['idx'],'round':rnd,'rule':rule,'prediction':pred,'kind':result.kind,'correct':correct})
                    if rnd==2 and rule=='original':
                        assert pred==row['majority'] and correct==row['ok'],'Original source vote/grade mismatch'
                        available=[a for a in stage['preds'] if a is not None];c=Counter(available)
                        assert row['tie']==(bool(c) and sum(v==max(c.values()) for v in c.values())>1)
            counts['n_questions']+=1
        summaries=[]
        for rule in RULES:
            final_decisions=[r for r in records if r['model_directory']==model and r['round']==2 and r['rule']==rule]
            answered=[r for r in final_decisions if r['kind']=='answer']
            summaries.append({'rule':rule,'initial_correct':sum(v[0,rule] for v in perq.values()),'final_correct':sum(v[2,rule] for v in perq.values()),
                'final_minus_initial':interval([int(v[2,rule])-int(v[0,rule]) for v in perq.values()]),
                'readout_by_round_interaction_vs_original':interval([(int(v[2,rule])-int(v[0,rule]))-(int(v[2,'original'])-int(v[0,'original'])) for v in perq.values()]),
                'final_coverage':len(answered)/500,'final_selective_risk':sum(not r['correct'] for r in answered)/len(answered) if answered else None,
                'unsupported_false_answer_rate':'NOT_ESTIMABLE_NO_UNSUPPORTED_LABELS'})
        assert all(v[r,'aware']==v[r,'original'] for v in perq.values() for r in range(3))
        results.append({'model_directory':model,'n_unique_questions':500,'sha256':sha(path),'counts':dict(counts),'current_source_extractor_mismatches':parser_mismatches,
            'summary':summaries,'aware_original_communication_interaction':interval([(int(v[2,'aware'])-int(v[0,'aware']))-(int(v[2,'original'])-int(v[0,'original'])) for v in perq.values()]),
            'semantic_abstention_labels':'NOT_PRESERVED_AS_DISTINCT_FIELD','provider_refusal_labels':'UNAVAILABLE','private_revision_call_matched':'UNAVAILABLE'})
    assert len(results)==8 and all(s==sets[0] for s in sets)
    save('results/external_gsm_results.json',{'independent_public_pipeline':True,'repository':'moozwen/sakana-debate','independent_question_ids':500,'models':results,
        'interpretation':'Original numeric-source aggregator already excludes parse failures. Aware equals original by contract; not a powered semantic-abstention null replication. Models share questions; never pool 4000 as independent units.'})
    write_rows('results/external_gsm_question_readouts.jsonl',records)
    return results


def benchmark():
    path=ROOT/'sources/musique_full_v1.0_dev.jsonl';rows=jl(path);groups=defaultdict(list)
    for i,r in enumerate(rows):
        assert type(r['answerable']) is bool and isinstance(r['answer'],str)
        assert r['paragraphs'] and len({p['idx'] for p in r['paragraphs']})==len(r['paragraphs'])
        assert all(isinstance(p['paragraph_text'],str) and p['paragraph_text'].strip() and type(p['is_supporting']) is bool for p in r['paragraphs'])
        groups[r['id']].append((i,r))
    assert len(rows)==4834 and len(groups)==2417
    for pairs in groups.values():
        assert len(pairs)==2 and {r['answerable'] for _,r in pairs}=={True,False}
        assert len({r['question'] for _,r in pairs})==1
    ordered,duplicate_groups=ranked_question_units(rows);pilot=ordered[:20];annotation=ordered[20:44];confirm=ordered[44:]
    # Hash-ranked source-level selection; no proposal/correctness/disagreement filtering.
    assignments={'selection_seed':'external-validation-v1','data_sha256':sha(path),'unit':'normalized question text; first hash-ranked ID represents each unit; preserve both contexts',
        'amendment':'001_question_identity_deduplication','cross_ID_duplicate_groups':duplicate_groups,'n_unique_question_text_units':len(ordered),
        'pilot_ids':pilot,'annotation_ids':annotation,'confirmatory_order':confirm}
    save('results/question_assignment.json',assignments)
    sample=[];key=[]
    for q in annotation:
        for i,r in groups[q]:
            opaque=hashlib.sha256(('blind-source-v1/'+q+'/'+str(r['answerable'])).encode()).hexdigest()[:20]
            sample.append({'case_id':opaque,'question':r['question'],'sources':[{'source_id':p['idx'],'title':p['title'],'text':p['paragraph_text']} for p in r['paragraphs']]})
            key.append({'case_id':opaque,'question_id':q,'answerable':r['answerable'],'reference':r['answer'],'reference_aliases':r['answer_aliases'],'original_row_0based':i,'source_sha256':sha(path)})
    for reviewer in ['A','B']:
        ordered_cases=sorted(sample,key=lambda c:hashlib.sha256((reviewer+c['case_id']).encode()).hexdigest())
        write_rows('annotation/blind_reviewer_'+reviewer+'.jsonl',ordered_cases)
        write_rows('annotation/labels_'+reviewer+'.jsonl',[{'case_id':r['case_id'],'source_sufficiency':'PENDING','supported_answer':None,'evidence_addresses':[],'missing_relation':None,'confidence':None} for r in ordered_cases])
    save('annotation/SEALED_KEY_DO_NOT_GIVE_ANNOTATORS.json',key)
    save('results/musique_inspection.json',{'n_rows':len(rows),'n_paired_question_ids':len(groups),'n_unique_question_text_units':len(ordered),
        'cross_ID_duplicate_groups':duplicate_groups,'paired_same_id_and_question':True,'answerable_rows':sum(r['answerable'] for r in rows),
        'unanswerable_rows':sum(not r['answerable'] for r in rows),'paragraph_count_distribution':dict(Counter(len(r['paragraphs']) for r in rows)),'nonempty_sources_verified':True,'data_sha256':sha(path),
        'official_evaluator_scoring':'answer EM/F1 on answerable rows; grouped sufficiency gates answer score on BOTH predicted answerability labels; retains world-reference answer on insufficient rows',
        'official_author_archive_byte_equality':'NOT_VERIFIED; pinned mirror matches published LFS SHA256','pilot_unique_questions':20,'annotation_unique_questions':24,'confirmation_pool':len(confirm),
        'semantic_annotation_status':'48 cases x two independent label files PENDING; no semantic labels fabricated'})
    return assignments


def local_inventory():
    files=list((REPO/'results').glob('**/predictions.jsonl'))+list((WEB/'artifacts').glob('*/E[1-5].jsonl'))+list((REPO/'results').glob('full_results_*.json'))
    inventory=[]
    for p in sorted(files):
        try:
            d=jl(p) if p.suffix=='.jsonl' else json.loads(p.read_text())
            inventory.append({'path':str(p.relative_to(REPO)),'sha256':sha(p),'bytes':p.stat().st_size,'root_type':type(d).__name__,'records':len(d),
                'schema_first_keys':list((d[0] if isinstance(d,list) and d else d))[:20],
                'independent_pipeline':'NOT_ESTABLISHED; local ProjectRL or old Web suite, shared provenance or missing distinct abstention/provider fields'})
        except Exception as e:inventory.append({'path':str(p.relative_to(REPO)),'status':'INSPECTION_ERROR','error':str(e)})
    save('results/local_inventory.json',{'scope':'results/**/predictions.jsonl; results/full_results_*.json; WEBExtension/artifacts/*/E[1-5].jsonl. Prior source-transition audit covers E3-v2. Not an exhaustive host-wide or internet absence claim.','files':inventory})


def main():
    historical();external_gsm();benchmark();local_inventory();public_sources()
    print('Completed independent historical 4-readout reanalysis, 8 external trace audits, MuSiQue paired schema and blinded packages; zero inference.')
if __name__=='__main__':main()
