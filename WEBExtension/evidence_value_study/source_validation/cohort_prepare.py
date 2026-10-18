"""Offline, outcome-blind candidate preparation. No provider/client imports or calls.

Historical input membership is fixed by historical_baseline.json. All output paths
are confined to this new study. Raw dataset availability is not study exposure.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import sys

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
SEED = 'E9-candidate-priority-v1-20261016'
RAW_HOTPOT = 'WEBExtension/data/hotpot_dev_fullwiki_v1.json'
RAW_MUSIQUE = 'WEBExtension/causal_study/external_validation/sources/musique_full_v1.0_dev.jsonl'
RANKING_PATHS = {
 'WEBExtension/causal_study/external_validation/results/question_assignment.json',
 'WEBExtension/causal_study/external_validation/amendments/001_question_identity_deduplication/results/question_assignment.json',
}
TEXT_SUFFIXES = {'.json','.jsonl','.csv','.tsv','.ipynb','.md','.html','.txt','.log','.body','.py','.yaml','.yml','.tex'}
ID_PATTERN = re.compile(r'(?<![A-Za-z0-9])(?:[a-f0-9]{24}|[234]hop[123]?__(?:\d+_)*\d+)(?![A-Za-z0-9_])')
QUESTION_LITERAL = re.compile(r'"(?:question|question_text|query)"\s*:\s*("(?:[^"\\]|\\.)*")')


def normalize_question(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def audit_exposure(root, records, paths, raw_paths, ranking_paths=RANKING_PATHS):
    """Conservative observed-log audit, including IDs and normalized question aliases.

    Full published source catalogs are excluded. Ranking-only IDs remain explicitly
    flagged, while reserved pilot/annotation/first1000 confirmation IDs are excluded.
    """
    by_question=defaultdict(set)
    for r in records:by_question[normalize_question(r['question'])].add(r['id'])
    known={r['id'] for r in records};max_len=max(map(len,by_question),default=0)+80
    id_paths=defaultdict(set);question_paths=defaultdict(set);ranked=set();inputs=[];issues=[]
    parse_issues=[];raw_skipped=[]
    def walk(x,out):
        if isinstance(x,dict):it=x.values()
        elif isinstance(x,list):it=x
        else:return
        for value in it:
            if isinstance(value,(dict,list)):walk(value,out)
            elif isinstance(value,str):
                if len(value)<=max_len:
                    out.update(by_question.get(normalize_question(value),()))
                if value[:1] in ('{','[') and ('question' in value or 'query' in value):
                    try:walk(json.loads(value),out)
                    except (ValueError,RecursionError):pass
    for name in sorted(paths):
        if name in raw_paths:raw_skipped.append(name);continue
        p=Path(root)/name
        if not(p.suffix in TEXT_SUFFIXES or '.json.' in name or '.jsonl.' in name):continue
        try:
            data=p.read_bytes()
            if data.startswith((b'\xff\xfe',b'\xfe\xff')):encoding='utf-16';text=data.decode(encoding)
            else:
                try:encoding='utf-8';text=data.decode(encoding)
                except UnicodeDecodeError:
                    try:encoding='cp1252';text=data.decode(encoding)
                    except UnicodeDecodeError:encoding='latin-1';text=data.decode(encoding)
        except (OSError,UnicodeDecodeError) as exc:issues.append({'path':name,'error':type(exc).__name__});continue
        info={'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'text_encoding':encoding}
        found=set(ID_PATTERN.findall(text))&known
        qfound=set()
        if name in ranking_paths:
            obj=json.loads(text);ranked.update(found)
            chosen=set(obj.get('pilot_ids',[])+obj.get('annotation_ids',[])+obj.get('confirmatory_order',[])[:1000])
            found &= chosen
            info['ranking_only_ids_not_equivalent_to_observed_outcomes']=len(ranked-chosen)
        elif ('question' in text or 'query' in text):
            if p.suffix in {'.json','.jsonl','.ipynb','.body'} or '.json.' in name or '.jsonl.' in name:
                try:
                    if p.suffix=='.jsonl':
                        for line in text.splitlines():
                            if line.strip():walk(json.loads(line),qfound)
                    else:walk(json.loads(text),qfound)
                except (ValueError,RecursionError):parse_issues.append(name)
            # Also catches literal questions inside historical non-JSON exports.
            for match in QUESTION_LITERAL.finditer(text):
                try:qfound.update(by_question.get(normalize_question(json.loads(match.group(1))),()))
                except ValueError:pass
        for rid in found:id_paths[rid].add(name)
        for rid in qfound:question_paths[rid].add(name)
        if found or qfound:
            info.update(matched_ids=sorted(found),normalized_question_match_ids=sorted(qfound))
        inputs.append(info)
    excluded=set(id_paths)|set(question_paths)
    # Close identity equivalence even when historical record carried only one ID.
    for ids in by_question.values():
        if ids&excluded:excluded.update(ids)
    return {
      'scope':'historical baseline regular text files; raw catalog files excluded; binaries and inaccessible history not interpreted',
      'normalization':'NFKC + Unicode casefold + Unicode word-token join',
      'excluded_ids':sorted(excluded),'ranking_only_ids':sorted(ranked-excluded),
      'id_evidence':{k:sorted(v) for k,v in sorted(id_paths.items())},
      'normalized_question_evidence':{k:sorted(v) for k,v in sorted(question_paths.items())},
      'inputs':inputs,'unreadable_inputs':issues,'structured_parse_fallbacks':parse_issues,
      'raw_catalogs_excluded':raw_skipped,
      'freshness_limit':'No observed prior selection/question-text match in audited logs is not proof of never-used or pretraining-unseen. Ranking-only membership is disclosed separately.',
    }


def make_candidate(row,row_index,dataset,docs,support_indices,provenance):
    if len(support_indices)!=2 or len(set(support_indices))!=2:return None
    if len(docs)<6 or any(not d['text'].strip() for d in docs):return None
    if len({d['title'] for d in docs})!=len(docs):return None
    nonsupport=[i for i in range(len(docs)) if i not in support_indices]
    nonsupport.sort(key=lambda i:digest([SEED,'distractors',dataset,row['id'],i]))
    chosen=support_indices+nonsupport[:4]
    if len(chosen)!=6:return None
    chosen.sort(key=lambda i:digest([SEED,'union-order',dataset,row['id'],i]))
    documents=[];support_ids=[]
    for pos,idx in enumerate(chosen):
        d=docs[idx];doc_id=f'd{pos+1}'
        documents.append({'id':doc_id,'title':d['title'],'text':d['text'],
          'provenance':dict(provenance,source_row_0based=row_index,source_document_idx=d.get('original_idx',idx),text_sha256=hashlib.sha256(d['text'].encode()).hexdigest())})
        if idx in support_indices:support_ids.append(doc_id)
    return {'id':row['id'],'dataset':dataset,'question':row['question'],'documents':documents,
      'support_ids':support_ids,'gold_answer':row['answer'],'gold_aliases':row.get('answer_aliases',[]),
      'source_record_ref':{'source_row_0based':row_index,'source_sha256':provenance['source_sha256']},
      'priority_hash':digest([SEED,dataset,row['id']])}


def hotpot_candidate(row,row_index,provenance):
    context=row['context'];titles=context['title'];sentences=context['sentences']
    supports=list(dict.fromkeys(row['supporting_facts']['title']))
    if len(supports)!=2 or any(t not in titles for t in supports):return None
    docs=[{'title':t,'text':''.join(ss),'original_idx':i} for i,(t,ss) in enumerate(zip(titles,sentences))]
    return make_candidate(row,row_index,'HotpotQA_fullwiki_dev',docs,[titles.index(t) for t in supports],provenance)


def musique_candidate(row,row_index,provenance):
    if not row.get('answerable') or not row['id'].startswith('2hop__'):return None
    pars=row['paragraphs'];supports=[i for i,p in enumerate(pars) if p.get('is_supporting')]
    if len(row.get('question_decomposition',[]))!=2:return None
    docs=[{'title':p['title'],'text':p['paragraph_text'],'original_idx':p['idx']} for p in pars]
    return make_candidate(row,row_index,'MuSiQue_full_v1.0_dev_answerable_2hop',docs,supports,provenance)


def lexical_checks(c):
    gold=normalize_question(c['gold_answer']);supports=set(c['support_ids'])
    hits=lambda docs:any(gold and gold in normalize_question(d['title']+' '+d['text']) for d in docs)
    return {'id':c['id'],'answer_string_in_question':bool(gold and gold in normalize_question(c['question'])),
      'answer_string_in_title':any(gold and gold in normalize_question(d['title']) for d in c['documents']),
      'answer_string_in_distractors':hits([d for d in c['documents'] if d['id'] not in supports]),
      'support_doc_count':len(supports),'global_union_document_count':len(c['documents']),
      'semantic_necessity':'UNASSESSED; string overlap is neither sufficiency nor necessity'}


def annotation_packages(candidates):
    # design.py imports provider helpers but performs no network operation at import.
    if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
    from WEBExtension.evidence_value_study.experiments.design import assign, VERSION
    packages={s:[] for s in ['R1','R2']};labels={s:[] for s in packages};keys=[]
    for ordinal,c in enumerate(candidates):
        index=c.get('candidate_priority',ordinal+1)-1
        plans={'primary':assign(c,index),'flipped':assign(c,index^1)}
        union_sha=digest({'id':c['id'],'question':c['question'],'documents':[{k:d[k] for k in ('id','title','text')} for d in c['documents']]})
        support=c['support_ids'];other=[d['id'] for d in c['documents'] if d['id'] not in support]
        subsets={'A_only':[support[0]],'B_only':[support[1]],'distractors_only':other,'A_plus_B':support,'full_union':[d['id'] for d in c['documents']],'full_union_minus_A':[d['id'] for d in c['documents'] if d['id']!=support[0]],'full_union_minus_B':[d['id'] for d in c['documents'] if d['id']!=support[1]]}
        for orientation,plan in plans.items():
            subsets['exact_receiver_missing_'+orientation]=plan['placements']['missing'][plan['receiver']]
        for slot in packages:
            for subset,ids in subsets.items():
                cid=digest([SEED,'annotation-case',slot,c['id'],subset])[:24]
                docs=[d for d in c['documents'] if d['id'] in ids]
                docs.sort(key=lambda d:digest([SEED,slot,cid,d['id']]))
                neutral=[];mapping={}
                for i,d in enumerate(docs):
                    nid='source_'+digest([SEED,slot,cid,i])[:8];mapping[nid]=d['id']
                    neutral.append({'id':nid,'title':d['title'],'text':d['text']})
                packages[slot].append({'case_id':cid,'question':c['question'],'documents':neutral})
                labels[slot].append({'case_id':cid,'reviewer_slot':slot,'sufficient':None,'answer_from_sources':None,
                  'cited_source_ids':[],'rationale':None,'title_metadata_leak':None,'uncertainty':None,'reviewed_at_utc':None})
                orientation='flipped' if subset=='exact_receiver_missing_flipped' else 'primary'
                keys.append({'reviewer_slot':slot,'case_id':cid,'question_id':c['id'],'subset':subset,
                  'document_id_key':mapping,'candidate_union_sha256':union_sha,
                  'planned_assignment_sha256':digest(plans[orientation]),
                  'assignment_index':index^1 if orientation=='flipped' else index,
                  'design_version':VERSION,'expected_sufficient':subset in ('A_plus_B','full_union')})
    for slot in packages:
        packages[slot].sort(key=lambda r:digest([SEED,'case-order',slot,r['case_id']]))
        byid={r['case_id']:r for r in labels[slot]};labels[slot]=[byid[r['case_id']] for r in packages[slot]]
    return packages,labels,keys


def human_gate(registry, label_sets, adjudications, private_key):
    """Conservative identity gate; final study launch validator is separate.

    This preparer does not certify or import completed external labels. The only
    supported prepared state is no completed independent reviews and STOP.
    """
    return {'qualified_count':0,'completed_human_reviewers':0,'paid_gate':'STOP',
      'reason':'Independent qualified HUMAN source-necessity reviews and adjudication have not been authenticated by the study launch validator.'}


def write_json(rel,obj):
    p=STUDY/rel;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        # Preparation reruns may verify identical data; they must never erase human work.
        old=p.read_text();new=json.dumps(obj,ensure_ascii=False,indent=2)+'\n'
        if json.loads(old)!=obj:raise FileExistsError('Refusing to replace existing artifact: '+str(p))
        return
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def write_jsonl(rel,rows):
    p=STUDY/rel;p.parent.mkdir(parents=True,exist_ok=True)
    text=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows)
    if p.exists() and p.read_text()!=text:raise FileExistsError('Refusing to replace existing annotation artifact: '+str(p))
    if not p.exists():p.write_text(text)


def main():
    baseline=json.loads((STUDY/'source_validation/historical_baseline.json').read_text())
    hot=json.loads((REPO/RAW_HOTPOT).read_text())
    mus=[json.loads(line) for line in (REPO/RAW_MUSIQUE).open() if line.strip()]
    hotprov=json.loads((REPO/'WEBExtension/data/hotpot_dev_fullwiki_v1.provenance.json').read_text())
    musprov=json.loads((REPO/'WEBExtension/causal_study/external_validation/sources/musique_mirror_acquisition.json').read_text())
    assert sha(REPO/RAW_HOTPOT)==hotprov['sha256'],'Hotpot source hash mismatch'
    assert sha(REPO/RAW_MUSIQUE)==musprov['expected_sha256']==musprov['sha256'],'MuSiQue source hash mismatch'
    exposure=audit_exposure(REPO,hot+mus,baseline['files'],{RAW_HOTPOT,RAW_MUSIQUE})
    for r in exposure['inputs']:
        if r['sha256']!=baseline['files'][r['path']]['sha256']:raise RuntimeError('Historical input changed: '+r['path'])
    excluded=set(exposure['excluded_ids']);rank_only=set(exposure['ranking_only_ids'])
    sources={
      'hotpot':dict(hotprov,source_path=RAW_HOTPOT,source_sha256=hotprov['sha256']),
      'musique':dict(musprov,source_path=RAW_MUSIQUE,source_sha256=musprov['sha256'],author_archive_equivalence='NOT_VERIFIED',license='CC BY 4.0 per preserved original-author repository LICENSE'),
    }
    hotcap=[c for i,r in enumerate(hot) if (c:=hotpot_candidate(r,i,sources['hotpot']))]
    muscap=[c for i,r in enumerate(mus) if (c:=musique_candidate(r,i,sources['musique']))]
    def eligible(candidates):
        seen=set();out=[]
        for c in sorted(candidates,key=lambda c:c['priority_hash']):
            q=normalize_question(c['question'])
            if c['id'] in excluded or q in seen:continue
            seen.add(q);out.append(c)
        return out
    hotfresh=eligible(hotcap);musfresh=eligible(muscap)
    # Source-capable is structural only. No model answers or outcomes are read here.
    pool=hotfresh if len(hotfresh)>=40 else musfresh
    chosen=pool[:40]
    if len(chosen)!=40:raise RuntimeError('Cannot prepare target30 + reserve10 structurally usable candidates')
    for i,c in enumerate(chosen):c['candidate_priority']=i+1;c['candidate_role']='target_priority' if i<30 else 'reserve_priority'
    summary={
      'status':'CANDIDATES_ONLY_NOT_SOURCE_NECESSITY_QUALIFIED','selection_seed':SEED,
      'ranking_rule':'SHA256(seed,dataset,id); deduplicate normalized questions; no model outcomes',
      'distractor_rule':'first4 original non-support paragraphs under fixed SHA256(seed,distractors,dataset,id,original_index); no gold lexical filtering',
      'union_rule':'two benchmark-designated original support documents + four fixed original non-support documents; whole benchmark text, no redaction',
      'selected_dataset':chosen[0]['dataset'],'candidate_count':len(chosen),'target_priority_count':30,'reserve_priority_count':10,
      'hotpot':{'source_rows':len(hot),'structurally_source_capable':len(hotcap),'not_observed_previously_selected_source_capable':len(hotfresh),'observed_excluded_ids':len({r['id'] for r in hot}&excluded)},
      'musique':{'source_rows':len(mus),'source_unique_ids':len({r['id'] for r in mus}),'answerable_2hop_rows':sum(r['answerable'] and r['id'].startswith('2hop__') for r in mus),'structurally_source_capable':len(muscap),'not_observed_previously_selected_source_capable':len(musfresh),'strict_exclude_all_ranking_membership_source_capable':sum(c['id'] not in rank_only for c in musfresh),'ranking_only_membership_candidate_count':sum(c['id'] in rank_only for c in chosen)},
      'exposure_scan':{'baseline_files':len(baseline['files']),'text_inputs_scanned':len(exposure['inputs']),'bytes_scanned':sum(r['bytes'] for r in exposure['inputs']),'unreadable_count':len(exposure['unreadable_inputs']),'structured_parse_fallbacks':len(exposure['structured_parse_fallbacks'])},
      'qualified_count':0,'completed_human_reviewers':0,'paid_gate':'STOP','qualification_reason':'Two independent qualified HUMAN reviewers plus adjudication missing',
      'freshness_limit':exposure['freshness_limit'],'semantic_checks':'All candidate necessity/sufficiency remains unassessed; benchmark labels and lexical flags are not human qualification',
    }
    packages,labels,key=annotation_packages(chosen)
    registry={'status':'UNASSIGNED','reviewers':[{'reviewer_slot':s,'human_identity':None,'is_human':None,'source_necessity_qualification':None,'independent_of_other_reviewer':None,'blinded_to_gold_support_hypothesis':None,'completed_at_utc':None} for s in ['R1','R2']], 'adjudicator':{'human_identity':None,'is_human':None,'source_necessity_qualification':None,'completed_at_utc':None},'external_preexisting_labels_imported':False}
    public=[{k:v for k,v in c.items() if k not in {'gold_answer','gold_aliases','support_ids'}} for c in chosen]
    write_json('source_validation/exposure_inputs.json',exposure.pop('inputs'))
    write_json('source_validation/exposure_audit.json',exposure)
    write_json('source_validation/cohort_source_provenance.json',sources)
    write_json('source_validation/cohort_audit.json',summary)
    write_json('source_validation/cohort_lexical_checks.json',[lexical_checks(c) for c in chosen])
    write_json('frozen/private_candidates.json',chosen)
    write_json('frozen/candidate_manifest.json',public)
    write_json('frozen/candidate_status.json',summary)
    write_json('frozen/private_annotation_key.json',{'status':'SEALED_CONSTRUCTION_KEY_DO_NOT_GIVE_REVIEWERS','cases':key,'gold_and_support_labels_path':'private_candidates.json'})
    write_json('annotations/reviewer_registry.json',registry)
    for slot in packages:
        write_jsonl(f'annotations/blind_{slot}.jsonl',packages[slot])
        write_jsonl(f'annotations/labels_{slot}.jsonl',labels[slot])
    write_jsonl('annotations/adjudication_template.jsonl',[{'question_id':c['id'],'adjudicator_identity':None,'A_only_insufficient':None,'B_only_insufficient':None,'distractors_only_insufficient':None,'A_plus_B_sufficient':None,'full_union_sufficient':None,'full_union_minus_A_insufficient':None,'full_union_minus_B_insufficient':None,'exact_receiver_missing_primary_insufficient':None,'exact_receiver_missing_flipped_insufficient':None,'candidate_union_sha256':digest({'id':c['id'],'question':c['question'],'documents':[{k:d[k] for k in ('id','title','text')} for d in c['documents']]}),'no_title_metadata_leak':None,'support_both_necessary':None,'disagreement_resolution':None,'qualified':None,'adjudicated_at_utc':None} for c in chosen])
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
