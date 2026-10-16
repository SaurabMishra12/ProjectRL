"""Stratify saved E1/E5 readouts by E1 partitioned annotated-gold coverage.

Offline only: no provider client, credentials, tokenizer, or model invocation.
Run: python -m WEBExtension.pilots.hotpot_readout_20261016.stratify
"""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

from .analyse import ROOT, E1, E5, RULES, prior

STRATA=('Disjoint-Gold','Co-located-Gold','Retrieval-Missing-Gold','Gold-Complete')
EXPECTED={'Disjoint-Gold':62,'Co-located-Gold':14,'Retrieval-Missing-Gold':224,'Gold-Complete':76}


def main():
    out=ROOT/'stratification';out.mkdir(exist_ok=True)
    hashes={}
    def read(path):
        path=Path(path);hashes[str(path)]=prior.sha(path);return path.read_bytes()
    gold={r['id']:r for r in json.loads(read(E1/'sample_gold.json'))}
    raw={}
    for exp,folder in [('E1',E1),('E5',E5)]:
        path=folder/(exp+'.jsonl');rows=[json.loads(l) for l in read(path).splitlines()]
        status=json.loads(read(folder/(exp+'_status.json')))
        assert status['status']=='COMPLETE' and status['sha256']==hashes[str(path)]
        assert len(rows)==len({r['id'] for r in rows})==300 and {r['id'] for r in rows}==set(gold)
        raw[exp]={r['id']:r for r in rows}
    # Confirm original paid ledgers remain untouched, without loading a backend.
    for path in (E1/'calls.jsonl',E5/'inference/calls.jsonl'):read(path)
    labels={};members={name:set() for name in STRATA};assignments=[]
    for qid,row in raw['E1'].items():
        supporting=set(map(tuple,gold[qid]['supporting_facts']))
        state=row['states']['P'];pool={(d['title'],i) for d in state['pool'] for i in range(len(d['sentences']))}
        own=[{(d['title'],i) for d in docs for i in range(len(d['sentences']))} for docs in state['assignment']]
        complete=supporting<=pool;colocated=any(supporting<=o for o in own)
        label='Retrieval-Missing-Gold' if not complete else 'Co-located-Gold' if colocated else 'Disjoint-Gold'
        labels[qid]=label;members[label].add(qid)
        if complete:members['Gold-Complete'].add(qid)
        for exp in ('E1','E5'):
            c=raw[exp][qid]['conditions']['P_NoComm']
            assert c['all_support_available']==complete
            assert c['split_support_across_nodes']==(complete and not colocated)
        assignments.append({'id':qid,'stratum':label,'gold_complete':complete,
                            'annotated_supporting_facts':sorted(supporting),'retrieved_titles':[d['title'] for d in state['pool']],
                            'assigned_titles':[[d['title'] for d in docs] for docs in state['assignment']]})
    assert {name:len(ids) for name,ids in members.items()}==EXPECTED
    packed=ROOT/'results/question_readouts.jsonl.gz'
    readouts=[json.loads(l) for l in gzip.decompress(read(packed)).splitlines()]
    previous=json.loads(gzip.decompress(read(ROOT/'results/provenance.json.gz')))
    for path,h in hashes.items():
        if path in previous['inputs_sha256']:assert previous['inputs_sha256'][path]==h,path
    official,evaluator_path=prior.official_evaluator();read(evaluator_path);read(Path(prior.__file__))
    grouped={};verified=Counter()
    for row in readouts:
        qid=row['id'];exp=row['experiment'];arm=row['arm'];assert qid in gold
        if exp in raw:
            c=raw[exp][qid]['conditions'][arm];assert row['call_keys']==[r['call_key'] for r in c['responses']]
            for rule in RULES:
                expected=c['responses'][1]['answer'] if rule=='sink1' else prior.aggregate(c['responses'],rule)
                assert row['readouts'][rule]['answer']==expected
        else:
            assert exp=='E5_projection'
            regime=arm[0];c=raw['E5'][qid]['conditions'][regime+'_Broadcast']
            initial=raw['E5'][qid]['conditions'][regime+'_NoComm']['responses']
            projected=[initial[0],c['responses'][1],initial[2]]
            assert row['call_keys']==[r['call_key'] for r in projected]
            for rule in RULES:
                expected=projected[1]['answer'] if rule=='sink1' else prior.aggregate(projected,rule)
                assert row['readouts'][rule]['answer']==expected
        key=(exp,arm)
        assert qid not in grouped.setdefault(key,{})
        grouped[key][qid]=row['readouts']
        for rule,v in row['readouts'].items():
            answer=v['answer'];reference=gold[qid]['answer']
            assert v['em']==100*official['exact_match_score'](answer,reference)
            assert abs(v['f1']-100*official['f1_score'](answer,reference)[0])<1e-8
            verified['answer_scores']+=1
    assert len(grouped)==30
    assert all(set(values)==set(gold) for values in grouped.values())
    summaries=[];contrasts=[]
    for (exp,arm),qs in grouped.items():
        for name,ids in members.items():
            for rule in RULES:
                values=[qs[q][rule] for q in sorted(ids)]
                summaries.append({'experiment':exp,'arm':arm,'regime':arm[0],
                    'stratum':name,'n':len(ids),'readout':rule,'correct':sum(v['em']==100 for v in values),
                    'em':sum(v['em'] for v in values)/len(ids),'f1':sum(v['f1'] for v in values)/len(ids)})
    def compare(exp_a,arm_a,exp_b,arm_b,label):
        a=grouped[exp_a,arm_a];b=grouped[exp_b,arm_b]
        for name,ids in members.items():
            for rule in RULES:
                contrasts.append({'contrast':label,'stratum':name,'n':len(ids),'readout':rule,
                    'em':prior.paired({q:a[q][rule]['em'] for q in ids},{q:b[q][rule]['em'] for q in ids},binary=True),
                    'f1':prior.paired({q:a[q][rule]['f1'] for q in ids},{q:b[q][rule]['f1'] for q in ids})})
    for regime in ('P','R'):
        for method in ('Broadcast','Learned_seed31'):
            compare('E5',regime+'_'+method,'E5',regime+'_NoComm',f'E5:{regime}_{method}-NoComm')
        compare('E5_projection',regime+'_InStar256_reused','E5',regime+'_NoComm',f'E5_projection:{regime}_InStar256-NoComm')
        compare('E5','%s_Learned_seed31'%regime,'E5',regime+'_Broadcast',f'E5:{regime}_Learned_seed31-Broadcast')
        compare('E5_projection',regime+'_InStar256_reused','E5',regime+'_Broadcast',f'E5_projection:{regime}_InStar256-Broadcast')
        for method in ('CW','CCW','SelfUpdate'):
            compare('E1',regime+'_'+method,'E1',regime+'_NoComm',f'E1:{regime}_{method}-NoComm')
    # Readout contrasts hold all saved agent responses fixed.
    readout_contrasts=[]
    for (exp,arm),qs in grouped.items():
        for name,ids in members.items():
            readout_contrasts.append({'experiment':exp,'arm':arm,'stratum':name,'n':len(ids),
                **{m:prior.paired({q:qs[q]['abstention_aware'][m] for q in ids},{q:qs[q]['original'][m] for q in ids},binary=m=='em') for m in ('em','f1')}})
    result={'status':'COMPLETE_OFFLINE_STRATIFICATION','new_provider_calls':0,'strata_counts':EXPECTED,
            'strata_definition':'E1 partitioned P initial annotated supporting-sentence availability and co-location; these same membership sets are used in E5 and for shared-R diagnostic rows.',
            'summaries':summaries,'paired_contrasts':contrasts,'readout_contrasts':readout_contrasts,
            'unavailable_requested_e1_arms':{'Broadcast':'NOT_SAVED_IN_E1','InStar256_reused':'NOT_SAVED_IN_E1','Learned_seed31':'NOT_SAVED_IN_E1'},
            'inference':'10,000 paired whole-question percentile bootstrap draws, seed 20261016; EM also has exact McNemar tests. Post hoc exploratory contrasts without multiplicity correction; no equivalence or causal mediation claim.',
            'overlap':'Gold-Complete is Disjoint union Co-located, not an independent sample. E1 and E5 use the same 300 question identities.',
            'verification':dict(verified)}
    for path,h in hashes.items():assert prior.sha(path)==h,path
    prior.save(out/'stratified_results.json',result);prior.jsonl(out/'question_strata.jsonl',assignments)
    with (out/'stratified_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(summaries)
    prior.save(out/'provenance.json',{'source_hashes':hashes,'script_sha256':prior.sha(__file__),'source_files_unchanged':True,'new_provider_calls':0})
    lines=['# Gold-evidence stratification of saved E1/E5 readouts','',
      'Zero provider calls. All 300 question identities retained. E1 initial partitioned evidence defines the strata: **Disjoint-Gold 62**, **Co-located-Gold 14**, **Retrieval-Missing-Gold 224**, and their combined **Gold-Complete 76**. E5 flags agree for all 300.','',
      '**Measured conclusion:** under abstention-aware readout, E5 Disjoint-Gold NoComm reaches 29/62 correct (F1 63.6753), versus Broadcast 31/62 (F1 67.1837). Broadcast minus NoComm is +3.2258 pp EM, 95% paired interval [-4.8387, +11.2903], exact McNemar p=0.6875; F1 is +3.5084 pp, interval [-2.9434, +10.5364]. Gold-complete absolute performance is much stronger than missing-gold performance, but a communication advantage remains uncertain after readout repair.','',
      'Membership uses annotated supporting sentence locators, not answer correctness. Gold completeness does not establish semantic necessity or actual use of evidence. A missing annotated source does not make correct answering impossible: the model may know a fact or find redundant evidence. Shared-R rows use E1-P membership for a common question slice; R itself does not partition those sources.','',
      '## Partitioned E5: exact EM counts and mean F1','',
      'Each cell is `correct / n; mean F1`. F1 is on the 0–100 scale. InStar256 is an exact-request projection reusing the saved Broadcast Hub output and initial non-hub outputs, not a new inference run.','']
    wanted=[('E5','P_NoComm'),('E5','P_Broadcast'),('E5_projection','P_InStar256_reused'),('E5','P_Learned_seed31')]
    lookup={(r['experiment'],r['arm'],r['stratum'],r['readout']):r for r in summaries}
    for name in STRATA:
        lines += ['### '+name,'','| Arm | Original vote | Abstention-aware vote | Fixed Hub 1 |','|---|---:|---:|---:|']
        for exp,arm in wanted:
            values=[lookup[exp,arm,name,rule] for rule in RULES]
            lines.append('| '+arm+' | '+' | '.join(f'{v["correct"]}/{v["n"]}; {v["f1"]:.4f}' for v in values)+' |')
        lines.append('')
    lines += ['## Partitioned paired contrasts','',
      'All differences are test minus baseline in percentage points. Parentheses contain paired 95% bootstrap intervals. P-values are exact McNemar tests for EM. These are exploratory unadjusted comparisons.','']
    for name in STRATA:
        lines += ['### '+name,'','| Contrast | Readout | EM difference (95% CI) | F1 difference (95% CI) | EM p |','|---|---|---:|---:|---:|']
        for c in contrasts:
            if c['stratum']!=name or ':P_' not in c['contrast']:continue
            def fmt(v):return f'{v["delta_pp"]:+.3f} [{v["ci95_pp"][0]:+.3f}, {v["ci95_pp"][1]:+.3f}]'
            lines.append(f'| {c["contrast"]} | {c["readout"]} | {fmt(c["em"])} | {fmt(c["f1"])} | {c["em"]["exact_mcnemar_p"]:.4g} |')
        lines.append('')
    lines += ['## E1: its actual saved partitioned arms','',
      'E1 contains NoComm, clockwise and counterclockwise rings, and SelfUpdate. Broadcast, learned seed31, and InStar256 were not saved in E1 and are not imputed.','',
      '| Stratum | Arm | Original vote | Abstention-aware vote | Fixed Hub 1 |','|---|---|---:|---:|---:|']
    for name in STRATA:
        for arm in ('P_NoComm','P_CW','P_CCW','P_SelfUpdate'):
            values=[lookup['E1',arm,name,rule] for rule in RULES]
            lines.append('| '+name+' | '+arm+' | '+' | '.join(f'{v["correct"]}/{v["n"]}; {v["f1"]:.4f}' for v in values)+' |')
    lines += ['','## Scope and reproducibility','',
      '- The disjoint slice is larger than the 2Wiki pilot, but that alone does not establish synthesis or a significant routing benefit.',
      '- Keep abstention-aware readout matched across arms; original-vote communication gains and readout gains are different contrasts.',
      '- E1 and E5 are paired analyses of the same identities, not independent replications.',
      '- Gold-Complete reuses the first two strata. Its intervals are not independent confirmations.',
      '- Fixed-Hub projection equality remains structural; a zero-width interval for reused outputs is not a statistical equivalence test.',
      '- Missing-gold classification is based on the retrieval pool, not proof that the omitted gold source was uniquely necessary.',
      '- P4/P5 provenance and proxy-token caveats in the existing synthesis remain unresolved. This stratification does not validate the proposed abstract\'s topology-invariance or lossless-halting claims.','',
      '[All 360 summary cells, including shared R](stratified_summary.csv) · [All paired intervals and readout contrasts](stratified_results.json) · [300 question memberships](question_strata.jsonl) · [Input hashes and verification](provenance.json)','',
      'Reproduce from repository root: `python -m WEBExtension.pilots.hotpot_readout_20261016.stratify`. No API backend is imported or called.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('STRATA',json.dumps(EXPECTED))
    for name in STRATA:
        for exp,arm in wanted:
            values=[lookup[exp,arm,name,rule] for rule in RULES]
            print(name,arm,' | '.join(f'{v["correct"]}/{v["n"]}; F1={v["f1"]:.4f}' for v in values))
    for c in contrasts:
        if c['contrast']=='E5:P_Broadcast-NoComm' and c['readout']=='abstention_aware':print('MATCHED_AA',c['stratum'],json.dumps({'em':c['em'],'f1':c['f1']}))
    print('VERIFIED',dict(verified),'SUMMARY_CELLS',len(summaries),'PAIRED_CONTRASTS',len(contrasts))


if __name__=='__main__':main()
