"""Read-only historical preservation and new artifact integrity checks."""
from pathlib import Path
import datetime,json,re,py_compile
from collections import Counter
from .reanalyse import ROOT,REPO,WEB,sha,save,jl


def main():
    protected=json.loads((ROOT/'historical_snapshot.json').read_text())
    for path,expected in protected.items():assert sha(path)==expected,path
    for manifest in [ROOT.parent/'artifact_manifest.json',ROOT.parent/'next_phase/artifact_manifest.json',ROOT.parent/'readout_reassessment/artifact_manifest.json']:
        for item in json.loads(manifest.read_text())['files']:
            path=Path(item['path']);path=path if path.is_absolute() else manifest.parent/path
            assert sha(path)==item['sha256'],path
    physical=json.loads((ROOT.parent/'readout_reassessment/results/physical_accounting.json').read_text())
    for path,expected in physical['source_hashes'].items():assert sha(path)==expected,path
    freeze=json.loads((ROOT/'frozen/freeze_hashes.json').read_text())
    for path,expected in freeze.items():assert sha(ROOT/path)==expected,path
    log=(ROOT/'results/full_tests.log').read_text();match=re.search(r'(\d+) passed, (\d+) subtests passed',log)
    assert match and 'failed' not in log.lower(),log[-2000:]
    assert int(match[1])==206 and int(match[2])==12
    for p in list(ROOT.glob('*.py'))+list((ROOT/'tests').glob('*.py')):py_compile.compile(str(p),doraise=True)
    old=json.loads((ROOT.parent/'readout_reassessment/results/matched_summary.json').read_text())
    prior={(r['arm'],{'original':'original','abstention_aware':'aware','abstention_lexical':'lexical'}[r['readout']]):r['correct'] for r in old}
    hist=json.loads((ROOT/'results/historical_four_readouts.json').read_text())
    assert all(r['correct']==prior[r['arm'],r['rule']] for r in hist['metrics'] if r['rule']!='strict_majority')
    raw=jl(WEB/'artifacts/controller_study_a0b3768357484b45/E5.jsonl')
    qresults=jl(ROOT/'results/historical_question_readouts.jsonl');assert len(qresults)==24000
    orig={(r['id'],r['arm']):r for r in qresults if r['rule']=='original'}
    from .contracts import hotpot_normalize
    for row in raw:
        for arm,state in row['conditions'].items():
            item=orig[row['id'],arm];assert item['correct']==(state['em']==100)
            assert hotpot_normalize(item['text'])==hotpot_normalize(state['answer'])
    ext=json.loads((ROOT/'results/external_gsm_results.json').read_text())
    assert len(ext['models'])==8 and all(r['current_source_extractor_mismatches']==0 for r in ext['models'])
    assert sum(r['counts']['agent_states'] for r in ext['models'])==36000
    assert sum(r['counts']['saved_numeric_parse_failures'] for r in ext['models'])==1
    assert len(jl(ROOT/'results/external_gsm_question_readouts.jsonl'))==48000
    ass=json.loads((ROOT/'results/question_assignment.json').read_text())
    splits=[set(ass[k]) for k in ['pilot_ids','annotation_ids','confirmatory_order']]
    assert [len(s) for s in splits]==[20,24,2366] and all(not(a&b) for i,a in enumerate(splits) for b in splits[i+1:])
    qrows=jl(ROOT/'sources/musique_full_v1.0_dev.jsonl')
    normq=lambda s:' '.join(re.findall(r'\w+',s.casefold()))
    by_id={r['id']:normq(r['question']) for r in qrows}
    assert len(set(by_id.values()))==2410 and len(ass['cross_ID_duplicate_groups'])==7
    selected=ass['pilot_ids']+ass['annotation_ids']+ass['confirmatory_order']
    assert len({by_id[q] for q in selected})==len(selected)==2410
    frozen=json.loads((ROOT/'frozen/question_manifest.json').read_text())['records']
    assert len({by_id[r['id']] for r in frozen if r['split']=='confirmation'})==1000
    assert Counter(r['id'] for r in frozen)==Counter({q:2 for q in ass['pilot_ids']+ass['confirmatory_order'][:1000]})
    oldq=json.loads((WEB/'data/hotpot_dev_fullwiki_v1.json').read_text())
    gsmq=jl(ROOT/'sources/moozwen__sakana-debate/experiments/phase1b_1c_regime_budget/data/gsm8k_subset_regime.jsonl')
    overlaps=set(by_id.values())&{normq(r['question']) for r in oldq+gsmq}
    assert not overlaps
    save('results/question_identity_audit.json',{'MuSiQue_paired_ids':2417,'unique_normalized_question_texts':2410,'duplicate_ID_groups':7,
        'confirmation_unique_text_units':1000,'disjoint_pilot_annotation_confirmation':True,
        'normalized_question_overlap_with_local_7405_Hotpot_and_external_500_GSM':0,'pretraining_exposure_not_assessed':True})
    for reviewer in ['A','B']:
        blind=jl(ROOT/f'annotation/blind_reviewer_{reviewer}.jsonl');labels=jl(ROOT/f'annotation/labels_{reviewer}.jsonl')
        assert len(blind)==len(labels)==48 and all(r['source_sufficiency']=='PENDING' for r in labels)
        assert {r['case_id'] for r in blind}=={r['case_id'] for r in labels}
        assert all(set(r)=={'case_id','question','sources'} and all(set(s)=={'source_id','title','text'} for s in r['sources']) for r in blind)
    power=json.loads((ROOT/'results/power_sensitivity.json').read_text())
    assert all('bounded_MC' not in r for r in power['grid'] if r['SD_upper_assumption']==2)
    assert all(r['bounded_MC']['support']==[-1,0,1] for r in power['grid'] if 'bounded_MC' in r)
    ledger=ROOT.parent/'paid_pilot/ledger.jsonl';e6=jl(ledger);assert sum(r['event']=='success' for r in e6)==144
    reports=sorted(ROOT.glob('0[1-8]_*.md'));assert len(reports)==8
    assert all((ROOT/'reviews'/f'{name}.md').exists() for name in ['importance','design','novelty'])
    audit={'status':'COMPLETE_OFFLINE_REVISE_NO_PAID_PILOT','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scientific_decision':'REVISE','current_contribution_category':'C_LIMITED_IMPLEMENTATION_AUDIT',
        'new_paid_inference_completions':0,'new_Azure_requests':0,'new_paid_judges':0,'PPO_training_runs':0,
        'public_HTTP_downloads':'Performed and recorded; not inference','historical_files_byte_identical':len(protected),
        'historical_physical_source_hashes_verified':len(physical['source_hashes']),
        'previous_E6_successes_unchanged':144,'previous_E6_ledger_sha256':sha(ledger),
        'tests_passed':int(match[1]),'subtests_passed':int(match[2]),'new_tests':19,
        'same_pipeline_counts_matched':60,'historical_question_readout_records':24000,
        'external_configs':8,'external_unique_question_ids':500,'external_agent_states':36000,'external_offline_readout_records':48000,
        'external_source_parser_disagreements':0,'external_saved_final_votes_and_grades_reconstructed':4000,
        'independent_semantic_Web_QA_interaction_replication':'NOT_ESTABLISHED','new_MuSiQue_model_outcomes':'NOT_RUN',
        'MuSiQue_records':4834,'MuSiQue_unique_paired_question_ids':2417,'MuSiQue_unique_question_text_units':2410,'preinference_duplicate_ID_exclusions':7,'prepared_blinded_context_cases':48,'independent_semantic_labels':'PENDING',
        'reviews':'Three independent automated perspectives plus final evidence addenda; not human peer review',
        'conditional_confirmation_questions':1000,'conditional_pilot_questions':20,'inference_launch_authorized':False,
        'fully_configured_deployed_runtime':'NOT_ESTABLISHED','minimum_inference_recommended_now':0,
        'requested_reports':[p.name for p in reports]}
    save('COMPLETION_AUDIT.json',audit)
    for report in reports+[ROOT/'README.md']:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',report.read_text()):
            if target.startswith(('https://','http://','#')):continue
            path=re.sub(r':\d+$','',target.strip('<>'));assert Path(path).is_absolute() and Path(path).exists(),target
    files=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(ROOT.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p!=ROOT/'artifact_manifest.json']
    save('artifact_manifest.json',{'scope':'New external_validation only; excludes caches and manifest itself','n_files':len(files),'files':files})
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
