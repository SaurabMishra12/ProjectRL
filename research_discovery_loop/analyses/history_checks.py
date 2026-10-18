"""Additional bounded historical checks; reads only, zero inference."""
import json, re, math
from collections import Counter,defaultdict
from pathlib import Path
from statistics import NormalDist
from offline_audit import ROOT,REPO,OUT,load,INPUTS,effect,normalize,vote

def main():
    result={};web={}
    for name in ('W1_web_cost_accuracy','W2_context_rationing','W3_corrected','W4_corrected'):
        rows=load('results/'+name+'/raw_results.jsonl')
        arms=[k for k,v in rows[0].items() if isinstance(v,dict) and 'f1' in v]
        web[name]={'n':len(rows),'unique_ids':len({r['id'] for r in rows}),
          'means':{a:{k:sum(float(r[a].get(k,0)) for r in rows)/len(rows) for k in ('f1','comm_tokens','rounds','latency_sec')} for a in arms}}
        if name=='W4_corrected':
            web[name]['learned_minus_static']=effect([r['Learned_SparseComm']['f1']-r['Static_Role']['f1'] for r in rows])
            web[name]['learned_minus_random']=effect([r['Learned_SparseComm']['f1']-r['Random_Routing']['f1'] for r in rows])
    result['legacy_web_raw']=web
    e8=load('WEBExtension/causal_study/offline/e8_question_conditions.jsonl');groups=defaultdict(list)
    for r in e8:groups[r['experiment']+':'+r['condition']].append(r)
    result['E8']={'rows':len(e8),'unique_ids':len({r['id'] for r in e8}),
       'groups':{k:{'n':len(v),'observability':dict(Counter(r['observability'] for r in v)),
                    'initial_any_correct':sum(bool(r['initial_any_em']) for r in v),
                    'final_any_correct':sum(bool(r['final_any_em']) for r in v),
                    'correct_agent_degraded':sum(bool(r['any_initial_correct_agent_became_incorrect']) for r in v),
                    'aggregator_rejected_correct':sum(bool(r['aggregator_rejected_correct_candidate']) for r in v)} for k,v in groups.items()}}
    obs=load('WEBExtension/causal_study/next_phase/source_artifacts/source_observations.jsonl')
    result['source_metadata']={cohort:{'n':sum(r['cohort']==cohort for r in obs),
      'annotated_support_complete':sum(r['all_annotated_support_present'] for r in obs if r['cohort']==cohort),
      'semantic_necessity_values':dict(Counter(r['both_articles_necessary_for_answer'] for r in obs if r['cohort']==cohort))}
      for cohort in sorted({r['cohort'] for r in obs})}
    ext=REPO/'WEBExtension/causal_study/external_validation/sources/moozwen__sakana-debate/experiments/phase1b_1c_regime_budget/results'
    models={};allids=[]
    for p in sorted(ext.glob('*/debate_N3_R2_seed1.jsonl')):
        rows=load(str(p.relative_to(REPO)));allids.append({r['idx'] for r in rows})
        assert all(len(r['rounds'])==3 for r in rows)
        preds=[z for r in rows for rd in r['rounds'] for z in rd['preds']]
        assert all(z is None or type(z) is float for z in preds)
        keys=list(rows[0]['rounds'][0]);models[p.parent.name]={'n':len(rows),'unique_ids':len(allids[-1]),'round_keys':keys,
          'numeric_parser_missing':sum(z is None for z in preds),
          'semantic_abstention_vote_candidates':0,
          'raw_empty':sum(not str(t).strip() for r in rows for rd in r['rounds'] for t in rd['raw']),
          'saved_correct':sum(bool(r['ok']) for r in rows)}
    result['external_trace_inventory']={'models':models,'unique_question_union':len(set.union(*allids)),
                                        'shared_questions':len(set.intersection(*allids))}
    ms=load('WEBExtension/causal_study/external_validation/sources/musique_full_v1.0_dev.jsonl')
    texts=defaultdict(set);idcounts=Counter()
    for r in ms:texts[' '.join(re.findall(r'\w+',r['question'].casefold()))].add(r['id']);idcounts[r['id']]+=1
    result['MuSiQue']={'records':len(ms),'paired_ids':len(idcounts),'distinct_question_texts':len(texts),
                       'cross_id_duplicate_groups':[sorted(s) for s in texts.values() if len(s)>1],
                       'all_ids_two_records':set(idcounts.values())=={2},'confirm_pool_after_20_and_24':len(texts)-44}
    z=NormalDist();factor=(z.inv_cdf(1-.025/2)+z.inv_cdf(.8))**2
    power=[{'sd':sd,'effect':d,'n_approx':math.ceil(factor*sd**2/d**2),'alpha_two_sided':.025,'target_power':.8}
           for sd in (.25,.5,1.,2.) for d in (.02,.05)]
    result['normal_approx_power_sensitivity']=power
    audit=json.loads((OUT/'offline_audit.json').read_text());t=audit['E6_timing'];inp=t['usage']['input_tokens']/144;out=t['usage']['output_tokens']/144
    result['budget_sensitivity']=[{'successful_calls':n,'input_token_reference':n*inp,'output_token_reference':n*out,
      'serial_request_seconds_reference':n*t['request_latency_mean_seconds'],
      'amortized_wall_seconds_reference':n*t['amortized_wall_seconds_per_success'],
      'monetary_formula':'input_tokens/1e6 * price_input + output_tokens/1e6 * price_output; prices unverified'} for n in (720,18000,36000,68472)]
    (OUT/'history_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'history_input_hashes.json').write_text(json.dumps(INPUTS,indent=2)+'\n')
    print(json.dumps({'E8_rows':len(e8),'E8_groups':len(groups),'MuSiQue':result['MuSiQue'],
      'external_shared_questions':result['external_trace_inventory']['shared_questions']},indent=2))

if __name__=='__main__':main()
