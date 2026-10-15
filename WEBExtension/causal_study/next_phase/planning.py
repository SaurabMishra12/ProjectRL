"""Offline paired-binary inference and cost/power sensitivity; no inference calls."""
from __future__ import annotations
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binom,norm

from WEBExtension.causal_study.provider import atomic_json

ROOT=Path(__file__).resolve().parent


def exact_mcnemar(gains,losses):
    if type(gains) is not int or type(losses) is not int or min(gains,losses)<0:raise ValueError('Nonnegative paired counts required')
    if gains+losses==0:return 1.
    return min(1.,2*float(binom.cdf(min(gains,losses),gains+losses,.5)))


def paired_analysis(rows,draws=10000,seed=20261015):
    if not rows or len({r['question_id'] for r in rows})!=len(rows):raise ValueError('Unique nonempty paired question set required')
    for r in rows:
        if any(k not in r for k in ['recorded_em','exchanged_em','recorded_f1','exchanged_f1']):raise ValueError('Incomplete paired cells')
        if r['recorded_em'] not in (0,100) or r['exchanged_em'] not in (0,100):raise ValueError('EM must be0or100')
        if any(not isinstance(r[k],(int,float)) or not math.isfinite(r[k]) or not 0<=r[k]<=100 for k in ['recorded_f1','exchanged_f1']):raise ValueError('F1 outside0..100')
    em=np.array([r['recorded_em']-r['exchanged_em'] for r in rows],float)
    f1=np.array([r['recorded_f1']-r['exchanged_f1'] for r in rows],float)
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(rows),size=(draws,len(rows)))
    gains=int(sum(x==100 for x in em));losses=int(sum(x==-100 for x in em))
    return {'n_questions':len(rows),'em_delta_pp':float(em.mean()),'f1_delta_pp':float(f1.mean()),
            'em_ci95_pp':np.quantile(em[indices].mean(axis=1),[.025,.975]).tolist(),
            'f1_ci95_pp':np.quantile(f1[indices].mean(axis=1),[.025,.975]).tolist(),
            'gains':gains,'losses':losses,'ties':len(rows)-gains-losses,
            'exact_mcnemar_p':exact_mcnemar(gains,losses),'sampling_unit':'unique_question',
            'primary':'paired_EM','secondary_F1':'descriptive_no_confirmatory_claim'}


def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,mid-half),min(1.,mid+half)]


def power_grid(effects=(.03,.05,.10),discordances=(.1,.25,.5),
               sample_sizes=(25,50,100,200,400,800,1600,3200),simulations=5000,seed=20261015):
    rng=np.random.default_rng(seed);rows=[]
    if simulations<1:raise ValueError('Positive simulation count required')
    for delta in effects:
        for discordance in discordances:
            if not 0<delta<=discordance<=1:raise ValueError('Need0<effect<=discordance<=1')
            probabilities=[(discordance+delta)/2,(discordance-delta)/2,1-discordance]
            for n in sample_sizes:
                if type(n) is not int or n<2:raise ValueError('Need at least two questions')
                draws=rng.multinomial(n,probabilities,size=simulations)
                gain,loss=draws[:,0],draws[:,1]
                p=np.minimum(1.,2*binom.cdf(np.minimum(gain,loss),gain+loss,.5))
                count=int(np.sum(p<=.05))
                rows.append({'effect_em_fraction':delta,'discordance':discordance,'n_questions':n,
                  'estimated_power':count/simulations,'power_ci95':wilson(count,simulations),
                  'simulations':simulations,'alpha':.05,'method':'exact_conditional_mcnemar_monte_carlo',
                  'candidate_finalizer_calls':2*n,'assumption':'iid paired binary question outcomes; hypothetical rates'})
    return rows


def fresh_pool():
    from WEBExtension.protocol import load_official,normalize
    from WEBExtension.causal_study.run import exposure_rows
    rows=load_official();blocked,_=exposure_rows()
    previous=json.loads((ROOT.parent/'frozen/manifest.json').read_text())
    blocked+=previous['questions']
    ids={r.get('id',r.get('_id')) for r in blocked};questions={normalize(r['question']) for r in blocked if r.get('question')}
    available=[]
    for row in rows:
        if row['id'] in ids or normalize(row['question']) in questions:continue
        docs=row['documents']
        if len(docs)<6 or len({d['title'] for d in docs})!=len(docs):continue
        if any(not d['sentences'] or any(not isinstance(s,str) or not s.strip() for s in d['sentences']) for d in docs):continue
        available.append(row['id'])
    return {'gold_free_source_capable_fresh_questions':len(available),'ids':available,
            'not_a_semantically_qualified_or_proposal_eligible_count':True,
            'eligibility':'unexposed exact IDs/normalized questions, >=6 unique nonempty source articles; no gold-support filter'}


def usage_reference(tok):
    directory=ROOT.parent/'paid_pilot'
    events=[json.loads(x) for x in (directory/'ledger.jsonl').read_text().splitlines()]
    success=[e for e in events if e['event']=='success']
    output=[];overhead=[];initial_input=[];initial_output=[];latency=[]
    for e in success:
        saved=json.loads((directory/'attempts'/(e['attempt_id']+'.json')).read_text())
        content=sum(tok.count(m['content']) for m in saved['request']['messages'])
        overhead.append(e['usage']['prompt_tokens']-content);output.append(e['usage']['completion_tokens']);latency.append(saved['latency_seconds'])
        if saved['metadata']['stage']=='initial':initial_input.append(e['usage']['prompt_tokens']);initial_output.append(e['usage']['completion_tokens'])
    return {'e6_calls':len(success),'e6_input_tokens':sum(e['usage']['prompt_tokens'] for e in success),
            'e6_output_tokens':sum(output),'e6_session_seconds':289.4476947784424,
            'mean_output_tokens':float(np.mean(output)),'output_p10_p50_p90':np.quantile(output,[.1,.5,.9]).tolist(),
            'provider_minus_content_tokens_min_mean_max':[int(min(overhead)),float(np.mean(overhead)),int(max(overhead))],
            'initial_mean_input_tokens':float(np.mean(initial_input)), 'initial_mean_output_tokens':float(np.mean(initial_output)),
            'mean_http_latency_seconds':float(np.mean(latency)),'observed_serial_seconds_per_call':289.4476947784424/len(success),
            'new_finalizer_latency_measured':False,'new_finalizer_usage_measured':False}


def csv_rows(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def main():
    from WEBExtension.causal_study.run import tokenizer
    from .study import verify_prepared
    manifest=verify_prepared();tok=tokenizer();reference=usage_reference(tok)
    directory=ROOT/'planning_artifacts';directory.mkdir(exist_ok=True)
    pool=fresh_pool();grid=power_grid();yield_ci=wilson(manifest['mechanically_eligible_cases'],manifest['source_cases'])
    yield_rate=manifest['mechanically_eligible_cases']/manifest['source_cases']
    prompt_lengths=[q['prompt_tokens'][a] for q in manifest['questions'] for a in ['recorded','exchanged']]
    finalizer_input_mean=float(np.mean(prompt_lengths))+reference['provider_minus_content_tokens_min_mean_max'][1]
    pilot={'questions':len(manifest['questions']),'cached_proposals_reused':3*len(manifest['questions']),
        'new_proposal_calls':0,'finalizer_calls':len(prompt_lengths),'counterbalanced_extra_repetitions':0,
        'paid_validation_calls':0,'judge_calls':0,'suggested_future_completion_ceiling':24,'suggested_future_attempt_ceiling':30,
        'retry_attempt_reserve':6,'projected_input_tokens':math.ceil(sum(prompt_lengths)+len(prompt_lengths)*reference['provider_minus_content_tokens_min_mean_max'][1]),
        'reference_output_tokens':math.ceil(len(prompt_lengths)*reference['mean_output_tokens']),
        'output_tokens_reference_p10_p90':[math.ceil(len(prompt_lengths)*x) for x in [reference['output_p10_p50_p90'][0],reference['output_p10_p50_p90'][2]]],
        'hard_output_token_cap':2048*len(prompt_lengths),
        'runtime_reference_seconds':len(prompt_lengths)*reference['observed_serial_seconds_per_call'],
        'runtime_unmeasured_finalizer_latency':True,'monetary_cost':None,'paid_authorized':False}
    recommendations=[]
    for delta in [.03,.05,.1]:
        for discordance in [.1,.25,.5]:
            for target in [.8,.9]:
                candidates=[r for r in grid if r['effect_em_fraction']==delta and r['discordance']==discordance and r['power_ci95'][0]>=target]
                selected=min(candidates,key=lambda r:r['n_questions']) if candidates else None
                approx=math.ceil((norm.ppf(.975)+norm.ppf(target))**2*(discordance-delta**2)/delta**2)
                n=selected['n_questions'] if selected else None
                screened=math.ceil(n/yield_rate) if n else None
                screening_calls=3*screened if screened else None;finalizer_calls=2*n if n else None
                calls=screening_calls+finalizer_calls if n else None
                recommendations.append({'effect_em_fraction':delta,'discordance':discordance,'target_power':target,
                  'smallest_simulated_n':n,'normal_approximation_n':approx,
                  'cached_available_20_is_sufficient':bool(n and n<=manifest['mechanically_eligible_cases']),
                  'assumed_conflict_yield':yield_rate,'yield_wilson_ci95':yield_ci,
                  'expected_source_questions_to_screen':screened,
                  'screening_source_questions_yield_ci_range':[math.ceil(n/yield_ci[1]),math.ceil(n/yield_ci[0])] if n else None,
                  'proposal_screening_calls':screening_calls,'finalizer_calls':finalizer_calls,
                  'expected_total_completions':calls,'attempt_ceiling_with_25pct_retry_reserve':math.ceil(calls*1.25) if calls else None,
                  'expected_input_tokens':math.ceil(screening_calls*reference['initial_mean_input_tokens']+finalizer_calls*finalizer_input_mean) if n else None,
                  'reference_output_tokens':math.ceil(screening_calls*reference['initial_mean_output_tokens']+finalizer_calls*reference['mean_output_tokens']) if n else None,
                  'serial_runtime_reference_hours':calls*reference['observed_serial_seconds_per_call']/3600 if calls else None,
                  'within_fresh_source_pool_at_point_yield':bool(screened and screened<=pool['gold_free_source_capable_fresh_questions']),
                  'monetary_cost':None,'status':'SENSITIVITY_NOT_AUTHORIZATION' if n else 'NOT_REACHED_IN_GRID'})
    summary={'status':'PLANNING_ONLY_NO_NEW_INFERENCE','reference':reference,'pilot':pilot,'cached_conflict_yield':yield_rate,
      'cached_yield_ci95':yield_ci,'fresh_pool':{k:v for k,v in pool.items() if k!='ids'},
      'finalizer_prompt_content_tokens_min_mean_max':[min(prompt_lengths),float(np.mean(prompt_lengths)),max(prompt_lengths)],
      'finalizer_input_mean_using_e6_overhead':finalizer_input_mean,
      'power_assumptions':'No E6 interaction SD extrapolation. Hypothetical paired EM effects/discordance; exact McNemar simulations.',
      'limitations':['Finalizer token/latency distribution unmeasured; E6 is a reference, not a calibrated forecast.',
                    'Cached conflict yield is post-inspection and not measured on fresh sources; Wilson interval does not cover distribution shift.',
                    'Question/citation semantic qualification and novelty are unresolved; pool count is not an accepted cohort.',
                    'Counterbalanced assignments stay clustered within question; no repeated orders included in minimum schedule.',
                    'Source or decoding failures consume quota; incomplete pairs are not wrong predictions.',
                    'No actual deployment tariff or billing export; monetary projections are unknown.']}
    atomic_json(directory/'summary.json',summary);atomic_json(directory/'fresh_pool.json',pool)
    atomic_json(directory/'power_grid.json',grid);atomic_json(directory/'recommendations.json',recommendations)
    csv_rows(directory/'power_grid.csv',grid);csv_rows(directory/'recommendations.csv',recommendations)
    lines=['# API budget and power analysis','', '**Status: planning only; new paid completions=0, new paid HTTP attempts=0.** The six unused E6 completion slots are not used or reassigned.','',
       '## Exact architecture and pilot proposal','',
       f"The cached diagnostic freezes{pilot['questions']} questions and reuses{pilot['cached_proposals_reused']} initial responses. It needs exactly{pilot['finalizer_calls']} finalizer calls, no new initial calls, no peer explanations, no extra candidate-order repetitions, no paid validation calls, and no judges. A separately approved pilot could cap successful completions at24 and HTTP attempts at30;6 attempts are reserved for explicit429 retries, not extra outcomes. Current code forbids paid execution.",'',
       f"Pinned-tokenizer content counts for the24 actual prepared prompts range{min(prompt_lengths)}–{max(prompt_lengths)}, mean{np.mean(prompt_lengths):.2f}. The two arms are exactly matched per question. Measured E6 provider-minus-content overhead is{reference['provider_minus_content_tokens_min_mean_max']}; applying its mean gives a pilot input projection of{pilot['projected_input_tokens']} tokens. It is still an estimate of server framing, not actual future usage.",'',
       f"E6 actually used92,594 input/34,564 output tokens in144 calls and289.45s. Mean output was{reference['mean_output_tokens']:.2f}; p10/median/p90 were{reference['output_p10_p50_p90']}. Reusing only those observed output references gives approximately{pilot['reference_output_tokens']} pilot output tokens, with an empirical-reference p10–p90 scenario of{pilot['output_tokens_reference_p10_p90']}. These are not prediction intervals for the new selector. The hard output cap is{pilot['hard_output_token_cap']} tokens, with2048/request. The finalizer can have different reasoning length or truncate; no new observations exist.",'',
       f"E6 serial timing implies a{pilot['runtime_reference_seconds']:.1f}s pilot reference at2.01s/call; a2s dispatch floor would be retained for any future live run. Changed prompt lengths, provider queueing and429s may increase latency. No verified deployment tariff or invoice is available: expected and actual monetary cost remain **UNKNOWN**, not zero. No model switch is planned.",'',
       '## Prospective power model','',
       'Primary outcome is paired binary selector EM, not the old E6 F1 interaction. Assume a true paired EM difference of3,5 or10 percentage points and total paired discordance of0.10,0.25 or0.50. For each question, gain/loss probabilities are(q+delta)/2 and(q−delta)/2. Simulate5,000 studies per scenario and apply the exact two-sided conditional McNemar test at0.05. Recommendations require the lower Monte Carlo95% bound to reach80% or90%. Simulation uncertainty is not uncertainty about the assumed population rates. All input rates are explicitly hypothetical; no new effect or variance has been measured.','',
       f"The pilot has only12 questions and the saved mechanical pool only20. Neither is a confirmatory power sample. Cached eligible yield is20/300={yield_rate:.4f}, Wilson95% CI{yield_ci}. Fresh source-capable questions available under gold-free document validation: **{pool['gold_free_source_capable_fresh_questions']}**. This differs from the old1,927 complementarity-screened upper bound because the association experiment does not require both annotated supports. Neither count guarantees substantive conflicts or semantic validity.",'',
       '## Cost of finding eligible fresh states','',
       'Cold confirmation must generate three initial responses for every screened source question, including those that fail eligibility. Under the point yield, each accepted question costs about45 proposal completions plus2 finalizers, not5 total calls. The number of screened questions is stochastic; point-yield projections are expectations, not spending guarantees. Predefine an absolute screening budget and stop without replacing cases when it is exhausted. Fresh response yield and semantic qualification may be worse.','']
    for r in recommendations:
        if r['effect_em_fraction']==.05 and r['discordance']==.25:
            lines.append(f"- Five-point effect, discordance0.25, {100*r['target_power']:g}% power: n={r['smallest_simulated_n']} accepted questions; expected screening={r['expected_source_questions_to_screen']} source questions, proposal calls={r['proposal_screening_calls']}, finalizer calls={r['finalizer_calls']}, total={r['expected_total_completions']}; input≈{r['expected_input_tokens']}, output-reference≈{r['reference_output_tokens']}, serial runtime≈{r['serial_runtime_reference_hours']:.2f}hours. Fits current source pool at point yield: **{r['within_fresh_source_pool_at_point_yield']}**. Money unknown.")
    lines.extend(['','Every3/5/10-point and discordance scenario,80%/90% target, yield range, actual call decomposition and source-pool flag is saved in `planning_artifacts/recommendations.csv`. Token projections use measured E6 initial-call means for proposal screening and actual prepared prompt sizes for selectors; source/context distribution shift remains unmeasured. No optional judge or explanation costs are hidden. Larger target samples can exceed the fresh source pool long before the nominal accepted-question count does.','',
       '## Decision','',
       'The minimum conditional mechanism-validation pilot is24 new completions/30 attempted requests, only after separate authorization and resolution of material source/novelty/review objections. The minimum actual expenditure justified **now is0**. No precise larger budget is justified from this phase: independent semantic annotation, a clear scientific distinction, a chosen minimum important effect, a credible fresh conflict yield and feasibility must precede commitment. Planning scenarios are not approval to run300/600 questions, train PPO, spend old quota, or change deployments.'])
    (ROOT/'05_API_BUDGET_AND_POWER.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'pilot':pilot,'fresh_pool':summary['fresh_pool'],'five_point_discordance_025':[r for r in recommendations if r['effect_em_fraction']==.05 and r['discordance']==.25]},indent=2))


if __name__=='__main__':main()
