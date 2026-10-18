"""Question-cluster interaction power sensitivity, and physical-usage budgets.
No historical seven-point estimate is used as an assumed future effect.
"""
from pathlib import Path
import json,math
import numpy as np
from scipy.stats import norm,t
from .reanalyse import ROOT,save,sha
ALPHA_PER_BACKBONE=.025
MIN_MEANINGFUL=.05


def n_normal(delta,sd,power=.8):
    if not 0<delta<1 or sd<=0 or not .5<power<1:raise ValueError('Positive valid design parameters required')
    return math.ceil(((norm.ppf(1-ALPHA_PER_BACKBONE/2)+norm.ppf(power))*sd/delta)**2)


def simulated_power(n,delta,second_moment,draws=20000):
    # Each draw samples n complete question-level interactions, after averaging
    # the two context variants. This is NOT an EM discordance/McNemar model.
    if second_moment>1:raise ValueError('No proposal-realizability claim for the loose algebraic bound')
    bound=1
    mass=second_moment/bound**2
    probabilities=[(mass-delta/bound)/2,1-mass,(mass+delta/bound)/2]
    if min(probabilities)<0:raise ValueError('Infeasible bounded interaction distribution')
    rng=np.random.default_rng(20261016)
    counts=rng.multinomial(n,probabilities,size=draws)
    means=bound*(counts[:,2]-counts[:,0])/n
    second=bound**2*(counts[:,2]+counts[:,0])/n
    variance=np.maximum(0,(second-means**2)*n/(n-1))
    se=np.sqrt(variance/n)
    significant=np.abs(means)>t.ppf(1-ALPHA_PER_BACKBONE/2,n-1)*se
    estimate=float(significant.mean())
    return {'simulated_power':estimate,'MC_standard_error':math.sqrt(estimate*(1-estimate)/draws),
        'draws':draws,'support':[-bound,0,bound],'probabilities':probabilities,'true_variance':second_moment-delta**2,
        'method':'two-sided studentized mean of the exact prespecified question-level contrast; sparse bounded sensitivity, not a forecast',
        'realizability':'J=+1: sufficient peer has [abstain,abstain,correct] and private [correct,correct,correct]; insufficient peer [abstain,abstain,abstain], private [abstain,abstain,wrong]. J=-1 swaps peer/private in both contexts. J=0 uses identical arms.'}


def main():
    assignment=json.loads((ROOT/'results/question_assignment.json').read_text());pool=len(assignment['confirmatory_order'])
    grid=[]
    for delta in [.02,.05,.10]:
        for sd in [.25,.5,1,2]:
            for power in [.8,.9]:
                n=n_normal(delta,sd,power)
                row={'minimum_meaningful_absolute_interaction':delta,'SD_upper_assumption':sd,'target_power':power,'unique_questions':n,
                    'successful_completions_per_backbone':18*n,'successful_completions_two_backbones':36*n,'available_confirmation_questions':pool,'fits_pool':n<=pool}
                # At delta .1, SD .25's ±1 distribution is impossible; no made-up simulation.
                if sd<=1 and sd**2>=delta and n>=3:row['bounded_MC']=simulated_power(n,delta,sd**2)
                grid.append(row)
    save('results/power_sensitivity.json',{'primary':'mean_q 0.5*(D_q,sufficient + D_q,insufficient), D=(U_peer,aware-U_private,aware)-(U_peer,original-U_private,original)',
        'unit':'unique question; both contexts and backbones remain clustered; model-specific tests, Bonferroni .025 each controls .05 family',
        'universal_bound_for_averaged_four_binary_cell_contrast':[-2,2],
        'minimum_meaningful_effect':MIN_MEANINGFUL,'effect_is_not_historical_estimate':True,'grid':grid,
        'caution':'Normal sample sizes are sensitivity guides, not a guaranteed powered design. MC is one bounded distribution per variance, not distribution-free assurance. Worst-case variance sensitivity exceeds the data pool.'})
    physical_path=ROOT.parent/'readout_reassessment/results/physical_accounting.json'
    physical=json.loads(physical_path.read_text());arms={r['arm']:r for r in physical['arms']}
    references=[]
    for regime in ['P','R']:
        initial=arms[regime+'_NoComm'];private=arms[regime+'_SelfUpdate'];peer=arms[regime+'_Broadcast']
        # Each arm includes the exact same initial requests. Deduplicate algebraically
        # for the 3initial+3private+3peer architecture; don't sum overlapping arm totals.
        inputs=private['observed_input_tokens_lower_bound']+peer['observed_input_tokens_lower_bound']-initial['observed_input_tokens_lower_bound']
        outputs=private['observed_output_tokens_lower_bound']+peer['observed_output_tokens_lower_bound']-initial['observed_output_tokens_lower_bound']
        references.append({'historical_regime':regime,'questions':300,'architecture_successes':2700,'input_tokens_known':inputs,
            'output_tokens_lower_bound':outputs,'input_per_question_context':inputs/300,'output_lower_bound_per_question_context':outputs/300,
            'missing_output_observations_not_imputed_zero':private['successful_attempts_missing_output_usage']+peer['successful_attempts_missing_output_usage']-initial['successful_attempts_missing_output_usage']})
    budgets=[]
    for purpose,n in [('operational_pilot',20),('illustrative_minimum_confirmation_SD_0.5_80pct',n_normal(.05,.5)),('conditional_frozen_confirmation',1000),('full_available_confirmation_pool',pool)]:
        successes=36*n;reserve=math.ceil(successes*.1)
        budgets.append({'purpose':purpose,'unique_questions':n,'contexts_per_question':2,'backbones':2,'successful_completions':successes,
            'failure_attempt_reserve_10pct':reserve,'hard_total_attempt_quota':successes+reserve,'reserve_is_not_guaranteed_success_probability':True,
            'all_success_output_token_cap':2048*successes,'all_attempt_output_token_cap':2048*(successes+reserve),
            'all_attempt_input_token_cap':32768*(successes+reserve),
            'historical_reference_scaled_tokens':[{'regime':r['historical_regime'],'context_size_multiplier':scale,'input_reference':math.ceil(r['input_per_question_context']*2*2*n*scale),
                'output_reference_lower_bound':math.ceil(r['output_lower_bound_per_question_context']*2*2*n)} for r in references for scale in [1,2,4]]})
    save('results/api_budget.json',{'new_paid_calls':0,'inference_api_calls_for_offline_work':0,'price_status':'Not quoted: no current provider deployment, price tier or version lock; dollars = input_tokens*input_price_per_million/1e6 + output_tokens*output_price_per_million/1e6 including failed/uncertain billed attempts.',
        'source':str(physical_path),'source_sha256':sha(physical_path),'historical_architecture_references':references,'budgets':budgets,
        'latency':'E6 144 successes /289.4476947784424 summed seconds =>2.01005343596 s/call historical serial reference only. MuSiQue long context, backbone and parallelism make forecasts unjustified.',
        'pilot_is_not_powered':True,'pilot_not_included_in_confirmation':True,'no_spending_recommended_now':True})
    print('Saved exact-interaction variance sensitivity and budgets; no API execution path.')
if __name__=='__main__':main()
