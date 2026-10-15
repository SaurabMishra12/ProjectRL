"""Offline analysis only. Never imports a transport or invokes inference.

Rebuild all new-study tables, figures and decision reports from saved artifacts:
``python3 -m WEBExtension.causal_study.analyze``.
"""
from __future__ import annotations
import csv
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from .design import validate_manifest
from .provider import BudgetBackend,atomic_json,digest
from .statistics import analyze_e6,power_sensitivity
from WEBExtension.protocol import score,normalize

ROOT=Path(__file__).resolve().parent


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(key for row in rows for key in row))
    if not keys: return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def collect():
    manifest=json.loads((ROOT/'frozen/manifest.json').read_text())
    if manifest['manifest_sha256']!=digest({k:v for k,v in manifest.items() if k!='manifest_sha256'}):
        raise ValueError('Manifest checksum mismatch')
    assignment=validate_manifest(manifest)
    directory=ROOT/'paid_pilot'
    status=json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else {'status':'NOT_RUN'}
    config=json.loads((ROOT/'frozen/config.json').read_text())
    # Read-only summary even if uncertainty correctly blocks backend startup.
    backend=BudgetBackend.__new__(BudgetBackend)
    backend.ledger=directory/'ledger.jsonl';backend.config=config
    usage=backend.summary()
    completed=[];incomplete=[]
    for q in manifest['questions']:
        path=directory/'questions'/(q['id']+'.json')
        if not path.exists():incomplete.append(q['id']);continue
        record=json.loads(path.read_text())
        if record['content_hash']!=digest({k:v for k,v in record.items() if k!='content_hash'}):
            raise ValueError('Question checksum mismatch')
        label=manifest['private']['labels'][q['id']]
        for name,cell in record['cells'].items():
            expected=score(cell['answer'],label['answer'])
            if expected['f1']!=cell['f1'] or expected['em']!=cell['em']:
                raise ValueError('Saved score differs from official-compatible recomputation')
        completed.append(record)
    events=backend._records()
    attempts=[]
    for event in events:
        if event['event']!='started':continue
        aid=event['attempt_id'];path=directory/'attempts'/(aid+'.json')
        response=json.loads(path.read_text()) if path.exists() else {}
        raw=response.get('raw') or {};raw=raw if isinstance(raw,dict) else {}
        provider_usage=raw.get('usage') or {}
        outcomes=[e for e in events if e.get('attempt_id')==aid and e['event']!='started']
        attempts.append({'attempt_id':aid,'request_hash':event['request_hash'],**event.get('metadata',{}),
                         'utc':event['utc'],'retry_index':event.get('retry_index'),
                         'outcome':outcomes[-1]['event'] if outcomes else 'UNRESOLVED',
                         'http_status':response.get('http_status'),
                         'input_tokens':provider_usage.get('prompt_tokens'),
                         'output_tokens':provider_usage.get('completion_tokens'),
                         'provider_usage':provider_usage,'latency_seconds':response.get('latency_seconds'),
                         'provider_request_ids':response.get('headers',{}),
                         'returned_model':raw.get('model'),
                         'finish_reason':(raw.get('choices')or[{}])[0].get('finish_reason')})
    return manifest,assignment,status,usage,completed,incomplete,attempts


def mechanism_rows(completed):
    rows=[]
    for record in completed:
        for name,cell in record['cells'].items():
            placement,arm=name.split('_',1)
            support_initial=cell['initial_agent_scores'];support_final=cell['revised_agent_scores']
            deliveries=cell['deliveries']
            delivered_to=defaultdict(set)
            for delivery in deliveries:
                delivered_to[delivery['recipient']].update((f['title'],f['sentence_id']) for f in delivery['facts'])
            citation_uptake=[]
            for node,response in enumerate(cell['responses']):
                citation_uptake.append(len(set(map(tuple,response['citations'])) & delivered_to[node]))
            rows.append({'question_id':record['question_id'],'placement':placement,'arm':arm,
                         'initial_majority_f1':record['diagnostics'][placement]['initial_majority_score']['f1'],
                         'initial_majority_em':record['diagnostics'][placement]['initial_majority_score']['em'],
                         'final_f1':cell['f1'],'final_em':cell['em'],
                         'any_initial_exact':cell['any_initial_exact'],'any_revised_exact':cell['any_revised_exact'],
                         'correct_to_wrong_nodes':cell['correct_to_wrong_nodes'],
                         'wrong_to_correct_nodes':cell['wrong_to_correct_nodes'],
                         'f1_improved_nodes':[i for i,(a,b) in enumerate(zip(support_initial,support_final)) if b['f1']>a['f1']+1e-9],
                         'f1_harmed_nodes':[i for i,(a,b) in enumerate(zip(support_initial,support_final)) if b['f1']<a['f1']-1e-9],
                         'answer_changed_nodes':[i for i,(a,b) in enumerate(zip(record['initial'][placement],cell['responses'])) if normalize(a['answer'])!=normalize(b['answer'])],
                         'aggregator_rejected_correct':cell['aggregator_rejected_correct'],
                         'directed_messages':len(deliveries),
                         'delivered_source_fact_instances':sum(len(d['facts']) for d in deliveries),
                         'new_annotated_support_delivered_by_node':cell['new_annotated_support_delivered_by_node'],
                         'all_support_accessible_by_node':cell['all_support_accessible_by_node'],
                         'all_support_accessible_but_wrong_nodes':[i for i,(accessible,s) in enumerate(zip(cell['all_support_accessible_by_node'],support_final)) if accessible and s['em']!=100],
                         'delivered_fact_citation_overlap_by_node':citation_uptake,
                         'invalid_citation_count':sum(len(r['invalid_citations']) for r in cell['responses']),
                         'provider_abstention_count':sum(r.get('provider_refusal',False) for r in cell['responses'])})
    return rows


def proposal_utilization(records,labels):
    """Gold-selected best scores are diagnostic references, never predictions."""
    rows=[]
    for record in records:
        gold=labels[record['question_id']]['answer']
        for placement in ['colocated','distributed']:
            initial=record['initial'][placement]
            initial_scores=[score(r['answer'],gold) for r in initial]
            best=max(s['f1'] for s in initial_scores)
            any_exact=any(s['em']==100 for s in initial_scores)
            comm=record['cells'][placement+'_comm']
            rows.append({'question_id':record['question_id'],'placement':placement,
                'initial_any_exact':any_exact,'broadcast_any_exact':comm['any_revised_exact'],
                'broadcast_selected_initial_proposal':normalize(comm['answer']) in {normalize(r['answer']) for r in initial},
                'broadcast_equals_initial_oracle_best_f1':abs(comm['f1']-best)<1e-9,
                'broadcast_introduced_exact_without_initial_exact':comm['any_revised_exact'] and not any_exact,
                'initial_oracle_best_f1':best,'broadcast_f1':comm['f1']})
    return rows


def figures(stats,power,records,tables):
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/projectrl-causal-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'analysis/figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    if records:
        fig,ax=plt.subplots(figsize=(8,4))
        names=['colocated_nocomm','colocated_comm','distributed_nocomm','distributed_comm']
        data=[[r['cells'][name]['f1'] for name in names] for r in records]
        for values in data:ax.plot(range(4),values,color='.75',linewidth=1,marker='o',markersize=3)
        ax.plot(range(4),np.mean(data,axis=0),color='#1b5c7a',marker='o',linewidth=2,label='Question mean')
        ax.set_xticks(range(4),['Co-located\nprivate revision','Co-located\nbroadcast','Distributed\nprivate revision','Distributed\nbroadcast'])
        ax.set(ylabel='Answer F1 (points)',xlabel='Placement and revision arm',ylim=(-3,103),title=f'E6 crossed pilot: {len(records)} unique questions')
        ax.legend();fig.tight_layout();fig.savefig(out/'e6_cells.png');fig.savefig(out/'e6_cells.svg');plt.close(fig)
        per=stats['per_question'];fig,ax=plt.subplots(figsize=(8,4))
        ax.bar(range(len(per)),[r['interaction']*100 for r in per],color='#1b5c7a')
        ax.axhline(0,color='.3',linewidth=.8)
        ax.set_xticks(range(len(per)),[r['question_id'][-6:] for r in per],rotation=45)
        ax.set(xlabel='Question ID suffix',ylabel='Paired interaction (F1 points)',title='E6 placement × communication: question-level effects')
        fig.tight_layout();fig.savefig(out/'e6_question_interactions.png');plt.close(fig)
    if power.get('grid'):
        fig,ax=plt.subplots(figsize=(7,4))
        for effect,style in [(.02,':'),(.05,'-'),(.1,'--')]:
            rows=[r for r in power['grid'] if r['variance_multiplier']==1 and r['effect']==effect]
            ax.plot([r['n_questions'] for r in rows],[r['estimated_power'] for r in rows],style,color='#1b5c7a',label=f'{effect*100:g} F1-point interaction')
        ax.axhline(.8,color='.5',linewidth=.8);ax.axhline(.9,color='.7',linewidth=.8)
        ax.set(xlabel='Fresh unique questions',ylabel='Simulated power (fraction)',title='Planning sensitivity, baseline SD scenario; approximate test',ylim=(0,1.03),xscale='log')
        ax.legend();fig.tight_layout();fig.savefig(out/'power_sensitivity.png');plt.close(fig)
    prior_path=ROOT/'offline/e8_summary.csv'
    if prior_path.exists():
        prior=list(csv.DictReader(prior_path.open()))
        selected=[next(r for r in prior if r['experiment']=='E5' and r['condition']==name)
                  for name in ['P_NoComm','P_Broadcast']]
        fig,ax=plt.subplots(figsize=(6,4));x=np.arange(2)
        ax.bar(x-.18,[int(r['final_any_em_count']) for r in selected],width=.36,color='.7',label='Any exact agent (oracle diagnostic)')
        ax.bar(x+.18,[int(round(float(r['final_em'])*3)) for r in selected],width=.36,color='#1b5c7a',label='Practical majority exact')
        ax.set_xticks(x,['NoComm','Broadcast'])
        ax.set(xlabel='Saved E5 partitioned condition',ylabel='Questions (out of 300)',title='E8: proposal availability and majority selection differ')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'e8_proposal_selection.png');plt.close(fig)


def format_points(value):return 'INSUFFICIENT_DATA' if value is None else f'{100*value:.3f}'


def format_ci(value):return 'INSUFFICIENT_DATA' if value is None else '['+', '.join(format_points(x) for x in value)+']'


def write_reports(manifest,assignment,status,usage,records,incomplete,attempts,stats,power,mechanisms,proposal_rows):
    report=ROOT/'reports';n=len(records)
    successes=usage['successful_completions'];latencies=[a['latency_seconds'] for a in attempts if a.get('outcome')=='success' and a['latency_seconds'] is not None]
    mean_latency=float(np.mean(latencies)) if latencies else None
    mean_input=usage['input_tokens']/successes if successes and usage['input_tokens'] is not None else None
    mean_output=usage['output_tokens']/successes if successes and usage['output_tokens'] is not None else None
    names={'colocated_nocomm_f1':'Co-located private revision','colocated_comm_f1':'Co-located broadcast',
           'distributed_nocomm_f1':'Distributed private revision','distributed_comm_f1':'Distributed broadcast'}
    lines=['# Pilot results','',f"Status: **{status['status']}**. Complete unique factorial questions: **{n}/8**. Incomplete question IDs: {incomplete or 'none'}. E7: **NOT_RUN_PAID_INFERENCE_DEFERRED**. No missing or uncertain call is scored as a wrong prediction.",'',
           '## Executed resources','',f"Successful provider completions: {successes}; HTTP attempts: {usage['http_attempts']}; retries: {usage['retries']}; explicit rejections: {usage['rejected_attempts']}; uncertain requests: {usage['uncertain_requests']}. Four E6 cells per complete question; both revision arms receive three updates per placement. The 18 physical calls/question share initial states only within placement.",'',
           f"Provider input tokens: {usage['input_tokens']}; output tokens: {usage['output_tokens']}; missing input/output usage rows: {usage['missing_input_usage']}/{usage['missing_output_usage']}. Known lower bounds: {usage['input_tokens_lower_bound']}/{usage['output_tokens_lower_bound']}. Actual monetary cost: **UNVERIFIED_DEPLOYMENT_PRICE**, not zero. Remaining authorized budget: {usage['remaining_completions']} completions / {usage['remaining_http_attempts']} attempts. E7 is not run to exhaust quota.",'',
           f"Runtime for the recorded pilot session: {status.get('runtime_this_session_seconds','NOT_RUN')} seconds. Successful HTTP latency sum: {usage['successful_http_latency_seconds']:.3f} seconds; mean: {mean_latency}. Full request usage, hashes and provider IDs: `../analysis/tables/request_usage.csv`. Provider reasoning is included only as exposed by provider usage. Returned model aliases: {sorted({a['returned_model'] for a in attempts if a.get('returned_model')})}.",'',
           '## Manipulation and paired effects','',f"Frozen assignment compliance: {assignment['annotated_support_placement_compliance']*100:g}% for equal six-document unions, source identity and intended support ownership. Two documents per agent, actual local evidence {assignment['local_evidence_tokens_min_max']} exact tokens. Full realized role/position counts: `../dry_run/assignment_validation.json`. Receiver-update counts and within-placement local context are held fixed; equal input-token cost is not claimed.",'']
    for name,label in names.items():
        cellname=name.removesuffix('_f1')
        em=float(np.mean([r['cells'][cellname]['em'] for r in records])) if records else None
        lines.append(f"- {label}: F1 **{format_points(stats['cell_means'].get(name))}** points; EM {em}%.")
    lines.append('')
    for name,label in [('colocated_comm_effect','Co-located communication effect'),('distributed_comm_effect','Distributed communication effect'),('interaction','Primary difference-in-differences')]:
        c=stats['contrasts'][name]
        lines.append(f"- {label}: **{format_points(c['mean'])}** F1 points; exploratory question-bootstrap95% CI {format_ci(c['ci95'])}; helped/harmed/unchanged {c['help_harm_unchanged']}. Two-sided sign-flip reference p={c['sign_flip_test']['p_value']}.")
    interaction=stats['contrasts']['interaction']['mean']
    direction='INSUFFICIENT_DATA' if interaction is None else ('near zero' if abs(interaction)<.01 else ('consistent with the working hypothesis' if interaction>0 else 'opposite to the working hypothesis'))
    lines.extend(['',f"Observed direction is **{direction}**. With n={n}, it remains too unstable to establish a population interaction. These p-values are exploratory symmetry-based references, not physical treatment-randomization p-values, and secondary tests are not multiplicity-adjusted. A zero/insignificant result is not equivalence or noninferiority. All question effects, including harms and ties, are retained in `../analysis/tables/e6_question_effects.csv`.",'',
                  '## Power and uncertainty','',f"Observed paired-interaction SD: {power.get('observed_interaction_sd')}; baseline planning SD: {power.get('base_sd')} in unit F1. The planning floor is0.25 (25 F1 points); sensitivity multiplies SD by1,1.5,2. Assumed effects are2,5,10 F1 points, never the observed pilot effect. Each grid cell uses{power.get('simulations_per_grid_point')} simulated studies. Recommendations use the lower Monte Carlo95% bound, a deliberately conservative coarse grid.",''])
    for r in power.get('recommendations',[]):
        if r['variance_multiplier']==1:
            lines.append(f"- Effect{100*r['effect']:g} F1 points, {100*r['target_power']:g}% power: simulated grid n={r['smallest_simulated_n']}, normal-approximation n={r['normal_approximation_n']}; {r['status']}.")
    lines.extend(['','All sensitivity scenarios and API-call projections are in `../analysis/tables/power_grid.csv` and `power_recommendations.csv`. The simulation uses a conditional-normal approximation to the paired sign-flip statistic. Empirical residual shape and variance are uncertain at n8; scaled residuals can leave the feasible interaction range. Inspect the reported violation fractions. Before confirmatory spending, validate the selected sample size with the exact planned analysis and a scientifically justified minimum important effect. No full study is launched.',
                  '',f"The current verified source has {power['current_source_remaining_eligible_questions']} remaining eligible unique questions after excluding this pilot and known exposures. A2,000-question scenario exceeds this pool and is not executable from the current source. A new complementarity screen can shrink the pool further; never duplicate questions to meet a target. Scenarios beyond the pool require a separately justified verified source/population and fresh power analysis.",'',
                  '## Data provenance','',f"Manifest SHA256: `{manifest['manifest_sha256']}`. Raw requests/attempts/responses, atomic question checkpoints and immutable inference settings are in `../paid_pilot/` and `../frozen/`. The mock ledger is separate and never included in these metrics. Reports are reproducible without inference using `python3 -m WEBExtension.causal_study.analyze`."])
    (report/'04_PILOT_RESULTS.md').write_text('\n'.join(lines)+'\n')
    msg_count=sum(r['directed_messages'] for r in mechanisms)
    facts=sum(r['delivered_source_fact_instances'] for r in mechanisms)
    newfacts=sum(sum(r['new_annotated_support_delivered_by_node']) for r in mechanisms)
    harms=sum(len(r['correct_to_wrong_nodes']) for r in mechanisms)
    recoveries=sum(len(r['wrong_to_correct_nodes']) for r in mechanisms)
    rejected=sum(r['aggregator_rejected_correct'] for r in mechanisms)
    wrong_with_support=sum(len(r['all_support_accessible_but_wrong_nodes']) for r in mechanisms)
    invalid=sum(r['invalid_citation_count'] for r in mechanisms)
    mechanism=['# Mechanism and failure analysis','',f"E6 completed-block diagnostics: {n} unique questions, {len(mechanisms)} question-cells. Across cells: {msg_count} directed broadcast messages, {facts} delivered source-fact instances, {newfacts} newly accessible annotated-support fact instances, {harms} EM-exact-to-nonexact agent transitions, {recoveries} nonexact-to-exact transitions, {rejected} cells rejecting an exact candidate, {wrong_with_support} EM-nonexact revised agent outputs despite all annotated support being accessible, and {invalid} invalid citation references. EM misses are metric-defined, not a claim that every response is substantively wrong. Agent/cell sums repeat questions; they are descriptive events, not independent sample sizes.",'',
        'Source packets are the frozen initial answer plus complete sentences matching valid sender-visible citations. Messages can carry an answer proposal with no source fact. The source-count manipulation and factual uptake must be reported separately. A citation to a delivered fact does not establish entailment or internal use; unsupported/hallucinated reasoning and contradictions are **UNOBSERVABLE_FROM_CONCISE_ANSWER_CITATION_OUTPUT** except for directly invalid citation indices. No paid judge was invoked. Source delivery is an accessibility measurement, not successful knowledge transfer.','',
        '## Proposal availability and what the intervention did','',
        f"Across{len(proposal_rows)} placement/question blocks, {sum(r['broadcast_selected_initial_proposal'] for r in proposal_rows)} broadcast final answers already occurred among the frozen initial proposals and {sum(r['broadcast_equals_initial_oracle_best_f1'] for r in proposal_rows)} equal the best initial agent F1. Broadcast generated a new exact proposal on{sum(r['broadcast_introduced_exact_without_initial_exact'] for r in proposal_rows)} blocks lacking any initial exact proposal. Best-agent F1 uses gold and is an oracle diagnostic only; it was never used to select answers or sample questions. This pattern is consistent with propagating/selecting existing answers, rather than evidence that agents needed to combine complementary missing facts. The delivered source sentences and answer proposals remain bundled, so their causal contributions are not separated.",'',
        'The source-only independent review in report06 finds that many single annotated support articles already contain the answer fact or share the answer attribute; annotation membership does not establish complementary necessity. Several distractors also mention the answer. The pilot was selected by frozen hashes and not by ease, but the eligible population still does not enforce the intended necessity intervention. A next source review must be blind to outputs and must not quietly replace this cohort based on outcomes. Treat this as a scientific design finding, not BM25 failure.','',
        '## Surface forms, abstentions, and the two nonzero interactions','',
        'Six of eight question interactions are exactly zero. The negative interaction is the Quinlan question: co-located private majority chooses noanswer while distributed private revision has a three-way tie selecting `Ernesto "Che" Guevara` (80F1); both broadcasts select `Che Guevara` (100F1). The positive interaction is the RosiGolan/Chicco question: broadcasts select `more than120` versus `more than120 countries`, scoring85.714 versus100 despite the same numerical answer. These observed response/tie/scoring paths make the small pilot direction fragile; they do not establish a causal explanation of population behavior.','',
        'For DannyJones/AleksiSihvonen, broadcast propagates `vocalists` to every agent, while the reference is `vocalist`; official EM and answerF1 are0 because the evaluator does not stem singular/plural. This is an observed surface-form limitation, not proof of failed factual reasoning. Preserve the prespecified official metric; no alternative post-hoc semantic score, revised gold answer, or paid regeneration is substituted. `Bi-fuel` and `Swiss Super League` also receive partial F1 against longer reference strings.','',
        '## Question-level trace register','']
    qlookup={q['id']:q for q in manifest['questions']}
    for record in records:
        q=qlookup[record['question_id']];label=manifest['private']['labels'][q['id']]
        mechanism.extend([f"### {q['id']}",'',q['question'],f"Reference answer: `{label['answer']}`. Support articles: {label['support_titles']}.",''])
        for placement in ['colocated','distributed']:
            initial=record['initial'][placement]
            mechanism.append(f"- {placement}: initial answers {[r['answer'] for r in initial]}; initial majority `{majority_answer(initial)}`.")
            for arm in ['nocomm','comm']:
                cell=record['cells'][placement+'_'+arm]
                mechanism.append(f"- {placement}/{arm}: revised answers {[r['answer'] for r in cell['responses']]}; selected `{cell['answer']}`, F1={cell['f1']:.3f}, EM={cell['em']:.0f}; new support by node={cell['new_annotated_support_delivered_by_node']}; majority rejected exact={cell['aggregator_rejected_correct']}.")
        mechanism.append('')
    mechanism.extend(['## E8 old-study comparison','',
        'E8 is wholly offline and observational, not an E6 intervention estimate. In saved E5 P, initial any-agent exact67/300 contrasts with majority10/300; broadcast any-agent exact66/300 contrasts with majority34/300. Of25 questions whose initially incorrect vote becomes exact,23 already had an exact initial proposal. This supports a proposal-utilization bottleneck as an observed failure mode; it does not identify whether content, repeated computation, graph reach or agreement formation caused the old gains. Learned seed19/31 reject48/49 exact revised candidates versus32 for broadcast. No oracle selection is a practical model result. Full43-group transitions, EM/F1 separation and intervals are in `../offline/`.',
        '', 'Earlier E1 partitioned gains and its imprecise interaction do not predict the same-union intervention. E6 conditions on support availability; older300 include224 cases with at least one annotated article missing, mostly absent before ranking. No estimates are pooled across these protocols. Corrected E3-v2 tests redundant shared-evidence presentation and does not establish novel transfer. E4 is replay stopping. Learned routing is unnecessary for the current fixed-communication question.',
        '', 'E7 paid outcomes are **NOT_RUN**. Six provisional exact-token-matched source triples were generated locally; no semantic non-support or treatment effect is asserted. A future E7 needs blind content verification, whole-passage treatment validation, and adequate frozen receiver states; no outcome-based receiver selection.'])
    (report/'05_MECHANISM_AND_FAILURE_ANALYSIS.md').write_text('\n'.join(mechanism)+'\n')
    complete=status['status']=='COMPLETE' and n==8
    # One overall decision: operational validity alone does not resolve the
    # central scientific mechanism or its direct prior-work overlap.
    decision='REVISE'
    effect5=[r for r in power.get('recommendations',[]) if r['effect']==.05 and r['variance_multiplier']==1]
    recommendations=[]
    for r in effect5:
        nq=r['smallest_simulated_n'] or r['normal_approximation_n'];calls=18*nq
        recommendations.append({'target_power':r['target_power'],'n_questions':nq,'calls':calls,
                 'input_tokens':math.ceil(calls*mean_input) if mean_input is not None else None,
                 'output_tokens':math.ceil(calls*mean_output) if mean_output is not None else None,
                 'serial_runtime_seconds':calls*max(mean_latency or 0,2.),'monetary_cost':None})
    atomic_json(ROOT/'analysis/next_study_resource_projection.json',recommendations)
    final=['# Final scientific decision','',f"**{decision}.** The E6 pipeline can be operationally valid, while the intended novel-evidence mechanism and publication distinction still need correction before a larger study. This decision does not depend on a favorable pilot effect. E7 requires semantic control review before paid use.",'',
        '## 1. What was executed','',f"E8:12,898 saved trace records/43 groups offline. E6:{n} complete unique questions, four primary cells, {successes} successful completions, {usage['http_attempts']} attempts, {usage['retries']} retries, {usage['rejected_attempts']} rejected attempts, {usage['uncertain_requests']} uncertain. Input/output tokens {usage['input_tokens']}/{usage['output_tokens']}; monetary cost unverifiable; session runtime {status.get('runtime_this_session_seconds')}s. Remaining approved quota {usage['remaining_completions']} completions/{usage['remaining_http_attempts']} attempts. E7 paid NOT_RUN. No full study, PPO, push or publication.",'',
        '## 2. What the pilot showed','',f"Co-located NoComm/Comm F1 {format_points(stats['cell_means']['colocated_nocomm_f1'])}/{format_points(stats['cell_means']['colocated_comm_f1'])}; distributed NoComm/Comm {format_points(stats['cell_means']['distributed_nocomm_f1'])}/{format_points(stats['cell_means']['distributed_comm_f1'])}. Primary interaction {format_points(interaction)} F1 points, exploratory95% CI {format_ci(stats['contrasts']['interaction']['ci95'])}; direction {direction}, too unstable for a conclusive hypothesis judgment.",'',
        '## 3. Evidence ranking','',
        '- Supported by observations: the implemented source-allocation intervention is mechanically compliant; old E8 traces expose correct-proposal rejection and revision loss. This establishes observable bottlenecks, not their causal remedy.',
        '- Supported by pilot observations: all16 broadcast finals already appeared among initial proposals and equal the initial best-agent F1 diagnostic. No new exact-proposal coverage is observed. This is proposal utilization, not a gold-based practical oracle selector or a causal attribution to answer copying.',
        '- Suggestive but inconclusive: the pilot point interaction opposes the working hypothesis, but six ties and two response/tie/surface-form-sensitive nonzero interactions cannot establish the population direction or practical effect size.',
        '- Unsupported: the claim that separating annotated articles necessarily creates complementary missing information. Source review found single-article sufficiency and answer-bearing distractors. Successful semantic transfer from mere packet delivery, natural Web generalization, model-general superiority, and broad novelty also remain unsupported.',
        '- Unsupported by prior policy comparisons: inherent learned-routing superiority or an optimal sparse topology. Those comparisons also vary recipient calls and transmitted material.',
        '', '## 4. Strongest honest narrative','',
        '**When Evidence Is Available but Answers Are Lost: Controlled Evidence Placement and Proposal Utilization in Multi-Agent QA.** The scientific question is where this fixed pipeline loses available evidence or correct proposals, and how redistribution changes peer-information value with revision computation controlled. The old E8 descriptive bottleneck is stronger evidence than any conclusive pilot interaction. Contemporary primary sources already study information asymmetry and proposal/readout separation; this title is a working framing, not a novelty claim. See report06 and literature_notes.md. No manuscript claim is rewritten ahead of evidence.',
        '', '## 5. Smallest next study and resources','',
        'First resolve the confirmatory target: the current intervention identifies the whole cited-source/answer broadcast pipeline, with article-length allocation and answer aggregation included. Freeze a scientifically important5-F1-point interaction target only if justified independently of this pilot. Select fresh eligible questions using the same exclusions plus these8 pilot IDs; retain same model/temperature/aggregation, one initial and one private/broadcast revision per node and placement. At18 calls/question, prospective sample/cost scenarios based on observed physical usage are:', '']
    for r in recommendations:
        final.append(f"- {100*r['target_power']:g}% power baseline-SD scenario: {r['n_questions']} fresh questions, {r['calls']} additional successful completions, approximately {r['input_tokens']} input/{r['output_tokens']} output tokens, serial runtime {r['serial_runtime_seconds']/3600:.2f}hours. Price remains unknown; do not turn token projections into dollars without the deployment tariff.")
    final.extend(['',
        f"These are sensitivity scenarios for the unchanged pipeline and eligible population, not an approved experiment or a definitive sample-size prescription. The current source has {power['current_source_remaining_eligible_questions']} eligible unique questions left;2,000-question stress scenarios are beyond that pool. A new complementarity-qualified population can have a different variance and fewer questions, so the same power estimates do not automatically apply. SD inflation scenarios can require far more questions; validate the approximate planning test at the candidate sample size before approval. Exact counts assume one placement replicate per question. A source-position/role-robust claim requires additional counterbalanced assignments clustered by question and multiplies calls accordingly; do not pretend those assignments add independent questions. Check residual feasibility and source lengths before committing resources. Runtime extrapolates observed mean latency with the current two-second serial floor, ignores future429s, and may drift.",
        '', 'The smallest zero-call correction is a fresh-cohort reservation, analysis freeze, blind necessity/complementarity/source-length review, and a precise novelty comparison. A future E7 source-content experiment must review relevant/redundant/distractor packets and control remaining lengths/semantics before estimating a powered effect. Do not replace valid existing E3-v2 or repeat old runs. Additional API cost is justified only if these controls separate a meaningful unresolved mechanism.',
        '', '## 6. Decision and objections remaining','',
        f"{decision}: "+('the eight-question E6 implementation and resource/resume checks work, but a whole-pipeline effect does not resolve the intended evidence-transfer mechanism; complete the smallest source-control and novelty corrections before confirming the central narrative.' if complete else 'the pilot is incomplete; preserve all saved outputs and correct/reconcile the blocking issue before further requests.'),
        '', 'A broad claim that novel evidence transfer explains multi-agent gains remains **REVISE**: E6 transmits proposals plus selected citations, E7 was not run, supporting annotations do not prove necessity, and aggregation can explain part of observed improvement. Controlled candidates are not natural retrieval; one deployment is not model generality; prior questions informed design; current primary literature threatens novelty. The eight-question pilot is not publication-ready evidence. Report06 gives severity, evidence, corrections and API needs for each objection. No further inference is started by this analysis.'])
    (report/'07_FINAL_RECOMMENDATION.md').write_text('\n'.join(final)+'\n')
    return recommendations


def majority_answer(responses):
    from .design import majority
    return majority(responses)


def main():
    manifest,assignment,status,usage,records,incomplete,attempts=collect()
    output=ROOT/'analysis';output.mkdir(exist_ok=True)
    rows=[record['statistics_row'] for record in records]
    stats=analyze_e6(rows);power=power_sensitivity(rows)
    remaining_pool=manifest['selection']['eligible_questions']-len(manifest['questions'])
    power['current_source_remaining_eligible_questions']=remaining_pool
    for row in power['grid']:
        row['within_current_remaining_pool']=row['n_questions']<=remaining_pool
    for row in power['recommendations']:
        planned=row['smallest_simulated_n']
        row['within_current_remaining_pool']=planned<=remaining_pool if planned is not None else False
    mechanisms=mechanism_rows(records)
    proposal_rows=proposal_utilization(records,manifest['private']['labels'])
    tables=output/'tables';write_csv(tables/'e6_question_scores.csv',rows)
    write_csv(tables/'e6_question_effects.csv',stats['per_question'])
    write_csv(tables/'request_usage.csv',attempts)
    resources=[]
    for placement in ['colocated','distributed']:
        for stage in ['initial','nocomm','broadcast']:
            subset=[a for a in attempts if a.get('placement')==placement and a.get('stage')==stage]
            success=[a for a in subset if a['outcome']=='success']
            resources.append({'placement':placement,'stage':stage,'http_attempts':len(subset),
                'successful_completions':len(success),
                'input_tokens':sum(a['input_tokens'] for a in success) if all(a['input_tokens'] is not None for a in success) else None,
                'output_tokens':sum(a['output_tokens'] for a in success) if all(a['output_tokens'] is not None for a in success) else None,
                'http_latency_seconds':sum(a['latency_seconds'] or 0 for a in subset),
                'note':'initial cost shared between two revision arms' if stage=='initial' else 'revision only; identical initial state within placement'})
    write_csv(tables/'resource_by_stage.csv',resources)
    write_csv(tables/'mechanism_events.csv',mechanisms)
    write_csv(tables/'proposal_utilization.csv',proposal_rows)
    write_csv(tables/'power_grid.csv',power['grid']);write_csv(tables/'power_recommendations.csv',power['recommendations'])
    cellrows=[]
    for name in ['colocated_nocomm','colocated_comm','distributed_nocomm','distributed_comm']:
        cellrows.append({'cell':name,'n_questions':len(records),
                         'f1_mean':float(np.mean([r['cells'][name]['f1'] for r in records])) if records else None,
                         'em_mean':float(np.mean([r['cells'][name]['em'] for r in records])) if records else None})
    write_csv(tables/'cell_summary.csv',cellrows)
    atomic_json(output/'e6_statistics.json',stats);atomic_json(output/'power_sensitivity.json',power)
    atomic_json(output/'usage.json',usage)
    figures(stats,power,records,tables)
    resources=write_reports(manifest,assignment,status,usage,records,incomplete,attempts,stats,power,mechanisms,proposal_rows)
    print(json.dumps({'status':status['status'],'n_questions':len(records),'usage':usage,
                     'cell_means':stats['cell_means'],'interaction':stats['contrasts']['interaction'],
                     'power_observed_sd':power['observed_interaction_sd'],'resource_projection':resources},indent=2))


if __name__=='__main__':main()
