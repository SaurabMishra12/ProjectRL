"""Generate matched E5 reports01-04 and scientific figures from offline results."""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path
from .analysis import ROOT, RULES, ARMS, METHODS, save, table


def read(name): return json.loads((ROOT / 'results' / name).read_text())
def rows(name): return [json.loads(line) for line in (ROOT / 'results' / name).read_text().splitlines()]
def link(name, label=None): return '[' + (label or name) + '](' + str(ROOT / name) + ')'
def number(n): return f'{n:.3f}' if isinstance(n, float) else str(n)
def interval(effect): return f"{effect['delta_pp']:+.3f} [{effect['ci95_pp'][0]:+.3f}, {effect['ci95_pp'][1]:+.3f}]"
def mdtable(headers, values):
    return '\n\n| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + ''.join('| ' + ' | '.join(map(str, row)) + ' |\n' for row in values) + '\n'


def figures(summary):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/projectrl-readout-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    labels = ['Original', 'Abstention-aware', 'Aware + lexical tie']
    plotted = []
    for ax, regime, title in zip(axes, ['P', 'R'], ['Partitioned evidence', 'Shared evidence']):
        x = np.arange(3)
        for offset, method, color in [(-.18, 'NoComm', '#8a9298'), (.18, 'Broadcast', '#326a89')]:
            values = [next(s['correct'] for s in summary if s['arm'] == regime + '_' + method and s['readout'] == rule) for rule in RULES]
            bars = ax.bar(x + offset, values, width=.34, color=color, label=method)
            ax.bar_label(bars, padding=3, fontsize=10)
            plotted.append({'regime': regime, 'method': method, 'readout_order': list(RULES), 'correct_counts': values})
        ax.set_xticks(x, labels, rotation=13, ha='right')
        ax.set_ylim(0, 90); ax.set_title(title); ax.set_xlabel('Identical readout applied to both conditions')
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, alpha=.15); ax.set_axisbelow(True)
    axes[0].set_ylabel('Exactly correct final answers (of 300 questions)')
    axes[0].legend(frameon=False, loc='upper left')
    fig.suptitle('Matched readouts reduce the apparent communication gain', x=.08, ha='left', fontsize=15)
    fig.text(.08, .025, 'Source: saved E5 outputs; exploratory offline reanalysis. The two regimes share the same 300 question identities.', fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.83, wspace=.1)
    directory = ROOT / 'figures'; directory.mkdir(exist_ok=True)
    fig.savefig(directory / 'matched_broadcast_no_comm.png', dpi=180)
    fig.savefig(directory / 'matched_broadcast_no_comm.svg'); plt.close(fig)
    save(directory / 'plotted_observations.json', plotted)


def main():
    summary = read('matched_summary.json'); costs = read('physical_accounting.json')
    effects = read('communication_effects.json'); interactions = read('interaction_effects.json')
    changes = rows('readout_answer_changes.jsonl'); transitions = rows('communication_transitions.jsonl')
    raw = rows('question_readouts.jsonl'); refs = read('response_and_reference_audit.json')
    code = read('original_code_comparison.json'); fresh = read('fresh_pool_verification.json')
    by = {(s['arm'], s['readout']): s for s in summary}; cost_by = {c['arm']: c for c in costs['arms']}
    joined = [{**s, 'agent_abstention_percent': 100 * s['agent_abstentions'] / (3 * s['n_questions']),
               **{k: v for k, v in cost_by[s['arm']].items() if k != 'arm'}} for s in summary]
    table(ROOT / 'results/matched_results_with_costs.csv', joined)
    figures(summary)

    text = ['# Independent baseline verification', '', '**Zero new inference. All original E5 endpoints are independently reproduced.** The new implementation reads raw E5 JSONL and original provider caches, not the E8 reconstructed outcome table. It imports neither the original vote nor the old analysis. Previous metadata is used to locate and hash-check source files; previous claimed counts are not prediction inputs.', '',
        '## Independent reproduction', '',
        'Every arm contains the same300 unique questions and three ordered final responses. All6,000 original answers, EM values and F1 values agree with an independent normalized-plurality implementation. All18,000 proposal references and18,000 readout scores are cross-checked against the three inspected pure functions extracted from the preserved official Hotpot evaluator. The original frozen source snapshot and current normalization, scoring, majority, role assignment, condition and routed-cost functions have identical ASTs. See ' + link('results/original_code_comparison.json') + ' and ' + link('results/verification.json') + '.']
    text.append(mdtable(['Regime/arm', 'Original correct', 'Aware correct', 'Aware lexical correct', 'Any-correct coverage'],
        [[a, by[(a, 'original')]['correct'], by[(a, 'abstention_aware')]['correct'], by[(a, 'abstention_lexical')]['correct'], by[(a, 'original')]['any_correct_proposals']] for a in ['P_NoComm', 'P_Broadcast', 'R_NoComm', 'R_Broadcast']]))
    text += ['The requested10,34,62,67 and66 counts are reproduced. Applying the same repaired rule to Broadcast gives65, which is the essential additional count. The earlier62 versus34 comparison changes both communication and readout and cannot show communication harm.', '',
        '## What the original aggregator actually does', '',
        'It lowercases, removes ASCII punctuation and English articles, pools identical normalized answers, chooses the largest count, and returns the first raw answer in the ordered response list with that count. Despite its name, this is plurality: it does not require more than half the agents to agree. In a three-way tie, node0 wins. There is no fresh tie lottery or answer-quality check. The code randomizes role assignment by question, and evidence partitioning separately; the final vote itself does not shuffle response positions. Cached request node metadata was checked against all18,000 final response references.', '',
        'Abstention-aware voting removes normalized `noanswer` and explicitly marked provider-refusal empty responses, then applies exactly the same plurality and original node-order tie. All-abstaining cases return `noanswer`. Whitespace-only or empty unmarked answers fail validation; nonempty answers with a refusal flag also fail. Synonyms such as unknown are not retrospectively recoded. Literal A, yes and no remain substantive. The lexical sensitivity baseline keeps the same filter/counts but chooses the lexicographically smallest normalized answer in a tied maximum. Its normalized winner is invariant to node permutation, though the first raw alias is used for rendering. This arbitrary but reproducible tie preference is a sensitivity analysis, not a fitted or optimal rule.', '',
        '## Why exclusion changes the result', '']
    detail = []
    for arm in ['P_NoComm', 'P_Broadcast', 'R_NoComm', 'R_Broadcast']:
        c = [r for r in changes if r['arm'] == arm and r['readout'] == 'abstention_aware']
        substantive = [r for r in c if r['abstention_overrode_substantive_answer']]
        detail.append([arm, len(c), len(substantive), sum(r['abstention_won_by_plurality'] for r in substantive), sum(r['abstention_won_by_node_tie'] for r in substantive), len(c) - len(substantive), sum(r['em_delta'] > 0 for r in c), sum(r['em_delta'] < 0 for r in c)])
    text.append(mdtable(['Arm', 'Changed normalized strings', 'Abstention overrides', 'Plurality overrides', 'Node-tie overrides', 'Only abstention marker changes', 'EM gains', 'EM losses'], detail))
    text += ['Partitioned NoComm changes114 endpoints because an abstention defeated an available substantive answer:109 plurality overrides and5 node-order ties. Only52 new selections are exactly correct; excluding abstentions does not turn every substantive candidate into truth. Partitioned Broadcast has74 such overrides plus2 all-abstaining marker standardizations. Those latter changes convert an empty refusal winner to `noanswer`; both remain incorrect, not new substantive proposals.', '',
        'If every abstention scores0, deleting only abstentions while preserving the same substantive plurality/tie rule cannot lower EM: an already substantive winner stays unchanged, and only a zero-score abstention can be replaced. This is a structural property under the stated scoring contract, not a new general theorem about reliable refusal. It explains why a large readout gain alone is weak novelty evidence.', '',
        '## Empty responses, refusals and reference validity', '',
        f"Among{refs['provider']['unique_provider_responses']:,} distinct saved request keys, there are{refs['provider']['explicit_provider_refusals']} explicit refusals, exactly{refs['provider']['raw_empty_answers']} empty answers,{refs['provider']['normalized_noanswer']:,} normalized noanswer responses and0 whitespace-only nonempty answers. They are retained with their provider markers and raw artifacts. Repeated appearances in different arms are not independent refusals.", '',
        'All300 selected references are nonempty, have nonempty official normalization and are not noanswer. Across the local7,405-question official dataset, no reference normalizes to noanswer, but two nonempty official reference strings normalize to empty: The The (`5ae4988655429970de88d9c1`) and !!! (`5a8b3d795542997f31a41cc1`). They are outside this E5 cohort. This official normalization quirk is distinct from classifying a raw answer as an abstention. The implementation does not classify a nonempty answer as abstaining merely because normalization is empty. A future evaluation must lock handling of such references before outcomes are inspected.', '',
        'The rule is reasonable for this answerable benchmark\'s answer-only accuracy and is gold-free at decision time. It is not a universally safe rejection policy. A context can be genuinely insufficient even when the benchmark has an answer; ignoring a justified local abstention may select an unsupported answer. On a task where noanswer is a legitimate correct response, the rule can lose accuracy, as explicitly tested. Report source support, answer coverage and rejection objectives alongside EM; do not silently redefine unanswerable requests.', '',
        'Graph Verify preflight: project projectRLMAIN, generation2026-10-13T17:28:45Z, fast/ready; E5 source paths and new modules are untracked. Material functions and raw sources were therefore inspected directly. Empty graph results are not used as absence/completeness evidence. This is a separate lead-agent implementation, not an external independent laboratory replication.']
    (ROOT / '01_INDEPENDENT_BASELINE_VERIFICATION.md').write_text('\n'.join(text) + '\n')

    text = ['# Complete matched aggregation results', '', '**Exploratory post hoc counterfactual readouts; zero new completions.** All20 arms×300 questions×3 readouts are present. P means partitioned evidence; R means shared evidence. Policy seeds are separate. The same readout is applied to the same saved ordered responses in each compatible arm; readout changes add no requests or tokens.', '',
        'Readouts: original normalized plurality with earliest-node ties; abstention-aware plurality with the same ties; abstention-aware plurality with normalized lexical ties. The latter is an order-sensitivity control. Every substantive selection is an existing proposal. When all agents abstain, the repaired rules return an explicit abstention marker; its EM is0 on this selected cohort. The any-correct ceiling therefore bounds these selection readouts here. It is not an upper bound on unrestricted generative synthesis.', '',
        'Selection% is correct final answers divided by questions with any correct final proposal. Coverage is an oracle diagnostic, never an input to voting. Abstention% is final-output abstention; agent-abstention counts/frequency are also in the full CSV. Changed compares normalized strings with the original readout, including explicit abstention-marker standardization.', '',
        'The complete joined table, including physical requests and token fields for every arm/readout, is ' + link('results/matched_results_with_costs.csv') + '.']
    for rule in RULES:
        text += ['','## ' + rule]
        text.append(mdtable(['Arm', 'Correct/300', 'EM%', 'Answer F1', 'Abstention%', 'Any-correct/300', 'Selection%', 'Ceiling gap pp', 'Changed'],
            [[s['arm'], s['correct'], f"{s['em']:.3f}", f"{s['f1']:.3f}", f"{s['final_abstention_percent']:.3f}", s['any_correct_proposals'], f"{s['correct_proposal_selection_rate_percent']:.3f}", f"{s['gap_to_proposal_ceiling_pp']:.3f}", s['changed_answers_vs_original']] for s in summary if s['readout'] == rule]))
    text += ['## Actual historical physical requests and tokens', '',
        'Each row below joins an arm\'s referenced request keys to original started/success events, dispatch/receipt files and hash-checked provider response caches. Counts are **nonadditive physical request references**, not exclusive incremental spending by that arm. The same initial request can support several policies and all three offline readouts. Summing rows or multiplying by three overcounts spend. Policy-training/search requests outside these referenced keys are not allocated to test-serving rows.', '',
        'Input and output columns are documented successful-provider usage. Output is a lower bound where usage is absent; unknown-execution billing can add more. Missing usage is never manufactured as complete zero. Readout computation itself adds0 model calls. Message-token volume is an explicit communication payload measure, not billed inference tokens.']
    text.append(mdtable(['Arm', 'Documented attempts', 'Successful keys', 'Input tokens', 'Output tokens LB', 'Missing output usage', 'Uncertain attempts', 'Message tokens'],
        [[c['arm'], c['documented_attempts'], c['referenced_unique_request_keys'], c['observed_input_tokens_lower_bound'], c['observed_output_tokens_lower_bound'], c['successful_attempts_missing_output_usage'], c['uncertain_execution_attempts'], c['delivered_message_tokens']] for c in costs['arms']]))
    text += ['Across the union:7,648 documented attempts,7,634 successful requests,7,786,951 known-success input tokens and at least1,992,938 output tokens;37 successful responses lack output usage and5 uncertain attempts retain unknown billing. No cached success lacks a documented dispatch. These totals are not new spend, nor a complete exact invoice. The saved token/call accounting is independent of which readout is displayed.', '',
        'No per-question ensemble is formed from learned seeds. Their final responses and historical requests often overlap, and even identical accuracy does not create independent replication. The raw row-level data, request/attempt joins and uncertainty flags are in ' + link('results/question_readouts.jsonl') + ', ' + link('results/physical_accounting.json') + ' and ' + link('results/attempt_inventory.json') + '.']
    (ROOT / '02_MATCHED_AGGREGATION_RESULTS.md').write_text('\n'.join(text) + '\n')

    text = ['# Communication effect reassessment', '',
        '**The large partitioned gain weakens sharply under matched abstention handling; it does not become a demonstrated communication harm.** Original Broadcast−NoComm EM is+8points, aware is+1point, and lexical-aware is0. Every comparison below holds the readout fixed. The same300 questions are paired, not pooled across policies, readouts or evidence regimes.', '',
        'Intervals are95% paired question bootstraps:10,000 draws, seed20261016. W/L/T counts are binary EM paired wins/losses/ties. P is the exact two-sided conditional McNemar test. All108 declared communication comparisons (including learned versus Static/Broadcast/same-seed MatchedRandom) form one Holm diagnostic family; all40 readout-versus-original comparisons form a separate family. The present tables show the54 comparisons against NoComm; all108 and every F1 result are retained in ' + link('results/communication_effects.json') + '. No interval or corrected p-value turns post hoc selection of a readout into a confirmatory superiority claim.']
    text += ['', '![Matched original and repaired readouts for NoComm and Broadcast](' + str(ROOT / 'figures/matched_broadcast_no_comm.png') + ')']
    for rule in RULES:
        text += ['','## Matched communication effects: ' + rule]
        text.append(mdtable(['Arm−same-regime NoComm', 'EM Δpp [CI]', 'W/L/T', 'Exact p', 'Holm p', 'F1 Δpp [CI]'],
            [[e['arm'], interval(e['em']), f"{e['em']['wins']}/{e['em']['losses']}/{e['em']['ties']}", f"{e['em']['exact_mcnemar_p']:.6g}", f"{e['em']['holm_p_within_declared_family']:.6g}", interval(e['f1'])] for e in effects if e['readout'] == rule and e['baseline'] == e['regime'] + '_NoComm']))
    text += ['## Difference between communication effects', '',
        'The following within-question contrast is (communication−NoComm under aware) minus (communication−NoComm under original). It is not binary, so no single-pair McNemar p-value is assigned. It is a deterministic readout-by-condition contrast, not identified causal mediation through abstentions.']
    text.append(mdtable(['Communication arm', 'EM interaction pp [CI]', 'F1 interaction pp [CI]'], [[e['arm'], interval(e['em']), interval(e['f1'])] for e in interactions]))
    text += ['For partitioned Broadcast, the21-question differential readout gain (52 NoComm gains versus31 Broadcast gains) reduces a net24-question communication gain to3. The point estimate attenuates87.5% in EM, with interaction−7.000pp [−10.333,−4.000]; F1 interaction is−8.701pp [−12.295,−5.341]. These percentages describe observed arithmetic, not the proportion of a causally isolated mechanism. Shared-evidence EM is tied under both repaired readouts.', '',
        'The Static partitioned point estimate reverses from+1.000 to−0.667EM points, with wide intervals. Learned partitioned effects shrink but remain slightly positive in each seed. Shared learned seed7/19 point effects remain positive; seed31 does not improve NoComm. The original qualitative claim of a large general partitioned communication benefit is weakened; a universal harm, benefit, equality or noninferiority conclusion is unsupported.', '',
        '## Quality–cost interpretation', '',
        'Partitioned seed19 retains a descriptive trade-off:65 correct at1,200 successful request references versus Broadcast65 at1,800; F1 difference−0.166pp [−1.597,+1.229]. Its3 EM wins/3 losses against Broadcast produce p=1. Seed7 nearly duplicates Broadcast in both quality and request demand; seed31 is cheaper with2 fewer correct endpoints. Shared seeds7/19 have favorable point estimates against Broadcast with fewer requests, but intervals and seed dependence prevent a general superiority claim. No post hoc best-seed recommendation is made.', '',
        'MatchedRandom is edge/quota-matched but not reliably recipient-call-matched. It matches recipient counts on only888/1,800 pairs, with912 mismatches. For P seed19 it matches on50/300 and for P seed31 on43/300. Therefore learned versus MatchedRandom cannot isolate a topology effect under equal receiver inference. All paired outcomes remain valid descriptive system-policy comparisons with their respective resource demands; selecting only matched rows would condition on policy-dependent behavior and is not used to manufacture a causal result. Counts and exact resource checks are in ' + link('results/physical_accounting.json') + '.', '',
        'These policies were trained/selected under the original reward/readout. They were not retrained for the repaired evaluation. The study evaluates existing frozen policies, not the best possible policy under a new objective. All pre-treatment strata (type, initial unanimity, number of substantive strings and annotated-support availability) are exhaustively saved in ' + link('results/all_pre_treatment_subgroups.json') + '. No post-treatment subgroup is promoted to a causal effect.']
    (ROOT / '03_COMMUNICATION_EFFECT_REASSESSMENT.md').write_text('\n'.join(text) + '\n')

    failures = []
    for t in transitions:
        if t['em_delta']:
            failures.append({'kind': 'communication_correctness_change', **t})
    for t in changes:
        if t['em_delta']:
            failures.append({'kind': 'readout_correctness_change', **t})
    save(ROOT / 'results/failure_case_index.json', failures)
    text = ['# Question-level failure mechanisms', '',
        '**Observable transitions are separated from causal explanations.** Every changed-correctness instance is included in ' + link('results/failure_case_index.json') + '. All16,200 paired communication rows, all2,378 normalized-answer readout changes and the complete sparse joint-cell memberships are saved; no favorable subset is substituted for the300-question analyses.', '',
        'A readout intervention holds the ordered response list fixed. A communication comparison can change answer identities, counts, abstentions and selected citations simultaneously. A newly correct proposal is temporal appearance, not proof it came from a delivered source. A correct proposal lost after update is observable damage, not proof of conformity. Exact gold agreement does not certify semantic source support.', '',
        '## Broadcast transitions under each identical readout']
    for regime in ['P', 'R']:
        for rule in RULES:
            group = [r for r in transitions if r['arm'] == regime + '_Broadcast' and r['readout'] == rule]
            cells = Counter((r['coverage_transition'], r['correctness_transition']) for r in group)
            text += ['', '### ' + regime + '_Broadcast: ' + rule]
            text.append(mdtable(['Any-correct initial→final', 'Vote00', 'Vote01', 'Vote10', 'Vote11'], [[c] + [cells[(c, v)] for v in ['00', '01', '10', '11']] for c in ['00', '01', '10', '11']]))
    text += ['Here01 means false→true and10 true→false. In partitioned original voting,23 gains retain correct coverage,2 gain coverage and1 loss loses coverage. Under aware voting,5 gains gain coverage,3 gains retain it, and all5 losses lose coverage. Under lexical-aware voting,5 newly covered gains and1 retained-coverage gain balance6 lost-coverage losses. Marginal coverage67→66 hides5 gained and6 lost memberships.', '',
        'Thus the matched residual does not establish a pure readout or generation mechanism. All13 aware EM-changing Broadcast questions change the candidate set; none is a fixed-set selection-only counterfactual. The offline readout intervention itself supplies the true fixed-proposal contrast. Copying an answer, independently correcting it and verifying a citation remain indistinguishable as causes in many saved updates.', '',
        '## Exact partitioned aware Broadcast gains and losses']
    for direction, label in [(100, 'Gains'), (-100, 'Losses')]:
        text += ['', label + ':']
        for t in transitions:
            if t['arm'] == 'P_Broadcast' and t['readout'] == 'abstention_aware' and t['em_delta'] == direction:
                text.append('- [' + t['question_id'] + '](' + t['raw_trace'] + '): coverage' + t['coverage_transition'] + '; ' + repr(t['before_answer']) + '→' + repr(t['after_answer']) + '; correct nodes' + str(t['correct_nodes_before']) + '→' + str(t['correct_nodes_after']) + '.')
    text += ['', '## All-arm joint transition inventory', '',
        'Each following row summarizes all300 questions for one arm/readout; coverage/vote labels identify exact joint cells. Complete question IDs for every cell are in ' + link('results/joint_transitions.json') + '. Repeated IDs across arms/readouts are paired observations, not independent examples.']
    text.append(mdtable(['Arm', 'Readout', 'Coverage/vote cells (n)'], [[j['arm'], j['readout'], '; '.join(k + '=' + str(v['n']) for k, v in j['cells'].items())] for j in read('joint_transitions.json')]))
    text += ['## Evidence availability and abstention', '',
        'The following initial-state audit uses the union of actual E5 initial request sentences, checked against annotated support identities. Both regimes share the same source unions. Counts are descriptive; missing annotations do not prove the union cannot answer the question.']
    strata = read('evidence_abstention_strata.json')
    text.append(mdtable(['Regime', 'Annotated support titles present', 'n', 'Initial agent abstentions/3n', 'All agents abstain', 'Aware correct', 'Any correct', 'Original overrides'],
        [[s['regime'], s['annotated_support_titles_present'], s['n'], f"{s['agent_abstentions']}/{3*s['n']}", s['all_abstaining'], s['correct'], s['any_correct'], s['original_abstention_overrides']] for s in strata if s['arm'].endswith('NoComm') and s['readout'] == 'abstention_aware']))
    text += ['For complete annotated support, shared initial responses abstain on7/228 agent slots, compared with148/228 partitioned slots. These are the same76 question IDs with different local evidence exposure, context lengths and proposal realizations. It is a striking observational diagnostic, not a randomized semantic-scarcity mediation estimate. Without complete annotated support, alternative answer-bearing sources and model knowledge remain possible. Independent sufficiency and citation-entailment labels are pending, not silently coded as negative.', '',
        'The case index records abstention overrides, node ties, rejected correct minorities, new/lost correct nodes, proposal multiplicity, set/multiset/order invariance and missing annotated support. Multiple flags can describe one case; their totals are not a disjoint causal decomposition. Insufficient source support is explicitly `NOT_INDEPENDENTLY_ASSESSED`. No paid model or judge was used to fill it in.']
    (ROOT / '04_QUESTION_LEVEL_FAILURE_MECHANISMS.md').write_text('\n'.join(text) + '\n')
    # Readability cleanup preserves code spans and Markdown/HTTP targets.
    for path in ROOT.glob('0[1-7]_*.md'):
        chunks = re.split(r'(\]\([^)]*\)|`[^`]*`|https?://\S+)', path.read_text())
        for i in range(0, len(chunks), 2):
            chunks[i] = re.sub(r'(?<![\w/])([A-Za-z]{2,})(?=\d)', r'\1 ', chunks[i])
            chunks[i] = re.sub(r'(?<=\d)(?=October|April|September|November|December|July|points|point|tokens|token|questions|question|completions|completion|seconds|hours|calls|judges)', ' ', chunks[i])
        path.write_text(''.join(chunks))
    print(json.dumps({'generated_reports': 4, 'matched_table_rows': len(joined), 'failure_case_instances': len(failures), 'new_provider_calls': 0}, indent=2))


if __name__ == '__main__': main()
