"""Explicit investigator assessments, separated from measured outputs.

This is a transparent record of judgments, not an automated novelty classifier.
"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

# All papers below were accessed as primary full texts. Venue status is deliberately
# limited to the edition verified, rather than inferred from secondary search hits.
PAPERS=[
('L1','OpenMAS-GCom. A Diagnostic Benchmark for Graph-enhanced Multi-Agent Systems','Kairui Yang; Xunkai Li; Kaixiang Zhang; Minghao An; Zekai Chen; Yuxuan Ba; Rong-Hua Li','arXiv:2609.21527v1, 18 September 2026; preprint edition reviewed','https://arxiv.org/html/2609.21527v1','2–4, 5, Appendix A and B intervention/record definitions',
 'Question: attribute organizational effects beyond final scores. Design: paired edge rewiring with degrees preserved, role removal, message corruption and worker failures; shared budget limits, 29-dataset interface and 400 multi-document stress tasks. Reports configuration-specific quality and resource changes. Limits: the estimand is the complete modified execution; equal limits do not imply equal realized calls, tokens or context. Longer interactive tasks remain an extension. Gap implication: generic component diagnosis and node/edge separation are already explicit. Exact activity-controlled Web effects are not thereby settled.'),
('L2','Discovering Efficient and Explainable Communication Topologies for LLM-based Multi-Agent Systems via Causal Inference','Junzhi Li; Peng He; Qirui Ji; Wei Wang; Lixiang Liu; Chuxiong Sun','arXiv:2608.12921v2, 14 August 2026; preprint edition reviewed','https://arxiv.org/html/2608.12921v2','Problem Formulation; E2-Explainer; Experiment; appendix token accounting and scope',
 'Question: find task-preserving communication subgraphs. Design: mask an edge and reexecute the frozen system; combine score and semantic stability, select a budgeted executable subgraph, distill an explainer. Six reasoning/coding benchmarks with Qwen3-8B; calibration uses training inputs, transfer spans graph generators. Reports token reductions and mixed task-level accuracy changes. Explicitly distinguishes edge/node counts from realized execution cost, with stronger savings after node pruning. Limits: downstream states change naturally; generator transfer is not Web evidence-regime transfer or proof of a single edge mechanism. Gap implication: merely observing that edge sparsity fails to predict total cost is already covered.'),
('L3','PEAR: Permutation-Equivariant Adaptive Routing Multi-Agent Debate','Yang Feng; Ziwei Xu; Xia Hu; Fengxiang He','arXiv:2606.20621v2, 31 August 2026; reviewed preprint edition, venue not needed for verdict','https://arxiv.org/html/2606.20621v2','3–6; Limitations; C.4–C.5',
 'Question: reduce positional advantage and error cascades through state-aware role assignment. Design: search relabelings of a base graph at fixed edge budget, score disagreement/confidence/influence, critique then revise. Four reasoning benchmarks and six backbones, with routing ablations and trajectory correction diagnostics. Reports gains over fixed and random routing. Limits: confidence calibration and debate cost remain concerns; equivariance theorems require exchangeability and compatible update/readout kernels. Gap implication: arbitrary seat identity, semantic roles and adaptive routing are already scientific objects. A Hotpot-only role permutation is chiefly an extension unless it establishes a new mechanism.'),
('L4','Adaptive Graph Pruning for Multi-Agent Communication','Boyi Li; Zhonghan Zhao; Der-Horng Lee; Gaoang Wang','arXiv:2506.02951v1, 3 June 2025; reviewed preprint edition','https://arxiv.org/html/2506.02951v1','3; 4.1–4.4; 5',
 'Question: jointly choose participating agents and communication strengths. Design: collect scored heterogeneous subgraphs and supervise a GNN with separate edge and node heads. Six knowledge/math/code benchmarks, gpt-4o-mini; component ablation compares pruning branches. Reports improved accuracy/cost tradeoffs. Limits: backend generality and modalities remain untested; supervision/search cost and a fixed indexed pool constrain transfer. Gap implication: participation and bandwidth are expressly separate degrees of freedom. This does not estimate the exact recipient-update-controlled effect in our scheduler.'),
('L5','AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration','Zhexuan Wang; Yutong Wang; Xuebo Liu; Liang Ding; Miao Zhang; Jie Liu; Min Zhang','ACL 2025, Volume 1: Long Papers, pages 24013–24035','https://aclanthology.org/2025.acl-long.1170/','PDF 3–4; domain transfer analysis; Limitations',
 'Question: remove redundant participation and exchanges across rounds. Design: node dropout from trained graph weights, then edge dropout over intra/inter-round communication. Three backbones; MMLU, GSM8K, MultiArith, AQuA, SVAMP, HumanEval; prompt and completion costs, component and transfer analyses. Reports efficiency and performance gains. Limits: conventional tasks, predefined roles/prompts, and residual high token cost. Transfer tests are largely within mathematics. Gap implication: fewer active agents as an efficiency mechanism and domain transfer are established topics, though evidence-regime interventions may still add specificity.'),
('L6','What Should Agents Say? Action-state Communication for Efficient Multi-Agent Systems','Chen Huang; Yuhao Wu; Wenxuan Zhang','arXiv:2606.05304v1, 3 June 2026; preprint edition reviewed','https://arxiv.org/html/2606.05304v1','3–5; 6 and implementation scope',
 'Question: which message content helps downstream agents efficiently? Design: compare full content, concise generation, conclusions, summaries and artifacts while holding the scaffold fixed; propose Action/State/Result handoffs. Includes split-evidence HotpotQA and 2Wiki, a four-role pipeline, three Qwen3 scales, field ablations and coding-harness transfer. Reports no uniformly best naive format, and evidence-grounded handoff gains. Limits: content formats also change turns and token usage; this is not an exact-length anchoring experiment. Gap implication: broad claims that evidence handoffs outperform bare conclusions or full transcripts are already directly tested on Web QA.'),
('L7','Beyond Memory Majority: Latent-Source Reasoning for Multi-Agent Memory Arbitration','Chenchen Lin; Wenhao Yuan; Xuehe Wang; Edith Cheuk Han Ngai','arXiv:2608.19701v1, 20 August 2026; preprint edition reviewed','https://arxiv.org/html/2608.19701v1','Methodology; Experiments; Appendix D correlation-aware construction',
 'Question: avoid false majorities from shared upstream evidence. Design: query-conditioned latent evidence slots, provenance priors, factor aggregation, and Expand/Trace/Stop recovery; controlled paraphrase/summary descendants plus distinct complementary entries. MemoryAgentBench, LongMemEval and LoCoMo variants; module ablations and recovery costs. Reports better arbitration and reduced correlation bias. Limits: synthetic derivations expose recorded lineage; genuine Web copying and source independence are harder to certify. Gap implication: duplicating a source versus adding independent evidence is already a direct intervention. Moving it to Wikipedia agents alone does not establish novelty.'),
('L8','When Debate Helps: Proposal Supply and Verification-Aware Readout in Multi-Agent Reasoning','Zihao Zhao; Tunyu Zhang; Haizhou Shi; Yusong Zhao; Xinxi Zhang; Hao Wang','arXiv:2610.04686v1, 3 October 2026; preprint edition reviewed','https://arxiv.org/html/2610.04686v1','2–5; Appendix A intervention controls; Appendix C',
 'Question: when does interaction recover correct minority proposals beyond voting? Design: decompose recovery/new-correctness/damage, hold proposals fixed while intervening on verification evidence, and construct coverage-selected societies. Two backbones, mathematical/scientific/multidiscipline tasks, matched computation and round-level controls. Reports headroom recovery and additional interaction effects. Limits: answer clustering, extra compute, limited open-ended validation, and damage to vote-correct cases. Gap implication: proposal supply versus selection and harmful flips are already directly studied; our E8 accounting alone is replication or application, not a new mechanism.'),
('L9','Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?','Hyeong Kyu Choi; Xiaojin Zhu; Sharon Li','arXiv:2508.17536v2, 23 October 2025; preprint edition reviewed','https://arxiv.org/html/2508.17536v2','2–3; 4; 6; Appendix H',
 'Question: how much debate benefit exceeds initial ensembling? Design: initial majority versus multiple debate topologies/depths, homogeneous and extended settings; stochastic belief-update analysis. Seven NLP benchmarks. Reports voting accounts for much observed benefit. Limits: the martingale conclusion relies on the modeled update assumptions; it is not a universal impossibility theorem for truth-sensitive evidence exchange. Gap implication: attribution to interaction needs a strong vote/private-compute baseline; another local vote sensitivity finding has limited novelty.'),
('L10','Sufficient Context: A New Lens on Retrieval Augmented Generation Systems','Hailey Joren; Jianyi Zhang; Chun-Sung Ferng; Da-Cheng Juan; Ankur Taly; Cyrus Rashtchian','arXiv:2411.06037v3, 23 April 2025; reviewed preprint edition','https://arxiv.org/html/2411.06037v3','3–5; supporting experiment definitions',
 'Question: distinguish retrieval insufficiency from failure to use available context. Design: a query/context sufficiency definition, 115 challenging human-labeled examples to validate an autorater, stratified RAG evaluation and selective generation. FreshQA, MuSiQue-Ans and HotpotQA. Reports both unsupported answering and unused sufficient context. Limits: autorater error and context truncation; annotation is not a randomized source-access intervention. Gap implication: missing benchmark supporting sentences cannot define unanswerability, and a retrieval-versus-utilization split by itself is already established.'),
('L11','Retrieval-Augmented Generation with Conflicting Evidence','Han Wang; Archiki Prasad; Elias Stengel-Eskin; Mohit Bansal','arXiv:2504.13079v2, 12 August 2025; COLM 2025 proceedings PDF also located','https://arxiv.org/html/2504.13079v2','3–6; 7 scope and computational appendix',
 'Question: integrate ambiguous, noisy and misleading retrieved evidence. Design: per-document agents, iterative debate and aggregator; RAMDocs varies support imbalance and entity-swapped misinformation. FaithEval, AmbigDocs and 500 RAMDocs queries; multiple backbones and aggregator/round ablations. Reports gains and degradation as misinformation increases. Limits: constructed conflicts and substantial residual error; support multiplicity is not necessarily common-source provenance. Gap implication: document frequency, minority evidence and conflicting-source debate are existing questions, while real lineage certification remains a distinct measurement challenge.'),
('L12','Don’t Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration','Shangbin Feng; Weijia Shi; Yike Wang; Wenxuan Ding; Vidhisha Balachandran; Yulia Tsvetkov','ACL 2024, Volume 1: Long Papers, pages 14664–14690','https://aclanthology.org/2024.acl-long.786/','PDF method sections; experiments; abstain/retrieval/multi-hop analyses; Limitations',
 'Question: identify knowledge gaps and abstain through collaboration. Design: cooperative feedback or competing alternatives, compared with calibration/training/prompt baselines; direct QA, absolute-abstention, retrieval and multi-hop analyses. Reports improved abstention/error tradeoffs. Limits: multiple-model overhead, model contamination uncertainty, domain and knowledge/safety differences. Gap implication: collaborative abstention and selective retrieval are not new; source-conditioned utility still requires valid answerability labels rather than suppressing a vote string.'),
('L13','MoDS: Moderating a Mixture of Document Speakers to Summarize Debatable Queries in Document Collections','Nishant Balepur; Alexa Siu; Nedim Lipka; Franck Dernoncourt; Tong Sun; Jordan Boyd-Graber; Puneet Mathur','NAACL 2025, Volume 1: Long Papers, pages 465–491','https://aclanthology.org/2025.naacl-long.20/','PDF 3–6; 8 Limitations',
 'Question: represent opposing document perspectives in balanced summaries. Design: moderator selects document speakers, tailors retrieval queries, builds a structured outline, and summarizes; compares content-planning and speaker-selection ablations. ConflictingQA and DebateQFS, principally GPT-4. Reports coverage/balance gains. Limits: LLM cost, prompt/model dependence and limited human evaluation; balancing perspectives is not factual independence. Gap implication: routing to document-specific expertise already has a meaningful Web application; our exact-answer traces do not validate this different output task.'),
('L14','Sequential Consensus for Multi-Agent LLM Debates: A Wald-SPRT compute governor with calibration-based failure detection','Andrea Morandi','arXiv:2605.19193v1, 18 May 2026; preprint edition reviewed','https://arxiv.org/html/2605.19193v1','III–V; VI-H; VII-C limitations and threats',
 'Question: when is consensus informative enough to stop? Design: judge-score likelihood ratio, explicit conditional-iid assumption, Beta calibration, simulations and real MMLU/GSM8K evaluation. Reports savings on GSM8K but cap-bound increased cost on MMLU. Limits: consensus is not correctness, calibration/domain failures, dependence and judge cost. Simulation headline results must be separated from real results. Gap implication: a consensus-based stopping rule and its calibration failure are existing contributions. A new correctness guarantee would need assumptions and evidence absent from this repository.'),
('L15','Improving Multi-Agent Debate with Sparse Communication Topology','Yunxuan Li; Yibing Du; Jiageng Zhang; Le Hou; Peter Grabowski; Yeqing Li; Eugene Ie','Findings of EMNLP 2024, pages 7281–7294','https://aclanthology.org/2024.findings-emnlp.427/','PDF 3–5, limitations/scope',
 'Question: can sparse static communication preserve debate performance at lower cost? Design: vary undirected connectivity among fixed agents and debate rounds; evaluate reasoning, MathVista and preference labeling with token accounting. Reports competitive sparse performance and cost reductions. Limits: static graphs and task/model specificity; density is not a realized-compute-controlled causal effect. Gap implication: sparse-versus-dense performance/cost is a mature baseline question, not a sufficient new contribution.'),
('L16','Proximal Policy Optimization Algorithms','John Schulman; Filip Wolski; Prafulla Dhariwal; Alec Radford; Oleg Klimov','arXiv:1707.06347, 2017; foundational preprint','https://arxiv.org/abs/1707.06347','PDF 3 and clipped objective figure',
 'The clipped surrogate is the minimum of unclipped and clipped ratio-weighted advantages. For positive advantage beyond the upper clipping point its policy-improvement term is flat. This mathematical definition supports the bounded current-code derivative check; fixing its local implementation is software validation rather than a new Web research question.'),
('L17','Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information','Rui Ai; Yuqi Pan; David Simchi-Levi; Milind Tambe; Haifeng Xu','arXiv:2510.01499v2, 19 May 2026; reviewed preprint edition','https://arxiv.org/html/2510.01499v2','2–3; 5–7; dependence scope',
 'Question: exploit model heterogeneity and agreement information beyond raw vote counts. Design: accuracy-weighted and second-order label-free estimators under a formal information model, simulations and UltraFeedback/MMLU/ARMMAN evaluations, with robustness tests. Reports improvements over voting. Limits: candidate-label symmetry and conditional-independence assumptions support the simplest optimality result; broader dependence is separately treated. Open-ended prompt-specific weighting remains future work. Gap implication: a better vote rule or observed disagreement alone does not establish a novel collaboration mechanism; theoretical optimality cannot be transferred to our free-form dependent agents.')]

DIMENSIONS=['Scientific importance','Novelty','Existing evidence','Causal identifiability','Feasibility','Statistical strength','WWW relevance','Robustness','Reproducibility']

# id, question, importance, known, gap, evidence, missing, intervention,
# positive, negative, fatal, references, scores
CANDIDATES=[
('K1','Do recipient activations, rather than edge routing, explain reported sparse-agent savings?',
 'Operators must know whether to optimize participation or communication.',
 'L1/L2/L4/L5 already distinguish participation, edges and execution costs.',
 'The exact controlled direct effect of evidence routing at fixed revision schedule remains unmeasured here.',
 'E5 has 912 recipient-count mismatches; P seed19/31 each activate one receiver on all 300 questions; W4 is inconclusive.',
 'A multi-system effect distinct from known node pruning; nondegenerate randomization support and provider-level cost logs.',
 'Freeze all initial states, activate all receivers under both peer and own-state updates, randomize source assignments and match token ceilings; use N>=4 where in-star support needs rewiring.',
 'A transportable decomposition could tell designers which execution component to optimize.',
 'An adequately precise zero routing effect would bound value for the registered systems, not establish universal absence.',
 'No substantive novelty established beyond existing component diagnosis; dominant N=3 in-star has no alternative graph conditional on its active recipient and edge count.',
 ['L1','L2','L4','L5'],[4,1,4,2,2,2,4,3,4]),
('K2','Does source/role assignment explain apparent communication-topology advantages?',
 'Web agents receive unequal evidence and expertise, so arbitrary seat choices can distort comparisons.',
 'L3 explicitly routes agent-to-role permutations; L1 includes role removals.',
 'Role-by-evidence interactions under source-grounded correctness are not identified by our CW/CCW contrast.',
 'Legacy permutations vary absolute scores; E2 CW/CCW uses the same cycle motif and asymmetric labeled sources.',
 'Crossed role/evidence permutations with order-invariant readout and multiple frozen source regimes.',
 'Randomize role, source bundle and graph labels independently before outputs; pair all configurations by question.',
 'A general interaction could guide placement of specialized Web agents.',
 'A narrow null would show label robustness for the chosen prompts only.',
 'A Web benchmark extension of already explicit positional-routing work; legacy evidence lacks clean provenance and independent replication.',
 ['L3','L1','L5'],[3,1,2,3,2,2,3,2,3]),
('K3','When does repeated upstream Web evidence become false independent corroboration during agent communication?',
 'Syndication and copying can amplify a single erroneous source into apparent agreement.',
 'L7 directly manipulates shared-source descendants; L11 studies evidence imbalance and misinformation.',
 'A causal contrast on genuinely traced Web copying, rather than synthetic copies, would be more specific.',
 'Saved facts have titles and sentence IDs, but these do not certify upstream independence; repeated proposals in E8 are only a diagnostic.',
 'A licensed corpus with validated derivation lineages, matched independent alternatives and blinded factuality labels.',
 'Randomize lineage duplication versus independent corroboration at fixed semantic evidence, length, agents and readout; distinguish metadata disclosure from actual dependence.',
 'A new real-source mechanism could improve provenance-sensitive Web evidence handling.',
 'A precise null would be informative only if lineage and intervention exposure were validated.',
 'The proposed synthetic mechanism is directly studied by L7; a real-Web version lacks a suitable identified evaluation population here.',
 ['L7','L11','L10'],[5,1,1,2,1,1,5,3,2]),
('K4','Are failures limited mainly by missing retrieval evidence or by failure to use available evidence?',
 'Retrieval and communication require different remedies in multi-hop search.',
 'L10 formalizes sufficiency; L6 directly tests split-evidence exchange; L11 studies document access.',
 'A quantified randomized retrieval-by-communication interaction is absent from current source-qualified data.',
 'Only 76/300 old pools contain all annotated support; E6 has 8 lexical-answer-bearing pools and no independent necessity qualification.',
 'Independent source-sufficiency annotations and an intervention that varies accessibility without changing difficulty or answer leakage.',
 'Randomize access to validated complementary sources and peer versus private revision, scoring supported answers and valid abstention separately.',
 'Could locate a causal bottleneck in a specified retrieval pipeline.',
 'A null remains ambiguous without proving that access changes meaningful evidence availability.',
 'The broad retrieval/utilization question is established; treating missing support as unanswerability invalidates the proposed shortcut.',
 ['L10','L6','L11'],[4,1,3,2,2,1,5,2,3]),
('K5','Does source sufficiency determine how aggregation changes the measured value and harm of communication?',
 'A readout that raises answerable EM may also select unsupported answers.',
 'L8 separates proposal/readout; L10 and L12 study sufficiency and abstention.',
 'A balanced interaction on semantically qualified paired Web contexts is unmeasured.',
 'Frozen P gain changes from 8 pp to 1 pp under aware voting, but remains 7.667 pp under strict majority; external arithmetic has no semantic-abstention intervention.',
 'Qualified labels, matched private revisions, immutable backbones and variance small enough for the independent population.',
 'On paired sufficient/insufficient contexts, compare peer versus own-state revision and apply frozen readouts offline; primary question-clustered utility interaction.',
 'Could identify a source-conditioned boundary for interpreting collaboration benefits.',
 'A tight null would bound the interaction for those contexts; weak abstention prevalence makes a null uninformative.',
 'No substantive mechanism beyond combined existing questions established; current external traces are ineligible, labels pending and high-variance power exceeds the available pool.',
 ['L8','L10','L12'],[4,2,4,3,2,2,5,3,4]),
('K6','Do sparse routing policies transfer across changes in Web evidence availability and conflict?',
 'Policies trained on one search regime may fail under retrieval shifts.',
 'L5 tests domain transfer and L2 transfers across graph generators.',
 'Evidence-regime transfer is a specific untested axis, but needs a mechanism beyond ordinary distribution shift.',
 'E5 policies differ greatly across seeds and collapse to simple activity patterns; no qualified transfer trial exists.',
 'Multiple frozen policies, independent train/test source regimes, common readout and meaningful interaction effect.',
 'Evaluate frozen policies on outcome-independent source corruption/access shifts against validation-selected static and random activity-matched controls.',
 'Might reveal a reusable failure mode in policy deployment.',
 'Failure alone could reflect local training bugs, poor policies or generic shift; success need not explain a mechanism.',
 'Generic transfer extension with no validated instrument or mechanism; a local PPO problem is not a WWW contribution.',
 ['L2','L5','L3'],[3,2,1,2,1,1,4,1,2]),
('K7','Does answer-conditioned message compression amplify an agent’s incorrect initial answer at fixed bandwidth?',
 'Short handoffs may preserve persuasive conclusions while deleting disconfirming evidence.',
 'L6 compares message content and grounding; L8 intervenes on verification content.',
 'The exact directional harm of a sender-answer-conditioned selector has not been isolated here.',
 'Legacy compressor scores sentences against the sender’s own extracted answer; E3-v2 is fixed-length evidence position, not this selector.',
 'A validated non-gold selector intervention with neutral and evidence-preserving controls, semantic adequacy labels and replicated harm.',
 'Freeze sender texts; randomize answer-conditioned versus query-conditioned versus randomized exact-length selections; all receivers revise with fixed roles/readout.',
 'Might quantify a specific harmful selection mechanism.',
 'A null could mean selector equivalence or weak manipulation, rather than absence of anchoring.',
 'At present it diagnoses a local heuristic; no established distinction from content/grounding interventions and no qualified manipulation population.',
 ['L6','L8','L11'],[3,2,2,3,2,1,3,2,3]),
('K8','Can consensus stopping avoid losing already-correct answers under dependent communication?',
 'Premature convergence can turn agreement into a misleading stopping signal.',
 'L14 studies consensus calibration failure; L8 measures damage; L3 studies influence and error cascades.',
 'A correctness-sensitive sequential guarantee under realistic dependence is absent, not supplied by ordinary SPRT thresholds.',
 'E4 has 6 wrong unanimous stops, no observed correction on continuation, identical final F1 and only hypothetical savings.',
 'Adequately many fresh late-correction cases and a calibrated stopping policy selected without test outcomes.',
 'Pre-register stopping rule; retain full trajectories in evaluation only; compare to fixed-depth and matched private-compute controls across source regimes.',
 'Could support a dependable deployment criterion if calibration transfers.',
 'No late corrections in a small sample cannot certify safety; a sufficiently precise harm bound would be required.',
 'Stopping and its calibration limits are already studied; repository trajectories do not identify a new dependence mechanism or validate a correctness guarantee.',
 ['L14','L8','L3'],[4,1,2,2,2,1,3,2,3]),
('K9','Do provider failures and refusals alter the effective evidence graph and bias efficiency comparisons?',
 'Retries and unavailable agents change real deployment costs and accessible evidence.',
 'L1 already disables workers and records retries/fallbacks; abstention differs from transport failures.',
 'Provider-specific missingness effects require a defined target population and failure mechanism.',
 'Historical E5 has uncertain attempts and incomplete usage; E6 has 0 failures in 144 calls, unsuitable for estimating failure impact.',
 'A representative failure distribution, declared failover semantics, and complete timestamped provider accounting.',
 'Randomize worker availability/failure injection before outcomes, freeze failover and include all assigned questions in intention-to-treat analysis.',
 'Could improve resilient execution if realistic Web consequences differ from existing worker-failure studies.',
 'An artificial failure null generalizes only to its injection law and could be operationally irrelevant.',
 'Existing worker-failure benchmark overlap and no representative local population; ledger repair alone is engineering.',
 ['L1','L12','L14'],[3,1,1,3,2,1,3,2,3])]

def main():
    papers=[dict(zip(('id','title','authors','publication_status','url','sections_read','assessment'),p)) for p in PAPERS]
    fields=('id','question','importance','current_knowledge','unresolved','repository_evidence','missing_evidence','intervention','positive_contribution','negative_contribution','rejection','closest_works','scores')
    candidates=[dict(zip(fields,c)) for c in CANDIDATES]
    for c in candidates:c.update(verdict='REJECT',survives=False,assessment_type='INVESTIGATOR_JUDGMENT_NOT_MEASURED_RESULT')
    (ROOT/'literature/references.json').write_text(json.dumps(papers,indent=2,ensure_ascii=False)+'\n')
    (ROOT/'results/candidates.json').write_text(json.dumps(candidates,indent=2,ensure_ascii=False)+'\n')
    text='# Literature frontier and gap map\n\nPrimary-source review refreshed 16 October 2026. This is a bounded search, not a proof that no novel question exists. No paper’s result was independently reproduced here. A preprint can defeat a novelty claim without its numerical claims being accepted as established fact. Publication status below identifies the edition verified; secondary claims of acceptance were not substituted for proceedings evidence.\n\n'
    for p in papers:
        text+=f"## {p['id']} — {p['title']}\n\n{p['authors']}. {p['publication_status']}. [Primary source]({p['url']}).\n\nSections examined: {p['sections_read']}.\n\n{p['assessment']}\n\n"
    text+='''## What remains unresolved, and what does not follow\n\nExact activity-controlled Web routing, naturally occurring copied-source dependence, and the communication-by-readout interaction under independently qualified sufficiency remain possible research spaces. None is established as a substantive new, feasible contribution by the present artifacts. Different datasets or an extra factorial term do not alone create a scientific gap. No reviewed work is claimed to have answered every possible formulation of these questions.\n\nGeneric sparse communication (K1), positional routing (K2), shared-source repetition (K3), retrieval/utilization separation (K4), readout/headroom (K5), transfer (K6), compact handoffs (K7), consensus stopping (K8), and worker failure (K9) all have direct precedents. The residual formulations fail evidence, instrument, population or contribution gates in report 03.\n\n## Search and reading limitations\n\nCoverage spans 2024–2026 collaboration, graph diagnosis, resource attribution, split-evidence Web QA, multi-agent RAG, abstention, provenance, aggregation and stopping, plus PPO as a mathematical reference. Relevant earlier foundations are reached through the primary papers; no new universal information-bottleneck or sequential-testing theorem is claimed. The arXiv literature-search skill guided retrieval; primary sources were inspected through the browser. Its terms and each source license apply. The complete used-paper URL list is this report and references.json. No paid literature or model endpoint was used.\n\nD3MAS (2510.10585) was screened but is not a decisive novelty source here. Unreviewed search leads (MANTA, topology mutation, workflow RL and other multi-agent RAG papers) are not counted as full-text evidence or used to reject candidates. Incorrect incidental page resolutions were excluded: arXiv 2502.10718 is not AgentDropout; NAACL-long.54 is not MoDS.\n'''
    (ROOT/'02_LITERATURE_FRONTIER_AND_GAP_MAP.md').write_text(text)

if __name__=='__main__':main()
