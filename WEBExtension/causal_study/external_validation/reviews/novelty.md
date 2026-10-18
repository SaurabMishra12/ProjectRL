# Independent adversarial review: novelty and external significance

Reviewer role: a separately delegated automated reviewer. Date: 16 October 2026. Scope: the new brief, the preserved matched-readout decision, and the primary sources cited below. This is a bounded literature and design review, not an exhaustive priority search, a human annotation, or an independent empirical replication. No inference or training was run for this review.

**Recommendation: REVISE. Present evidence supports C, a limited implementation audit. A findings paper remains conditional; a new benchmark or general theory is not established.** The verified local observation is that matched reaggregation changes one partitioned-evidence Broadcast–NoComm EM contrast from +8 to +1 percentage point on the same 300 questions. The independent implementation checks strengthen computational reliability, but they do not create independent datasets, model generations, or sampling populations. The reported 87.5% attenuation is arithmetic on these estimates, not a causal mediation fraction.

## Closest prior work and what it already covers

- **Debate or Vote** explicitly separates the initial ensemble from subsequent communication and evaluates whether debate adds value. Therefore, “communication gains can be explained by aggregation” is already occupied. Its mathematical conclusions depend on its response/update assumptions; this review does not extend them to a universal impossibility claim. [Primary paper, §§2–4](https://arxiv.org/html/2508.17536v2).
- **When Debate Helps** explicitly distinguishes proposal supply and readout, decomposes recovery and damage, and tests evidence interventions with fixed candidate proposals. Neither “minority correct answers can be lost by voting” nor “readout matters independently of candidate supply” is a new mechanism here. Its fixed-proposal probes are particularly close to the previously prepared citation-association direction. [Primary paper, §§3 and 5](https://arxiv.org/html/2610.04686v1).
- **Beyond Majority Voting** studies aggregation using model reliability and dependence information. An ordinary improvement over unweighted voting cannot by itself support a new evaluation theory. Its formal aggregation problem differs from a source-bound abstention contract, so it does not by itself settle the proposed interaction. [Primary paper, introduction and formal setting](https://arxiv.org/html/2510.01499v2).
- **Don’t Hallucinate, Abstain** already uses cooperative and competitive multi-LLM interaction to identify knowledge gaps and make abstention decisions, including retrieval and multihop analyses. Multi-agent abstention or collaboration-assisted uncertainty detection cannot be claimed as new. [ACL 2024 primary paper, §§1–3](https://aclanthology.org/2024.acl-long.786.pdf).
- **Sufficient Context** already distinguishes context sufficiency from answer entailment, analyzes errors by sufficiency, and evaluates selective generation across coverage levels. A sufficient context may even support a plausible answer different from the reference. Thus reference-answer presence and omitted annotated facts cannot replace the required semantic distinction. [Primary paper, §§3–5](https://arxiv.org/html/2411.06037v3).
- **MuSiQue-Full** already supplies answerable/unanswerable contrasts with identical questions and marginal context changes, and evaluates context-based answerability. Using it is an appropriate independent test, not a new benchmark contribution. A memorized correct world answer can still violate this context-bound task. [Original dataset paper, construction S7 and §7.1](https://aclanthology.org/2022.tacl-1.31.pdf).
- **EARS** explicitly treats sub-agent abstention as information passed to a coordinator, distinguishes failure categories, and evaluates a production system. “Abstention is a communication signal” is therefore also occupied, although that coordinator architecture differs from symmetric QA voting. [Primary preprint, §§1–2](https://arxiv.org/html/2606.18668v1).
- **MADAM-RAG** studies multi-agent processing of conflicting retrieved evidence and directly ablates the aggregator and number of debate rounds. It is especially close to any generic aggregator-by-communication claim in Web-derived RAG. The proposed study must distinguish its answerability semantics and paired estimation from those existing ablations. [Primary paper, §§4 and 6.1](https://arxiv.org/html/2504.13079v2).

These bounded comparisons establish substantial overlap. They do not establish that no paper has studied the exact proposed interaction.

## Is there an unresolved scientific question?

Potentially: **does a benchmark-compatible abstention contract materially change the estimated benefit of peer communication relative to an equally revised private control, and does that change differ between sufficient and insufficient evidence?** An independently verified answer would connect previously studied ingredients. Merely listing four familiar ingredients together is insufficient novelty.

A meaningful result would quantify when a common evaluation practice changes a practical system choice or falsely improves apparent utility by suppressing warranted abstention. It should survive competent, prespecified aggregation controls, preserve the benchmark’s response obligations, and show the relevant harm or benefit on independent data. Neither the local seven-point interaction nor an external GSM8K parsing-failure effect demonstrates this evidence-sufficiency mechanism. Parser failure and expressed abstention have different semantics.

The Web connection is currently limited to Web-derived, static Wikipedia evidence. To support a stronger Web contribution, the result must concern how retrieved evidence availability, provenance, or conflicting sources affects answer obligations and system comparisons. A generic math voting result or the word “Web” in the task description is insufficient. Live browsing is not automatically required; the claim must match the tested environment.

## A, B, and C assessment

**A — findings:** plausible only after independent confirmation of a consequential contract-by-communication effect, with false-answer behavior included. Directional heterogeneity is acceptable if estimated honestly and explained by a prespecified distinction. A larger rerun of the existing pipeline alone is weak evidence for a general failure mode.

**B — protocol or benchmark:** currently unsupported. Existing datasets and sufficiency/abstention distinctions cover much of the proposed content. A contribution would require validated reusable labels or evaluation contracts, independent implementations, documented ambiguity resolution, and evidence that competent existing protocols miss something consequential. Packaging scripts is not by itself benchmark novelty.

**C — implementation audit:** supported now. The audit is useful and should remain reproducible even if it does not warrant a WWW main-track submission. External traces that reproduce only malformed-output handling would strengthen this category without establishing A or B.

## Design conditions that matter for novelty

The prospective estimand must compare peer revision with private revision using the same update-call count, shared initial states, frozen parsers, and identical offline readouts. This identifies the effect of the specified communication bundle; it does not isolate citation truth, information transfer, or internal reasoning. Extra prompt tokens remain part of that bundle and must be measured.

Separate the benchmark’s context-bound answerability label from independently annotated sufficiency and the model’s decision to answer. Blinded source review must not see system outputs or method identities. Do not retrospectively relabel inconvenient unsupported answers as acceptable world knowledge. With paired MuSiQue-Full variants, both contexts and all models belong in one question cluster; they are not extra independent questions. A source-sufficiency interaction is exploratory unless its contrast, weighting, and multiplicity treatment are prospectively declared and powered.

## Required conclusions under all five outcomes

1. **Broadcast still helps under matched aggregation:** report the residual benefit and its uncertainty. Readout sensitivity may coexist with useful communication; abandon any universal “communication gain is an artifact” claim.
2. **Communication adds little value:** assess uncertainty and the minimum meaningful effect before claiming practical equivalence. This is potentially useful evidence about the tested system, not novelty beyond Debate or Vote by itself.
3. **Abstention-aware voting helps answerable QA but harms unanswerable QA:** reject the universal repair. Report false answers, coverage, selective risk, and benchmark utility. This is a meaningful failure boundary, but its novelty depends on an independently consequential communication interaction beyond existing abstention research.
4. **Models or datasets show opposite effects:** report the signed estimates and heterogeneity. Do not pool away the reversal or select a favorable backbone. The finding becomes conditional; a general direction-of-effect claim fails.
5. **A standard competent aggregator removes the issue:** stop the general-failure narrative and retain C. Unless independent evidence shows a prevalent evaluation practice with substantive consequences, purchasing more inference is unwarranted.

All five outcomes belong in the analysis report. None guarantees a publishable conference paper.

## Decision gate

**REVISE**, with no automatic recommendation to spend. Finish the zero-call external compatibility check and blinded annotation preparation. If those reveal only a local missing-abstention implementation choice, choose **STOP THIS NARRATIVE**. A conditional small pilot is defensible only after a distinct unresolved contract/sufficiency question, a competent conservative baseline, and a feasible question-cluster interaction budget are concrete. Do not use this review as paid-inference authorization.

## Final-evidence addendum

I inspected final reports 01, 03, 04, and 08 after the offline work. Their conclusions are proportionate to the reported evidence, and my recommendation remains **REVISE / category C / no paid pilot now**.

The public GSM8K reconstruction adds useful independent-pipeline evidence: eight configurations share 500 questions, and the original numeric decisions reconstruct exactly. The source already excludes parser failures; no separate semantic-abstention category is available. Consequently, original and aware decisions coincide by contract. This is neither an external replication of ProjectRL's abstention attenuation nor evidence that semantic abstention has no effect. The lexical and strict-majority contrasts establish only ordinary readout sensitivity, which the report correctly treats as known rather than a new Web finding. No matched private-revision control or evidence-sufficiency labels are present.

The historical strict-majority result also matters: its Broadcast–NoComm gain remains approximately 7.667 points. That prevents a claim that any reasonable change of aggregator removes communication benefit. Its lower answerable accuracy does not establish whether it improves unsupported-question behavior; no such empirical outcome exists here. A consensus threshold is a conservative policy choice, not an automatically correct universal benchmark contract.

MuSiQue's definitions, source availability, and evaluation contract justify its suitability assessment. Mirror hash verification is properly separated from unverified author-archive byte equality. The 48 blinded source cases remain PENDING, and a frozen question/model specification with unresolved launch gates is preparation rather than a completed confirmation. Neither item supplies new outcome evidence or benchmark novelty.

Report 08's stop rule is adequate because the next step is explicitly zero inference, continuation is conditional on an independently consequential unresolved problem, and failure of that gate leads to **STOP THIS NARRATIVE**. Interpretation should remain strict: annotation completion or provenance repair alone does not meet the novelty gate. A future pilot requires a concrete contrast that existing competent aggregation and the cited abstention/sufficiency literature do not already resolve, plus a feasible budget for the question-cluster interaction. If only the local implementation choice survives, retain the audit and stop this WWW narrative.
