# 3. Narrative support, manuscript corrections and replacement prose

Snapshot generated 2026-10-14T17:52:34.756444+00:00. Scope: completed frozen WEBExtension E1–E5, controller training/validation/pilot, operational cost evidence, and the supplied 13-page manuscript. Each experiment uses the same 300 held-out question identities; the unit of inference is the question, not the arm, agent, seed, or repeated epoch. All analysis is offline; no inference calls were issued. Earlier raw artifacts and the 129-question audit are preserved. Display tables round to six significant digits; CSV/JSON files retain machine-readable precision. NA is unknown or inapplicable, never an invented zero.

Run identifiers: `run_f6aa61bac2c3dfae`, `controller_study_a0b3768357484b45`, split `controller_splits_2e025f8ae3d9815e`. Input hashes and final artifact hashes are in `final_manifest.json`.

## Verdict

The completed experiments support a careful empirical paper about the conditions under which communication helps and the difficulty of attributing savings to routing, content selection and stopping. They do not support the current abstract's broad claim that each component independently preserves reasoning fidelity, or a general claim of learned routing superiority. Completion removes the missing-E5 concern; it does not remove the ineffective E3 manipulation, quality loss versus Broadcast in P, weak routing-specific evidence, single-backbone scope or legacy-provenance questions.

The strongest defensible story is: **evaluate communication as an information-delivery decision, audit whether intended interventions actually change inputs, and distinguish whole-policy quality/resource tradeoffs from isolated structural effects.** E1–E5 provide useful positive and negative evidence for this story. A strong final paper needs claims that survive those negative findings. Acceptance and novelty cannot be guaranteed by finishing an experiment checklist.

## Claim-by-claim changes to the supplied 13-page PDF

| Location | Current claim or presentation | Evidence and required action |
|---|---|---|
| Abstract, p.1 | 50.4–90.2% transmission reduction with statistical noninferiority across primary benchmarks | Those are historical P-series claims, not E-suite findings. Retain only with separately verified legacy raw provenance and NI protocol. For E5 report 79.53% R / 46.28% P payload reductions against its quota-capped Broadcast, alongside CIs and the P quality deficit. No E-suite NI margin exists. |
| Abstract, pp.1 and conclusion p.10 | Each mechanism independently reduces overhead while preserving fidelity; stopping/rationing are primary drivers | E3 has two actual changes, E4 saves only 1.5% hypothetical calls here, E5 changes support/quota/recipient calls together. Replace with protocol-specific measured tradeoffs and unresolved attribution. |
| Introduction, p.1 | One controller framework and common protocol across all results | Clearly distinguish legacy continuous SparseComm and the new categorical one-step SparseComm-Web instrument. The new study does not validate the old architecture or GAE/halting implementation. |
| Section 3, pp.3–4 | “measure-zero” equal-volume set | Equality may be systematic for discrete lengths, symmetric supports and sender-preserving cycles. Require a continuous-density assumption or remove this claim. E2 explicitly constructs equality. |
| Proposition 2 | Volume shifts demonstrate a large mediated quality effect | Algebraic decomposition is valid given well-defined potential outcomes, but a 3.4× volume change measures volume, not the size of the indirect outcome effect. Avoid claiming identified mediation. |
| Proposition 3 and proof | Every support induces a proper simplex face; fixed total alone establishes pure routing | Nonempty proper supports yield proper faces; full support yields the whole simplex, and the empty support is infeasible for positive mass. Specify message content/allocation, receiver calls and update rules as part of the intervention. |
| Proposition 4, p.4 | Uses arm mean volume in the observed causal contrast and says equal mean recovers fixed-mass identification | The displayed identity is for deterministic interventions at scalar volumes, not automatically the actual average outcome under question-varying realized volumes. Equal means do not imply pairwise equality. Replace as below. |
| Section 4.2, p.4; Algorithm 1, p.12; Table 14, p.13 | 20-D state, 61-D Gaussian action, low-rank intensities, MiniLM, GAE, MAP/Platt and learned halt | E5 has 53 observable inputs, 55 categorical actions, two 64-unit Tanh layers, Adam 3e-4, one-step reward, deterministic argmax at test, unweighted normalized majority. No MiniLM, GAE, calibrated MAP or trained stopping in E5. Give separate methods/configuration tables. |
| Section 5.1, p.5 | Common 512-token cap and shared calibrated aggregator for all protocols | E-suite freezes max_tokens=8192, context limit=32768, temperature=0 and a pinned exact tokenizer. Top-p should only be stated if explicitly fixed in the request. Do not inherit old 512-token settings. |
| Table 1, p.5 | Existing P/W map | Add E1–E5 with exact intervention, primary endpoint, n=300, evidence regime and inferential tier. E2 is overlapping reanalysis. Label old W1–W4 as historical. |
| Section 6.3, p.6 | Common caps remove the volume confound | Common caps need not match realized messages or receiver work. Describe capped whole-policy comparisons; actual per-question equality is necessary for the stated narrow structural estimand. |
| Section 7.1, p.8 | Prefix “demonstrating” superiority over arbitrary crops | The old W2 CI crosses zero and Holm p=.750; new E3 is degenerate. Change to an uncertain point estimate in the old protocol and an ineffective content intervention in E3. |
| Section 7.2, pp.8–9 | W4 proves sparse gains require fresh multi-round reasoning propagation | No randomized fresh-vs-replayed payload comparison identifies this necessity. State only that the fixed replayed support did not outperform its baseline. |
| Discussion, p.9 | Efficient coordination is primarily governed by adaptive depth rather than per-question rewiring | This is not established across protocols; E4 shows small savings, and E5 is one-step. Explain when each effect was observed and when evidence is absent. |
| Section 9, p.10 | Web word-proxy costs and condition tags | Keep this as a historical limitation of W1–W4. E-suite uses exact pinned-token counts and internal cache namespaces. Clearly separate these evidence tiers. |
| Table 3 vs Table 11, pp.6/13 | Peak PPO and five-seed summaries | Audit identities and denominator: cap-100 Table 11 lists a 94.0 seed but Table 3 calls 92.67 the “peak”. If different runs are intended, label them; otherwise correct from originals. Current E-suite does not validate either number. |
| Table 8 vs Table 2 and text, pp.12/6 | Hotpot NI p-values at margins 5 and 10 | Main text gives .0012 at margin 5; table transcriptions differ (.001 and <.0012). Recompute from legacy paired records and use one rounding convention. |
| Artifact URL, p.1/p.10 | All code/checkpoints/raw traces available | This analysis did not verify the anonymous repository contents. Publish a sanitized manifest and verify accessibility/completeness before retaining that assertion. |

## Corrected estimand statement

A scalar token total does not specify which text is delivered to which recipient. Let a fixed, fully specified allocation/content rule define the intervention Y_q(t,v). If t changes message versions or recipient calls beyond the intended destination intervention, name those changes rather than assuming them away. For a common reference volume v0(q) feasible for each compared arm, define:

TE = E_q[Y_q(t,V_t(q)) − Y_q(t′,V_t′(q))].

CDE(v0) = E_q[Y_q(t,v0(q)) − Y_q(t′,v0(q))].

Then TE = CDE(v0) + R_t − R_t′, where R_t = E_q[Y_q(t,V_t(q)) − Y_q(t,v0(q))]. If each question/arm has a valid Lipschitz bound L_q,t over the relevant feasible interval, then

|TE − CDE(v0)| ≤ E_q[L_q,t |V_t(q)−v0(q)| + L_q,t′ |V_t′(q)−v0(q)|].

This is a conditional sensitivity bound, not an empirically estimated correction. No L values were measured in E1–E5. Under exact question-level equality V_t(q)=V_t′(q)=v0(q), the residual terms vanish, provided the intervention is well-defined and other mechanisms are controlled. Equal dataset means are insufficient. A common upper cap alone supplies neither equality nor an informative outcome bound. The scalar-volume decomposition does not control content quality or recipient computation by itself.

For fixed binary A, the projection argument in Proposition 1 is elementary and correct for fixed B≥0. In a multi-round system, B can itself change with routing, so the fixed-B support subtraction formula should not be presented as the entire policy-induced change in generated volume. Avoid treating these bookkeeping identities as a novel causal identification theorem without a precise assumptions/contribution comparison to prior mediation work.

## Recommended paper structure and figure replacements

1. Abstract/introduction: information availability, interference and resource-attribution challenge in candidate-based Web QA; state the one-backbone scope on the first page.
2. Estimands and protocol validity: define controlled destination allocation versus whole-policy effects, include a diagram of shared versus partitioned evidence and a check of whether message interventions actually differ.
3. Methods: categorical controller, saved-data split, reward and validation-only selection; separate legacy continuous architecture if retained.
4. Results: E1 primary interaction first, followed by conditional P gains; E2/E3 validity findings; E4 stopping replay; E5 full policy tradeoffs with all seeds.
5. Evidence availability and failure analysis: the three gold strata, missing-candidate issue, real harmed/helped examples, seed-dependent static behavior.
6. Discussion/limitations: partial support for the narrative, no global NI claim, single backbone/benchmark/partition, unknown bills, legacy results requiring their own audit.

Replace a “paper in one figure” that implies uniform superiority with (a) evidence-availability counts, (b) primary and E5 paired effects with uncertainty, and (c) payload/call tradeoffs against both Broadcast and NoComm. Show all selected seeds rather than only a peak. Move full scalar/stratum tables to an appendix linked to Report 1. State that E2 shares E1 data. Keep E3's ineffective manipulation visible instead of deleting its null result.

## Suggested title

**When Does Communication Help? Evidence Availability and Resource Attribution in Multi-Agent Question Answering**

A version preserving the current title is **Signal over Chatter: Auditing Evidence Delivery and Communication Tradeoffs in Multi-Agent Question Answering**. “Decomposing routing, rationing and stopping” should be qualified if used, because the new suite does not fully identify all three components independently.

## Replacement abstract (for the verified E-suite scope)

Multi-agent question-answering systems exchange evidence through messages whose routing, content and length jointly affect both predictions and resource use. We study these effects in a controlled three-agent setting using 300 paired HotpotQA fullwiki candidate-retrieval questions and a frozen gpt-oss-120b deployment. Our protocols compare shared and partitioned evidence, reverse cyclic destinations while preserving sender payloads, test exact-length content selection, replay consensus stopping, and evaluate a validation-selected categorical PPO controller. The primary interaction between communication and evidence partitioning is inconclusive: 2.88 answer-F1 points (95% CI −0.26 to 6.01). In partitioned evidence, the learned policies improve over NoComm by 5.50 points, but underperform quota-capped broadcast by 4.43 points while transmitting 46.28% fewer payload tokens. Under shared evidence, they transmit 79.53% fewer payload tokens than broadcast without an established quality difference, although NoComm remains a strong lower-cost baseline. Input-level audits reveal that 224 of 300 selected pools lack at least one annotated gold article, and 298 prefix/crop pairs are identical. These findings support conditional communication gains and highlight the need to verify intervention strength and realized resources before attributing performance to graph structure.

This abstract intentionally describes only authenticated new results. Incorporating legacy multi-benchmark claims requires independent evidence validation and an explicit two-instrument methods section.

## Replacement results paragraph

On partitioned evidence, the three selected controller policies achieve a seed-mean answer F1 of 10.56, compared with 5.06 for NoComm, 6.22 for SelfUpdate and 6.42 for the validation-selected static support. Paired F1 gains over these baselines remain significant under the report's Holm correction. Broadcast achieves 14.99 F1; the learned policies therefore trade 4.43 F1 points (95% CI −6.31 to −2.71 for learned minus broadcast) for 46.28% fewer transmitted payload tokens and 22.35% fewer logical calls. The learned policies do not establish an F1 advantage over the edge-count/quota-matched random control after multiplicity adjustment. This control also differs in realized recipient calls and payload on some questions, so it is a whole-policy comparison rather than a pure routing intervention.

## Replacement discussion and limitations paragraph

The results support communication when agents hold restricted evidence, but do not imply that a learned router is uniformly preferable. Shared-evidence NoComm already performs comparably to communicating policies with fewer calls, while the partitioned controller's quality is lower than broadcast. Deterministic routing is strongly seed-dependent: one selected policy broadcasts on 98% of partitioned questions, and another uses a single in-star throughout. The apparent content comparison is also weak because most paired payloads are identical. These findings motivate reporting realized resources, policy diversity and intervention fidelity together with answer quality. Our evidence is limited to one deployment, one benchmark sample and three optimizer seeds; it does not establish cross-backbone robustness, citation entailment, calibrated noninferiority, or realized monetary savings.

## Remaining work, in priority order

**Can be fixed from current evidence:** replace unsupported claims, use the correct categorical method/settings, keep the primary null visible, separate old/new protocols and token units, report all seeds and resource mismatches, repair Proposition 4, add costs/missingness, and supply a reproducibility manifest.

**Requires additional evidence if those claims remain:** a content intervention that changes meaningful text on enough questions; recipient/call/payload-matched routing controls; independent partition/role seeds and preferably a second backbone or benchmark; an online stopping run if actual latency/billing savings are claimed; separately authenticated P-series/W-series raw records and preregistration for historical NI claims. These were not silently run or fabricated. Define new protocols before new data collection and retain the present test results unchanged; now-observed held-out data must not become a tuning target for a new confirmatory claim.

**Literature/venue validation:** this report audits claim-evidence alignment, not an exhaustive novelty ranking or every bibliographic entry. The official WWW2027 research-track call requires substantive Web relevance on the first page; candidate Wikipedia retrieval alone does not establish it. Explain the scientific information-access challenge, and avoid describing offline candidate reranking as live-Web search or retrieval over all Wikipedia. Primary references checked: [WWW2027 research track](https://www2027.thewebconf.org/research-track-papers/), [HotpotQA](https://hotpotqa.github.io/), [official evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).

## Adversarial self-review

- Contribution: a useful controlled empirical audit is supported; broad algorithmic superiority and full causal decomposition are not.
- Clarity: distinguish E from P/W, F1 from thresholded accuracy, actual tokens from proxies, and policy means from a deployed ensemble.
- Experimental strength: all planned E-suite rows exist; that does not repair E3's ineffective manipulation or small subgroups.
- Evaluation completeness: reports are complete within the frozen scope; independent robustness and legacy provenance remain separate work.
- Method soundness: exact split checks and frozen actor replay pass; arbitrary-volume potential outcomes require explicit implementation assumptions, and random controls do not equalize all realized resources.
