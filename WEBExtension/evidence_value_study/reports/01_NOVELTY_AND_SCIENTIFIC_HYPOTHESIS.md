# E9 novelty gate and scientific hypothesis

Review date: 2026-10-16. This is a prospective design assessment. It contains no new model results. Primary-source details, exact versions, short extracts and reading limits are in [primary_sources.json](../source_validation/literature/primary_sources.json); queries, failed leads and exclusions are in [search_and_reading_log.json](../source_validation/literature/search_and_reading_log.json).

## Gate verdict

**Conditional GO for a narrow diagnostic experiment; high overlap and incremental-contribution risk.** The testable object is whether removing a necessary source from a receiver changes the *net value of consulting the same genuine peer packet instead of spending the same receiver update call privately*. The inspected prior interventions do not identify this particular interaction. This is a bounded assessment of retrieved sources, not a guarantee of originality, acceptance, or comprehensive coverage.

The substantive alternative explanation is that an apparently useful consultation merely gives a receiver another opportunity to reason. The private-update control addresses that explanation within each evidence placement. Reusing the identical sender packet across placements then asks whether the packet's utility depends on the receiver's access to its source, rather than on changes in sender quality. A reversal from useful consultation under missing evidence to harmful or unhelpful consultation under complete evidence would affect when a system should expose a receiver to peers. This is more informative than reporting another average multi-agent gain, although the protocol alone does not establish a new algorithm or a deployable routing policy.

**STOP the broad novelty claims.** Complementary evidence, information asymmetry, frozen peer responses, evidence-aware readout, source sufficiency, communication costs and selective routing already have direct prior treatments. Moving those observations to another dataset, using equal call counts, adding human labels, or changing the final metric would by themselves be procedural distinctions. The GO applies only while the receiver-access interaction and its private-compute alternative remain the scientific question.

## Three closest collisions

1. **HiddenBench — Yuxuan Li, Aoi Naito and Hirokazu Shirado.** First preprint 2025-05-15; inspected v4 dated 2026-05-13; published at ICML 2026, PMLR 306:69250–69268. Hidden versus full profiles, pre/post discussion, and forced-disclosure probes already isolate distributed-information failures. Its main contrasts do not supply E9's matched private re-update of one receiver. Treat its information-pooling diagnosis as prior knowledge, not an E9 discovery. [Official publication](https://proceedings.mlr.press/v306/li26ej.html), [method and ablations](https://arxiv.org/html/2505.11556v4).

2. **InfoDelphi — Yuante Li, Yicheng Tao, Kate Zhang, Taozhi Wang, Gefei Gu and Yaxin Zhou.** Preprint submitted 2026-07-02; v1 inspected; no peer-reviewed venue independently established. Its fixed-evidence forecasting experiment manipulates public/private partitions and rationale sharing. This directly collides with the broad information-diversity explanation. Its inspected one-component-at-a-time ablations do not estimate E9's crossed receiver contrast against private revision. [Metadata](https://arxiv.org/abs/2607.01661), [Sections 3–5 and Table 2](https://arxiv.org/html/2607.01661v1).

3. **When Debate Helps — Zihao Zhao, Tunyu Zhang, Haizhou Shi, Yusong Zhao, Xinxi Zhang and Hao Wang.** Preprint submitted 2026-10-03; v1 inspected; no peer-reviewed venue independently established. Fixed-proposal probes already freeze target states and vary peer evidential content. Consequently, a frozen-state intervention is not itself novel. E9 instead changes receiver source access and uses unselected genuine packets; it does not select correct peers or inject gold-support messages. [Metadata](https://arxiv.org/abs/2610.04686), [Section 5 and Table 3](https://arxiv.org/html/2610.04686v1).

The saved records specify each paper's datasets, actual outcome contrasts, numerical findings and causal limits. These three are closest in intervention structure, not a claimed exhaustive ordering of the literature.

## Other boundaries checked

AgentWebBench already studies provider-restricted evidence access and synthesis; MASS-RAG already studies complementary evidence views and synthesis. Their complete-system comparisons do not isolate this receiver interaction. Sufficient Context and MuSiQue-Full make evidence sufficiency and answerability established questions. HotpotQA support annotations identify candidates for examination; they do not certify necessary-source removal in our retained context. MasRouter and PACT establish routing and recipient context-processing costs as existing design objects. [AgentWebBench](https://arxiv.org/html/2604.10938v1), [MASS-RAG](https://aclanthology.org/2026.findings-acl.480/), [Sufficient Context](https://arxiv.org/html/2411.06037v3), [MuSiQue](https://aclanthology.org/2022.tacl-1.31/), [HotpotQA](https://aclanthology.org/D18-1259/), [MasRouter](https://arxiv.org/html/2502.11133v1), [PACT](https://arxiv.org/html/2606.05304v1).

Two further collisions merit explicit retention. UnCoBench manipulates incomplete colleague evidence and verifies role necessity; E9 cannot claim to introduce either idea. M3-ACE uses frozen assistant contexts to revise an anchor. Tran and Kiela compare single/multi-agent reasoning at matched thinking budgets and study degradation. Their Section 5.3 prose describes context perturbations, while Appendix E specifies corruption of generated thought text. That unresolved implementation scope prevents treating it as an exact necessary-document-access intervention here. [UnCoBench](https://arxiv.org/html/2610.16117v2), [M3-ACE](https://arxiv.org/html/2603.08369v1), [Tran and Kiela, including Appendix E](https://arxiv.org/html/2604.02460v2).

## Proposed scientific contrast

For a question q, let A and B be independently qualified necessary documents and d1–d4 be retained distractors. The global pool is always the same six documents. Before inspecting model outcomes, counterbalance receiver/sender identities, which support document is retained, and document positions.

Use these placements:

- Complete receiver: receiver A+B; designated sender B+d2; third agent d1+d3+d4.
- Missing receiver: receiver A+d1; designated sender B+d2; third agent d1+d3+d4.

The six-document union and the per-agent document counts 2/2/3 remain fixed. These allocations intentionally overlap. Human qualification must examine the actual rendered texts, truncation and all potential alternative clues: A+B is sufficient, removing either necessary document breaks the intended contextual inference, and the distractors do not supply the missing connection. Supporting-fact labels alone are insufficient. Contextual necessity does not imply that a model cannot answer from memorized knowledge.

Generate three independent initial responses per placement. Within each placement, fork the cached receiver initial state into exactly one private revision and exactly one consultation update. For both consultation branches reuse the *same* frozen designated-sender packet generated in the complete placement, since its source inputs are identical. Other initial responses are diagnostic only. The packet must be genuine, generated without gold answers or support labels, and chosen by a rule fixed before scoring. Sender-selected verbatim quotes may be checked against visible source text. Do not select or rewrite a packet because it supports the correct answer.

The study therefore uses six initial completions, four receiver update completions and one single-agent completion over the full six-document union: **11 completions/question**, before retries or prespecified replications. The duplicated sender/third initial calls are research overhead. This accounting is not a deployment-efficiency claim. The union baseline contextualizes the cost of distributing access, but one union call does not provide a matched-compute upper bound or enter the primary interaction.

Let Y_q(c,u) be receiver answer F1 after an update, with c in {missing, complete} and u in {peer, private}. Define

```text
tau_missing  = E[Y(missing, peer)  - Y(missing, private)]
tau_complete = E[Y(complete, peer) - Y(complete, private)]
delta = tau_missing - tau_complete
```

The primary hypothesis is delta > 0; the null for the prespecified primary test is delta = 0. Estimate the four-cell interaction using paired question-level contrasts and uncertainty clustered at the question level. Initial receiver responses are shared only between private/peer branches of a placement; they naturally differ across source placements. Include all qualified questions in the primary contrast, without selecting initial errors, correct sender answers, successful quote transmission or disagreement cases. The primary endpoint is the receiver answer, not majority vote.

The estimand concerns the total utility of this consultation package. It includes evidence, interpretations, answer suggestions, anchoring and distraction. Even a source-faithful quote can contain the final answer. A positive interaction therefore does not identify pure evidence transfer, persuasion, or verification as separate mediators. Equal receiver call counts control a discrete opportunity for extra reasoning; they do not equalize input tokens, actual output tokens, reasoning tokens or monetary cost.

## What the outcomes could establish

A positive delta with positive tau_missing and negligible or negative tau_complete would support access-sensitive consultation in this qualified cohort. A positive delta with both simple effects negative would instead indicate that consultation is harmful in both placements, with different severity. Report the simple effects alongside the interaction so those cases cannot be confused.

A precise small interaction would weaken source access as a useful moderator for this packet and model; a negative interaction would challenge the proposed missing-evidence advantage. A wide interval is inconclusive. A null can also reflect weak or unreliable sender packets, so retain packet coverage diagnostics without conditioning the primary estimand on them. Predeclare a practically relevant F1 threshold and sample-size rationale before live inference.

Human source-access labels are researcher information. They do not show that a real receiver can detect its own deficiency. Any learned consultation policy requires a later held-out test using only inference-time signals and charging the cost of producing them. E9's immediate value is to decide whether that policy premise deserves further work.

Reopen the gate and **STOP further experimental implementation** if an exact prior receiver-access × matched-private-update study is verified, if remaining distinctions are only operational, or if contextual necessity cannot be qualified. Offline candidate annotation and preserving the evidence trail remain useful under a STOP. No literature finding here licenses claims about unrun E9 results.

