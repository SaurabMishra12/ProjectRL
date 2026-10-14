# SparseComm → The Web Conference 2027: Research and Experiment Blueprint

**Prepared 13 October 2026.** Basis: uploaded prior ACL manuscript, current 8-page WWW manuscript, three supplied simulated reviews, public ProjectRL repository, and external primary literature. **This is an experimental plan, not a claim that proposed experiments have been run or that any result is guaranteed.**

## Executive decision

**Preferred scientific question:** *When does communication improve multi-hop Web question answering because it transfers missing, complementary evidence, and how do routing, content choice, and stopping change that value under explicit resource constraints?*

**Positioning:** an evaluation and empirical-finding paper using SparseComm as an instrument; not a new state-of-the-art RL router. The current paper's narrower framing (“separating routing, rationing and stopping in an audit”) is defensible but not sufficient on its own against closely related causal topology papers. Add one **controlled, novel evidence-availability intervention** and one **budget-matched allocation intervention**; preserve negative or null results honestly.

**Scope:** candidate-based Wikipedia retrieval on HotpotQA, not live browsing. The Web challenge is whether evidence that is available somewhere in a retrieval-augmented agent team reaches the agent that needs it, not merely whether a sparse graph costs less.

## 1. Manuscripts and reviews: what is actually established

### Prior ACL paper (17 pages)

- Mathematical reasoning GSM8K/AQuA-RAT plus **closed-book** HotpotQA; 3 agents; learned continuous edge intensities, semantic sentence retention, PPO and stopping; exact-token transmission measurement in reported main experiments.
- Analyses include a factorial experiment, all 15 directed two-edge supports on GSM8K, 5 PPO realizations, and a fixed-depth/static-versus-controller control. Its component-isolation GSM8K control found controller top-two support identical to a static support and reported a stopping-associated token reduction.
- **Not reusable as an open-book Web estimate:** retrieval contexts, task objective, message selectors, stopping and scoring differ. Use as conceptual antecedent and supplementary evidence only. Do not import its prespecified 5-point noninferiority margin into a Web F1 claim.

### Current WWW paper (8 pages)

- 100 paired HotpotQA examples; each agent sees the **same** BM25-retrieved candidate passages within an evidence condition; 3 roles; four recorded runs W1–W4; recomputed official answer F1/EM.
- W1: 84.87% less **word-proxy transmitted volume** than broadcast, but F1 44.51 versus 48.36 (paired difference −3.85, CI [−8.85, 0.88]); **NoComm has F1 46.79 and zero communication** and therefore dominates sparse on these two measured axes.
- W2: prefix vs random crop, +2.08 F1 (CI [−0.92, 5.75]); approximate, not exact, volume matching, model-visible labels, only 5 nonzero pairs.
- W3: stopping reduces recorded message proxy 38.61–40.36%; prompt labels differ; unanimous agreement is the active branch and confidence threshold 80 is unreachable on [0,1]. No calibrated claim of correctness preservation.
- W4: current paper's *corrected* frozen-message comparison gives equal recorded word proxy by question, but controller chooses **the same two-edge support on all 100 questions**, 15 cases transmit zero; no fresh-message propagation, differing condition labels; controller vs static −1.87 F1, CI [−5.53,1.20]. It does not test learned task-adaptive routing.
- Word proxy excludes full prompt and provider input/output tokens; do not conflate with price or end-to-end compute.

### Three supplied simulated reviews

1. Reject: causal/experimental identification, cost accounting, 100-item single-harness sample, reproducibility.
2. Borderline: a bounded audit need not establish positive learned-routing effects, but originality and comparison with EMNLP 2025 and causal subgraph work are especially underdeveloped.
3. Cost-constrained plan: prioritize controlled static allocation, exact-budget prefix/crop, token audit; approximately 600–800 new receiver completions. Useful as a **fallback**, but alone it may still be a one-harness audit with weak Web-specific novelty.

## 2. Literature and candidate gap

| Work | Already known | Relevance to novelty |
|---|---|---|
| Li et al., Findings EMNLP 2024, *Improving Multi-Agent Debate with Sparse Communication Topology* | Sparse debate graphs can rival dense graphs. | “Sparse graphs work” is not novel. |
| Smit et al., ICML 2024, *Should We Be Going MAD?* | Debate does not always beat cheaper prompting/ensembles. | NoComm/simple baselines are mandatory. |
| Zhang et al., ICLR 2025, *Cut the Crap / AgentPrune* | Communication redundancy, edge/message pruning, costs. | A broad “routing versus communication cost” claim overlaps. |
| Zhang et al., ICML 2025, *G-Designer* | Learns task-aware graph topologies. | Do not claim to invent adaptive routing. |
| Wang et al., ACL 2025, *AgentDropout* | Removes redundant agents/messages and measures token costs. | Distinguish agent invocations, rounds and messages. |
| Shen et al., EMNLP 2025, *Understanding the Information Propagation Effects...* | Causal analysis of correct/incorrect answer propagation under topology. | Most important overlap: distinguish **documentary evidence availability/coverage at fixed payload budget**, not simply correct/wrong output diffusion. |
| Li et al., arXiv Aug 2026, *E2-Explainer* | Uses edge interventions to explain compact critical communication subgraphs. | Edge-level causal attribution alone is not new. |
| Tang & Yang, COLM 2024, *MultiHop-RAG* | Multi-hop retrieval requires distinct supporting documents. | Motivates Web-specific complementarity and evidence coverage. |

**Narrow research gap:** Existing efficiency/topology work and causal propagation studies do not automatically answer *how the value of message delivery changes when necessary retrieved evidence is distributed unevenly across agents and message budgets constrain which facts reach a recipient*. Demonstrate this particular gap empirically and carefully; do **not** claim no one has studied evidence distribution or information propagation before.

## 3. Proposed hypothesis and estimands

Let `q` denote a question, `E_q` an evidence-allocation regime, `T_q` a graph/allocation intervention, `C` a fixed content rule, `H` fixed stopping and `B_q` the verified per-question payload budget. Let `Y_q(E,T,C,H,B)` be official answer F1 (0–1).

### Primary estimand: incremental communication value

`D(E,T) = mean_q [Y_q(E,T,C,H,B) - Y_q(E,NoComm,C,H,0)]`, calculated **within the same evidence regime** using the same frozen initial-agent answers. This tests value above simply aggregating agents that have already read their own evidence.

### Evidence-complementarity effect modification

`I = D(Complementary,T) - D(Redundant,T)` using randomly assigned, preregistered evidence regimes and identical condition-neutral templates. This interaction is interpretable as the effect of the **experimental evidence-allocation regimes** on incremental communication value, *not* as a universal pure causal “complementarity coefficient.” Per-agent evidence distributions and input costs differ by design; document and report them.

### Controlled allocation effect

`A(E) = mean_q [Y_q(E,T_A,C,H,B_q) - Y_q(E,T_B,C,H,B_q)]` where the *same initial messages* are transmitted exactly once, with the same sender message selection and the same exact tokenizer-counted payload total on each question. Because destinations/recipient inputs change, the estimand is the effect of **who receives fixed information**, not an effect independent of recipient computation or prompt context.

**Quality:** official HotpotQA answer F1 primary; EM secondary, source/supporting-fact recall, answer citation correctness diagnostic. Track robust CIs, not just p-values.

## 4. Experiments in order of scientific importance

### E0: Code/provenance and cost audit — compulsory, no new inference if sufficient logs

1. Locate every experiment's actual source revision, job configuration, question-selection seed, model deployment, system/role prompts, response parameters, controller checkpoint hash and calibration, and cache origin.
2. Create file checksums; publish raw question-level predictions, exact prompt content *as executed*, role/edge routes each round, stop reasons, exception/cache flags, and scored answer strings.
3. Separate `payload_tokens`, `full_prompt_input_tokens`, `output_tokens`, `num_model_calls`, `rounds_executed`, `cache_hit`, `provider_usage`, and `latency` only if measured cleanly. Count actual model tokenizer tokens; label reconstructed counts as estimates.
4. Confirm the code path in the WWW harness: current selection is sentence-prefix, not prior ACL embedding similarity; verify the unreachable confidence branch; ensure model-visible condition tags are removed; use cache keys including model/version, decoding, prompt hash, evidence set, response settings, graph and seed.
5. Audit whether an actual usable controller checkpoint is available. If not, **do not call the static fixed out-star “learned task-adaptive routing.”**
6. Important repository discrepancy to reconcile: public `experiment_manifest.json` points to `results/W4_counterfactual_routing`, and visible entries in *that uncorrected raw results file* have different `comm_tokens` across static, learned, and random arms. The **WWW paper explicitly describes W4-corrected** with equal word proxy. Establish and publish the corrected run's exact file path, manifest and hashes so reviewers cannot accidentally reproduce the uncorrected experiment. This is a traceability issue, not proof the manuscript's corrected analysis is inaccurate.

**Exit criterion:** independently regenerate all reported tables from a frozen manifest.

### E1: Evidence-availability intervention — highest potential for a genuinely new WWW finding

**Question:** Is inter-agent communication more useful when agents each have access to different, complementary pieces of a multi-hop answer than when they see redundant evidence?

**Dataset:** choose an a priori, substantially larger HotpotQA validation sample (e.g., 200–300 questions) independent of observed performance; ideally include bridge and comparison strata. Keep candidate article pool and retrieval procedure fixed within each question. Test whether both supporting documents are available among the candidates. Report availability failure as a separate retrieval bottleneck, not a routing failure.

**Regimes:**
- `R`: shared/redundant evidence visible to all agents (the current Web pipeline, but rerun with cleaned prompts).
- `P`: complementary partition of the same union candidate pool across retrieval, verification and reasoning roles. Randomize the partition by fixed seed. Do not use gold supporting facts to choose inference-time partitions; gold is for *post-hoc* evidence-coverage diagnostics and an explicitly labeled oracle upper bound only.

Per regime, produce the same three role-conditioned initial answers and hold them frozen. Include a **NoComm ensemble** from those three answers, and one or two controlled delivery/recipient-update conditions using those same initial messages. Fix message selection, budget, decoding, round count and aggregation within each comparison. Retain source-document identifiers in evidence payloads so that supporting-fact delivery can be assessed.

**Main analysis:** Paired effect of communication versus NoComm within `R` and `P`; interaction between regime and communication value; stratify by whether two required supporting facts were accessible to different agents (annotation-defined, post hoc). Test whether correct answers rise when previously absent supporting evidence actually arrives. Measure *helped*, *harmed* and *unchanged* answers; do not rely only on averaged F1.

**Validity warning:** Different evidence partitions naturally change individual-agent prompt contents and costs. The causal target is the *assigned evidence-distribution regime*, not “pure complementarity independent of all other factors.” Record the per-agent and total costs, and avoid a claim of a single isolated mediating variable unless the design justifies it.

**Indicative completions:** With 100 questions, two regimes, 3 initial agents, 2 routing conditions, and 2 updated agents per condition (one step), `100 × 2 × (3 + 2 × 2) = 1,400` new calls. At 200 questions: ~2,800 calls. If all 3 recipients are updated for each condition, these counts increase to 1,800 / 3,600. Cache from a different prompt/evidence regime must **not** be silently substituted.

### E2: Exactly controlled *destination* intervention — best affordable mechanistic test

**Question:** Does the identity of the recipient matter when the same source messages and the same transmitted payload budget are held fixed?

Use the existing 100 frozen initial responses and a neutral one-step update template.

**Most tightly balanced option** (recommended if 600 calls affordable): compare two 3-edge directed cycles:
- `T_CW = {0→1, 1→2, 2→0}`
- `T_CCW = {0→2, 2→1, 1→0}`

Each agent sends its **same frozen, preselected initial message once** in both arms; each agent receives exactly one message and is updated exactly once; graph edge count, per-sender payloads, total received payload tokens, model-call count and model-visible template are held equal. Role assignment stays fixed within question; randomize/counterbalance which semantic roles occupy node identities **before** inference. There are 3 receiver calls per arm: `100×2×3 = 600` new completions.

Interpretation: a *one-step wiring/destination effect* under fixed information and invocation, **not** learned adaptive graph selection or genuine two-hop propagation. The fixed two-edge out-star/chain option in supplied Review 3 takes 400 calls for two arms and holds sender payload total fixed, but recipient roles/invocation differ; identify it honestly as a weaker allocation intervention.

**Optional follow-up:** allow revised messages to propagate over a second round with a fixed per-edge/per-question tokenizer quota. Divergent messages mean payload *identity* will not remain equal across arms, so label it a dynamic policy comparison, not a pure destination-only effect. Report the second round separately, with evidence transfer traces.

### E3: Exact-budget content selection — lower-cost secondary test

On one fixed source→receiver edge, compare order-preserving prefix, seeded random contiguous crop, and optionally a simple lexical evidence-oriented selection that is selected **without gold-answer access**. Construct all payloads at the **same measured tokenizer budget per question**, remove mode labels from model-visible text, use identical receiver prompts and initial answers. `100×2 = 200` calls for prefix vs crop, or `300` calls for three methods. Ensure deterministic padding/truncation does not inadvertently add answer cues. Primary content question is whether message *identity/informativeness* matters at equal volume; don't call prefix selection “semantic compression.” Evidence coverage may be more informative than a tiny F1 gap.

### E4: Matched-prefix stopping audit — optional if budget permits

Generate one complete, condition-neutral fixed-depth trajectory for each question and **replay proposed stop decisions on the same saved prefix**. Compute actual per-question stop depth, number of calls avoided *under a hypothetical online stop*, F1 at the stopped output versus continued output, wrong-unanimous rate and cases where continuation corrects an initially wrong consensus. Predeclare a maximal acceptable F1 degradation/noninferiority margin justified by the Web task; report its confidence bound. Do not claim a real monetary speedup from offline replay. The current W3 can remain exploratory if this test cannot be completed.

## 5. Priority under a tight deadline

**Lean route, roughly 600–800 completions:** E0 first; E2 using two balanced 3-edge cycles (`600`), plus E3 if 200 additional calls are affordable. Keep the narrative narrower: *fixed-information recipient allocation can be evaluated separately from message volume, but existing learned routing claims are not identified*. This can strengthen rigor but might **still lack a sufficiently consequential Web-specific finding**.

**Preferred stronger route, about 1,400–2,800 calls:** E0 + E1 with randomized evidence distribution and controlled two-edge allocation on 100–200 prespecified questions; add E3 only after the E1 result and artifact are stable. The core finding is then whether evidence scarcity/distribution moderates the value of communication. If the primary result is null, report the effect and CI as a tested boundary; do not force a positive result.

**Full robustness:** 300+ questions, a second benchmark or independent backbone, ≥3 partition/routing seeds, message-level evidence annotations and one dynamic round. Only do this if resources/time permit and all main validity checks pass.

Approximate completion counts are **not** billable token estimates. Estimate spend from pilot input/output token distributions before scaling.

## 6. Statistics and reporting rules

1. Freeze primary experiment, sample, hypothesis, endpoints, budget and seeds before evaluation. Write analysis in a script that runs without inspecting treatment labels.
2. Use HotpotQA official normalized answer F1 (0–100 points) primary; EM/evidence recall/cost secondary. Specify one primary contrast or one primary interaction for formal inference; label others exploratory and correct for multiplicity where appropriate.
3. Paired bootstrap CI by question; additionally account for independent partition/route/model seeds if run. Use appropriate paired randomization tests; with many tied per-question F1 values, do not assert precision from the number of questions alone.
4. Report per-question **improve, worsen, unchanged** transitions, including “correct→wrong” vs “wrong→correct,” and examples with evidence traces.
5. For quality preservation claims, predeclare a defensible margin and demonstrate an appropriate CI/one-sided test excludes an unacceptable loss. `p > 0.05` is not noninferiority.
6. Plot actual transmitted payload tokens and separately total prompt+completion tokens/model calls. Make no end-to-end speedup claim absent appropriate clean timing.
7. Report answer relevance **and** grounding: where possible, supporting-fact evidence recall and source-backed answer citations; avoid unsupported generation traces.

## 7. Suggested manuscript map (8 self-contained main pages, ≤12 total)

| Section | Decision |
|---|---|
| Title | Prefer *When Does Agent Communication Help Multi-Hop Web QA? Evidence Availability and Budget-Matched Message Delivery*. Use only if E1 is actually completed. Fallback: retain “Routing or Rationing?” with a clarified controlled-evaluation subtitle. |
| Abstract | Lead with empirical, conditional finding (after results exist), not policy buzzwords or proxy savings alone. Include exact estimand, benchmark scope, credible effect + CI, actual token accounting and limitations. |
| Introduction | Web challenge: complementary retrieved evidence may remain stranded in one agent; more communication isn't necessarily useful; currently NoComm dominates sparse on proxy/F1 axes. State two research questions and the intervention. |
| Related work | Distinguish sparse topology/debate, causal propagation/subgraph explanation, and multi-hop retrieval; specifically cite Shen et al. 2025 and Li et al. 2026. |
| Method | Define evidence regime, message content and budget, recipient allocation, stopping, and measured cost; define estimands. Remove or demote bookkeeping-only Proposition 1-type presentation unless it yields useful conditions/testable hypotheses. |
| Evaluation | Pictured intervention diagram, randomized question/evidence protocol, leakage checks, controls, identical prompt templates, cost ledger. |
| Results | E1 primary interaction and paired effects; E2 matched messages/recipient effect; E3 or E4 secondary. Retain W1–W4 as **legacy audit/baseline** with explicit distinct protocols, not as four simultaneous causal claims. |
| Discussion | State where communication helps, where it harms, and whether missing supporting evidence arrives; boundary between routing attribution and cost attribution. |
| Artifact / appendix | Full prompts, checksum manifest, fixed seeds, gold-label restrictions, stats scripts, legacy scoring corrections, detailed negative results and limits. |

**Required figure set:** (1) Three-panel diagram contrasting redundant and complementary evidence and routing; (2) “value of communication” paired F1 changes by evidence regime, with uncertainty; (3) actual delivered-token and total-inference-cost versus F1 operating points. **Required table:** condition × evidence regime × topology × F1 × EM × support recall × payload tokens × full input/output tokens × calls.

## 8. Repo-specific reconciliation checklist

Public repository: https://github.com/SaurabMishra12/ProjectRL

Known public `experiment_manifest.json` states commit `79a02ffcce0dfbd91ebfd267a1e0b57e7eb167db`, 100 fullwiki questions, gpt-oss-120b/Azure and W1–W4 directories; it records completion at 2026-10-07. Check whether this is genuinely the revision that generated **corrected** W3 and W4 in the October 13 PDF, because the file names and labels suggest an earlier pipeline state.

The prior repository audit (`PAPER_AUDIT_REPORT.md`, August 2026) documents legacy ACL/runtime inconsistencies such as nonexistent four-mode compression descriptions and checkpoint/calibration uncertainty. That audit is useful as a to-do list but **not** a substitute for auditing the current October Web runner line-by-line; the two implementations are different.

Prioritize resolving: (a) corrected result artifact path; (b) checkpoint presence and actual source of controller decisions; (c) version of the official HotpotQA scorer; (d) question sampling; (e) cache equivalence; (f) parsed provider token fields versus word proxy; (g) exact prompts including hidden condition strings; (h) stopping logic; (i) malformed/empty saved answers. If corrected W4 output cannot be linked to verifiable artifacts, downgrade/remove the W4 causal-looking claims.

## 9. Dates and gates

Official WWW 2027 research deadlines: **abstract 18 October 2026**, **paper 25 October 2026**, end-of-day Anywhere on Earth. Long paper: **8 self-contained main pages, ≤12 pages total including references/appendix**, anonymous `acmart` `sigconf,anonymous,review`. First page must explicitly address a Web research challenge; using HotpotQA alone does not ensure scope fit.

- **13–14 Oct:** audit and freeze experiment + validate sample/evidence support availability; 10–20 question pilot to check all token/call/prompt invariants (not to choose results).
- **15–17 Oct:** execute E1 if feasible; otherwise E2. Record complete costs and saved messages.
- **18 Oct:** submit a non-placeholder abstract making only supportable claims; author list must be final.
- **19–21 Oct:** paired analysis, robustness checks, rewrite the first eight pages around the completed result.
- **22–24 Oct:** independent table recreation, artifact and anonymity audit, paper proofreading.
- **25 Oct:** submit by AoE deadline.

**Go/no-go decision:** If E1 is not completed with clean controls, do not change the title/abstract to promise evidence-complementarity findings. Submit the narrower controlled-evaluation study with E2 and explicit limitations, recognizing that originality remains a substantive risk.

## 10. Primary research links

- WWW 2027 research call: https://www2027.thewebconf.org/research-track-papers/
- EMNLP 2025 causal propagation: https://aclanthology.org/2025.emnlp-main.623/
- E2-Explainer (2026 preprint): https://arxiv.org/abs/2608.12921
- AgentPrune: https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbc461518c59a2a8d64e70e2c38c4a0e-Abstract-Conference.html
- G-Designer: https://proceedings.mlr.press/v267/zhang25cu.html
- AgentDropout: https://aclanthology.org/2025.acl-long.1170/
- Sparse topology debate: https://aclanthology.org/2024.findings-emnlp.427/
- Should We Be Going MAD?: https://proceedings.mlr.press/v235/smit24a.html
- MultiHop-RAG: https://openreview.net/pdf?id=t4eB3zYWBK
- ProjectRL: https://github.com/SaurabMishra12/ProjectRL
- ProjectRL manifest: https://raw.githubusercontent.com/SaurabMishra12/ProjectRL/master/experiment_manifest.json
- ProjectRL uncorrected W4 record (must distinguish from W4-corrected): https://raw.githubusercontent.com/SaurabMishra12/ProjectRL/master/results/W4_counterfactual_routing/raw_results.jsonl

