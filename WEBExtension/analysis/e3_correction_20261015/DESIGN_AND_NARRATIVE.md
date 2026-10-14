# E3 correction and core-narrative experiment design — 15 October 2026

## Diagnosis verified from completed source and saved outputs

Original E3 is complete but its manipulation is ineffective: `exact_slice` uses `min(256, source_length)`. The median source is 7 tokens; 298 of 300 sources fit in the cap. Prefix and crop are consequently identical on 298 questions. The reported near-zero contrast cannot establish that evidence selection is unimportant, or that the two approaches are equivalent. It mostly compares the same input with itself. The source consists of a concise initial answer plus only citations that the sender returned; abundant available Wikipedia text is omitted when the sender abstains or returns no citations.

Do not change the historical file, hide its failure, lower the cap repeatedly until an attractive effect appears, or present a new run as prospective confirmation of a previously examined test set.

## Corrected E3-v2, authorized for real Azure execution

The 300 original question identities, roles, retrieved pools and initial agent outputs remain fixed. Reuse the original R states without regenerating initial answers. Build a gold-free source bank from **every sentence actually available to sender 0**, in frozen retrieval rank then original sentence order. Compare:

1. The first 128 exact tokenizer tokens.
2. A seeded randomized contiguous 128-token slice, restricted to valid UTF-8 substrings that differ from the prefix.

Both update receiver 1 once over edge 0→1, with identical initial state, receiver evidence, model, system prompt and decoding settings. Seed-randomize arm execution order per question. Freeze all payloads, eligibility, code hashes and question order before inference. No padding or fabricated evidence. Saved full HTTP bodies, requests, parsed completions, usage/refusals, unknown-cost liabilities, call timings and question checkpoints remain local.

Preflight: 299 eligible questions have distinct equal-length pairs at exactly 128 tokens. Question `5a8cfee555429941ae14df5c` has zero available documents. Retain it as `NO_SENDER_EVIDENCE_AVAILABLE`, no fabricated treatment and no new calls for it. This is an explicit evidence-availability exclusion from the **paired content estimand**, not removal of a bad model answer. There are 300 screened questions and 299 paired treatment questions; report both denominators.

Maximum first-attempt completions: 598. Saved NoComm and SelfUpdate receiver answers are descriptive anchors. Primary contrast: paired receiver answer F1, prefix minus crop. Secondary: EM, citation-set recall/precision, complete-fact delivery, invalid citations, refusals, actual input/output tokens, latency and logical calls. Bootstrap/sign-flip inference operates on questions; no equivalence or noninferiority claim without an independently justified margin. Preserve all signs of the effect.

**Interpretation boundaries:** E3-v2 estimates the effect of presenting different evidence fragments under this protocol. R receivers already see the same documents. It does not isolate novel information transfer or prove semantic evidence quality. Contiguous token crops can break sentences/JSON; report how many complete structured facts survive. Source-bank construction and cap changed from E3, so E3-v2 is a distinct follow-up. Because the original test results were inspected, label it exploratory even though this new payload plan is frozen before its outcomes.

## Is the proposed strictly resource-matched topology/adaptivity experiment already done?

No. E5's MatchedRandom selects graphs with the same **edge count and quota**, not the same realized payload or recipient count. Of 1,800 paired evaluations, 912 differ in recipient update calls and 254 differ in transmitted tokens. Static selection considered CW, CCW and OUTSTAR at quota 256; it did not include in-stars. P seed 31 selects the same in-star on all 300 questions; P seed 7 chooses broadcast on 294/300. These policies do not provide strong evidence of question-dependent routing adaptation. The existing Learned−MatchedRandom contrast is +1.01577 F1 in P with Holm-adjusted p=.502650, not multiplicity-robust.

Use the adjacent `existing_E5_resource_audit.json` to inspect genuinely matched saved subsets without paying for inference. Such subsets are descriptive: selecting pairs after seeing their actions/resources does not recreate a prospective experiment or answer the missing static in-star comparison. Missing counterfactual outputs cannot be recovered by algebra.

## Recommended new core test: separate recipient selection from route structure

### A. Resource-matched adaptive aggregator test (highest priority)

Use three agents, exactly two incoming edges to one aggregator, one receiver update, one communication round, and the same exact L tokens on each edge for **every arm and question**. Construct sender-visible evidence banks without gold. Require all three banks to support L before treatment assignment; record unsupported cases separately. Do not pad short banks or silently change quota per arm.

- **Learned dynamic in-star:** a new controller chooses the aggregator from {0,1,2} based only on observable initial state. Its action space is constrained to these three in-stars during training, not projected onto them after seeing held-out outcomes.
- **Static in-star:** choose the single best aggregator on validation, freeze it, and use it for every held-out question. Include all three fixed centers during validation, with identical resources.
- **Random in-star:** choose the center uniformly with a frozen seed independent of gold and outcomes. Match **recipient count** and calls; recipient identity changes by design.
- **Action-shuffled learned control:** permute the learned center assignments among held-out questions using a frozen seed, independently of outcomes. This preserves the action histogram and budget and directly tests whether the state/action association helps. A constant learned policy yields a degenerate shuffle, honestly indicating no demonstrated adaptation.

All edges transmit exactly L tokens; each arm consumes 2L communication tokens and one update. Score the selected aggregator answer in every arm; report initial aggregator F1 and the same common NoComm ensemble as anchors. Changing the center also changes which initial answer is used, so report the gain relative to the selected center's own initial answer. Describe this as **adaptive recipient/aggregator selection**, not a pure effect of graph structure with fixed recipients. Generated output tokens and latency remain outcomes; exact matching on future generated-token counts would bias comparisons.

Train and select on disjoint real questions. Use a **new uninspected held-out set** for confirmation, excluding previous train/validation/pilot/test question IDs and normalized duplicate questions; document document-title overlap rather than claiming universal pretraining-leakage elimination. Lock reward, seeds, source banks, budget and selection criteria before held-out inference. Three training seeds and 300 fresh held-out questions are a reasonable starting design, not a power guarantee. Use paired question-level bootstrap CIs, paired tests and Holm adjustment for the planned contrasts; do not pool seeds as independent question replicates. Report seed-specific action distributions/entropy and performance.

### B. Pure routing test with exact recipient identities

With three nodes, two edges and **one fixed recipient**, the in-star is unique (both other nodes must send to it). A different recipient-identical random in-star does not exist. Likewise, constraining every sender's out-degree and every recipient's in-degree can leave no alternatives. Enumerate the feasible graphs and publish how many distinct controls exist before requesting inference.

An inexpensive three-node construction is CW versus CCW with each sender sending the same exact-L message once and all three recipients updating once. Payload 3L and three calls match exactly, and every node receives once. E2 already implements this **static direction comparison in R**, but not a learned adaptive choice, not P replication, and not an in-star control. A fresh controller trained only on the two cycles, a validation-selected fixed cycle, a seeded random cycle and an action-shuffled controller can test state-dependent choice under these constraints.

If the main claim specifically concerns richer topology structure while holding recipient identities, sender loads and actual payload fixed, use four or more agents and a non-degenerate degree-constrained graph family. This needs a new controller/schema/training study; the existing three-agent controller cannot be asserted to evaluate it.

## Other additions, in priority order

1. Complete the corrected E3-v2, report fidelity checks and negative findings honestly.
2. Run test A if the paper claims learned adaptivity; add its shuffled-action control rather than more unmatched baselines.
3. Ensure the evidence source supports the information-complementarity claim: the current fullwiki candidate contexts frequently omit gold. On a separate real HotpotQA distractor cohort, retain both-gold-disjoint, both-gold-co-located and missing-evidence strata. Gold is used only to diagnose/stratify after predictions, not to choose routes or select successful questions. The official distractor setting supplies 10 paragraphs, whereas current cached candidate contexts are fullwiki-derived; do not relabel them.
4. A second frozen backbone and partition seeds support generality only after the core controlled contrast is informative. There is little value in many runs of a degenerate intervention.

These are proposed additions, not claimed executed results. Only E3-v2 is being run under the current request.

Primary references: [NIST randomized block designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm) motivates comparing treatments within question blocks; [official HotpotQA](https://hotpotqa.github.io/) defines the dataset/settings and answer/support evaluation. The exact constraints above are derived from this project's implementation and graph feasibility, not claimed as external experimental findings.

## Pseudocode

```
verify original E1/E3 completion hashes and frozen scientific source
load original gold-free inputs and saved R initial states
for each of all 300 original IDs:
    bank = concatenate all sender-visible evidence sentences in fixed order
    if bank is empty:
        freeze ineligible record and reason; no model request
    else:
        prefix = valid exact-128-token prefix(bank)
        crop = seeded valid exact-128-token nonidentical substring(bank)
        assert tokens(prefix) == tokens(crop) == 128
freeze bank/slices/order/eligibility/source hashes
for each unfinished question:
    for treatment in frozen randomized order:
        if identical request has a valid saved response: reuse it
        else: save request; call real Azure; save body before parsing
        update the same receiver with the same own evidence/initial answer
    attach saved baseline receiver answers
    score only after generation; fsync question result and progress
verify identities, pair distinctness, tokens and final file checksum
report all outcomes and uncertainty; retain empty-evidence record
```
