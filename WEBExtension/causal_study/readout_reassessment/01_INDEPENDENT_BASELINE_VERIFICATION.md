# Independent baseline verification

**Zero new inference. All original E5 endpoints are independently reproduced.** The new implementation reads raw E5 JSONL and original provider caches, not the E8 reconstructed outcome table. It imports neither the original vote nor the old analysis. Previous metadata is used to locate and hash-check source files; previous claimed counts are not prediction inputs.

## Independent reproduction

Every arm contains the same 300 unique questions and three ordered final responses. All 6,000 original answers, EM values and F1 values agree with an independent normalized-plurality implementation. All 18,000 proposal references and 18,000 readout scores are cross-checked against the three inspected pure functions extracted from the preserved official Hotpot evaluator. The original frozen source snapshot and current normalization, scoring, majority, role assignment, condition and routed-cost functions have identical ASTs. See [results/original_code_comparison.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/results/original_code_comparison.json) and [results/verification.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/results/verification.json).


| Regime/arm | Original correct | Aware correct | Aware lexical correct | Any-correct coverage |
|---|---|---|---|---|
| P_NoComm | 10 | 62 | 64 | 67 |
| P_Broadcast | 34 | 65 | 64 | 66 |
| R_NoComm | 69 | 73 | 73 | 79 |
| R_Broadcast | 70 | 73 | 73 | 76 |


The requested 10,34,62,67 and 66 counts are reproduced. Applying the same repaired rule to Broadcast gives 65, which is the essential additional count. The earlier 62 versus 34 comparison changes both communication and readout and cannot show communication harm.

## What the original aggregator actually does

It lowercases, removes ASCII punctuation and English articles, pools identical normalized answers, chooses the largest count, and returns the first raw answer in the ordered response list with that count. Despite its name, this is plurality: it does not require more than half the agents to agree. In a three-way tie, node 0 wins. There is no fresh tie lottery or answer-quality check. The code randomizes role assignment by question, and evidence partitioning separately; the final vote itself does not shuffle response positions. Cached request node metadata was checked against all 18,000 final response references.

Abstention-aware voting removes normalized `noanswer` and explicitly marked provider-refusal empty responses, then applies exactly the same plurality and original node-order tie. All-abstaining cases return `noanswer`. Whitespace-only or empty unmarked answers fail validation; nonempty answers with a refusal flag also fail. Synonyms such as unknown are not retrospectively recoded. Literal A, yes and no remain substantive. The lexical sensitivity baseline keeps the same filter/counts but chooses the lexicographically smallest normalized answer in a tied maximum. Its normalized winner is invariant to node permutation, though the first raw alias is used for rendering. This arbitrary but reproducible tie preference is a sensitivity analysis, not a fitted or optimal rule.

## Why exclusion changes the result



| Arm | Changed normalized strings | Abstention overrides | Plurality overrides | Node-tie overrides | Only abstention marker changes | EM gains | EM losses |
|---|---|---|---|---|---|---|---|
| P_NoComm | 114 | 114 | 109 | 5 | 0 | 52 | 0 |
| P_Broadcast | 76 | 74 | 72 | 2 | 2 | 31 | 0 |
| R_NoComm | 13 | 11 | 11 | 0 | 2 | 4 | 0 |
| R_Broadcast | 15 | 12 | 12 | 0 | 3 | 3 | 0 |


Partitioned NoComm changes 114 endpoints because an abstention defeated an available substantive answer:109 plurality overrides and 5 node-order ties. Only 52 new selections are exactly correct; excluding abstentions does not turn every substantive candidate into truth. Partitioned Broadcast has 74 such overrides plus 2 all-abstaining marker standardizations. Those latter changes convert an empty refusal winner to `noanswer`; both remain incorrect, not new substantive proposals.

If every abstention scores 0, deleting only abstentions while preserving the same substantive plurality/tie rule cannot lower EM: an already substantive winner stays unchanged, and only a zero-score abstention can be replaced. This is a structural property under the stated scoring contract, not a new general theorem about reliable refusal. It explains why a large readout gain alone is weak novelty evidence.

## Empty responses, refusals and reference validity

Among 7,634 distinct saved request keys, there are 68 explicit refusals, exactly 68 empty answers,4,676 normalized noanswer responses and 0 whitespace-only nonempty answers. They are retained with their provider markers and raw artifacts. Repeated appearances in different arms are not independent refusals.

All 300 selected references are nonempty, have nonempty official normalization and are not noanswer. Across the local 7,405-question official dataset, no reference normalizes to noanswer, but two nonempty official reference strings normalize to empty: The The (`5ae4988655429970de88d9c1`) and !!! (`5a8b3d795542997f31a41cc1`). They are outside this E5 cohort. This official normalization quirk is distinct from classifying a raw answer as an abstention. The implementation does not classify a nonempty answer as abstaining merely because normalization is empty. A future evaluation must lock handling of such references before outcomes are inspected.

The rule is reasonable for this answerable benchmark's answer-only accuracy and is gold-free at decision time. It is not a universally safe rejection policy. A context can be genuinely insufficient even when the benchmark has an answer; ignoring a justified local abstention may select an unsupported answer. On a task where noanswer is a legitimate correct response, the rule can lose accuracy, as explicitly tested. Report source support, answer coverage and rejection objectives alongside EM; do not silently redefine unanswerable requests.

Graph Verify preflight: project projectRLMAIN, generation 2026-10-13T17:28:45Z, fast/ready; E5 source paths and new modules are untracked. Material functions and raw sources were therefore inspected directly. Empty graph results are not used as absence/completeness evidence. This is a separate lead-agent implementation, not an external independent laboratory replication.
