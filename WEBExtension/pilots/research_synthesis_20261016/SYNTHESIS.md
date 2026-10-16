# Evidence-checked synthesis: evidence access, readout, and communication control

The proposed shared-context / partitioned-evidence organization is useful, with one material restriction: **the inspected P4/P5 streams cannot currently be presented as verified publisher GSM8K results**. Their recorded values are reproducible, but their question provenance fails the local publisher-split comparison below. Keep them as archival diagnostics until lineage is resolved. P1–P3 were not reaudited here and should not inherit verification from P4/P5 or the newer Web runs.

No model calls, downloads, retraining, or changes to historical outputs were made for this synthesis. [Machine-readable checks and source hashes](claim_checks.json) accompany the text.

## Shared evidence: distinguish a redundant route from topology invariance

The inspected P4 sweep does not establish topology invariance. Across 15 two-edge supports, recorded accuracy ranges from **73.33% to 80.00%** at nominal volume 50, and **75.33% to 82.67%** at nominal volume 100, on the same 150 question strings. These differences are descriptive; no new paired topology test was performed. S₁ is not the best observed support in that sweep. A controller repeatedly choosing S₁ therefore does not establish that S₁ is globally optimal.

P5 has **100 questions**, not 300 independent examples. Its fixed-depth learned controller chooses S₁ on **300/300 decisions = 100 questions × three rounds**, with 91/100 correct, exactly matching its static-routing cell. Adaptive cells have 93/100 correct and reduce the recorded mean communication counter from 46 to 14, **69.565%**. The raw discordances are three wins and one loss; the stored paired interval is **[-2, +5] percentage points**, with exact McNemar **p=0.625**. This supports an observed efficiency difference in the archival run; it does not establish an accuracy improvement or a noninferiority claim.

These counts are **communication proxies**. [`utils.count_tokens`](../../../utils.py) returns `max(1, int(1.3 * len(text.split())))` for nonempty text. The 46→14 result is not a measured model-token or billing reduction. Bandwidth rationing and adaptive halting also require separate contrasts; P5's stopping comparison alone does not identify an independent content-selection benefit.

The cleaner shared-context Web evidence supports the narrower statement that these comparisons do not establish a communication advantage:

- E2 shared clockwise minus counterclockwise: **-0.1065 answer F1 points**, stored 95% interval **[-1.4761, +1.1473]**, randomization **p=0.8801**. This is inconclusive about a small topology effect, not proof of invariance.
- E5 shared abstention-aware NoComm and Broadcast both have **73/300 correct**, but Broadcast gains five questions and loses five. Mean F1 is **33.9140 versus 33.2732**, respectively. Broadcast-minus-NoComm EM is 0 pp, bootstrap interval **[-2, +2]**; F1 is **-0.6408 pp**, interval **[-2.4370, +1.0465]**. Equal aggregate EM does not mean identical outputs or formal equivalence.

Shared R refers to the same retrieved candidate pool at every agent. It does not mean every pool contains all necessary gold evidence.

## Partitioned Web synthesis: centralization is a required baseline

In the exploratory 2Wiki pilot, full-union single-agent inference has **8 correct out of 20 assigned**, versus **6** for complete In-Star and Dense. NoComm has **1**, while the one-sender heuristic and random routing each have **4**. Nineteen records completed; one question remains unfinished after a timeout. Invalid and missing outputs count as operational failures in these counts; official answer scores remain incomplete.

Full union uses **70,439** known logical model-token references versus Dense's **263,351**, a **73.2528% reduction**, across the **19 completed question records**. This is a comparison of logical input/output references, not paid incremental cost; shared calls overlap across arms. Accuracy and cost denominators must remain explicit. In-Star and Dense share the same one-round sink request in this pilot, so their equal sink outcomes are structural.

This supports retaining a centralized, full-evidence single-agent baseline whenever the complete evidence fits the allowed prompt. It does **not** establish that distributed isolation is the only possible justification for multi-agent systems. A claim about genuinely isolated deployment would require that constraint to be part of the actual study design; these experiments simulate partitioning. The NoComm→In-Star improvement is observed under that partitioned design, not a result that outweighs the centralized control.

## Readout repair changes the interpretation of communication gains

On the same 300 HotpotQA identities in E5 P, a common abstention-aware rule changes NoComm from **10/300 to 62/300** correct and answer F1 from **5.0598 to 28.8823**. Broadcast changes from **34/300 to 65/300**, with answer F1 **14.9920 to 30.1132**. Under matched readout, Broadcast's difference is **+1.00 pp EM**, bootstrap 95% interval **[-1.3333, +3.3333]**, exact McNemar **p=0.5811**. The F1 contrast is +1.2309 pp, interval [-0.8018, +3.3890].

Thus original voting mixes changes in answer supply, revision, and selection; it is not an isolated measure of information synthesis. The reaggregation is post hoc and exploratory. It establishes a large observed readout bottleneck, not that communication never helps or that evidence starvation has disappeared.

E1 is a corroborating analysis on the **same 300 IDs**, with a different initial-response namespace: partitioned NoComm rises **11→62**, clockwise **20→64**, and counterclockwise **19→62** under abstention-aware voting. E1 contains rings, not a separately saved In-Star/Broadcast comparison. E1 and E5 must not be pooled as 600 independent questions.

## One-round fixed-sink sufficiency has a precise scope

With a synchronous single round, fixed Hub 1, unchanged initial states, identical ordered incoming messages, and matched cap 256, deleting Broadcast's four edges that do not enter Hub 1 preserves its request and allowed-source set. The offline audit reconstructs identical cache keys for **300/300 P and 300/300 R** cases. Its reused Hub outputs are therefore exactly the saved Broadcast outputs: **37/300 correct, F1 17.3639 in P**, and **69/300 correct, F1 31.7159 in R**.

This is structural request preservation, not a separate stochastic experiment or a learned-policy superiority result. The actually saved cap-128 seed-31 policy chooses In-Star on all 300 P questions, but only **291/300 requests (97.0%)** match Broadcast. Its Hub score is **36/300**, not 37/300. The 97% figure applies to P; in R, seed 31 chooses In-Star on only 59 questions, with 47 request matches within that subset.

For the cap-256 **partitioned projection**, directed deliveries fall **1,800→600**, recipient update calls **900→300**, and payload tokens **39,862→12,830**: reductions of **66.7%, 66.7%, and 67.81%**, respectively. Including the 900 shared initial calls, total logical calls fall **1,800→1,200**, a **33.3%** reduction. Missing model-usage fields prevent an exact all-in model-token saving claim. Payload reduction is cohort-specific; it is not always equal to the edge reduction.

The two saved 2Wiki failures show why a single sender can be insufficient when the sink lacks complementary remote facts. They motivate testing a k∈{0,1,2} router that selects evidence covering unresolved entity/relation needs. They do not prove that every one-sender route must fail on every two-hop query. No such revised router or N=100 study has been evaluated here.

## P4/P5 dataset-provenance gate

The local Hugging Face cache identifies GSM8K `main`, with **7,473 train** and **1,319 test** questions. Comparing exact question strings after whitespace normalization gives:

- P4 reference stream: **0/150** match train; **0/150** match test.
- P5 fixed learned stream: **0/100** match train; **0/100** match test. Its questions are exactly the first 100 of the P4 reference stream.

The first archived question is “Bob has 6 apples. Grace gives Bob 10 more apples. How many apples does Bob have now?” The source scripts label these runs GSM8K, but those labels are not a publisher-data lineage record. The current loader fails closed on download failure; that does not retroactively authenticate older streams. This audit establishes a mismatch against the inspected cached splits, not the exact historical data-generation route or a claim about every P1–P5 artifact. Preserve the streams and resolve their lineage before reusing their percentages as GSM8K benchmark evidence.

## Paper-ready core narrative

Our experiments separate evidence access, answer readout, and communication cost. When the full evidence pool fits a single prompt, centralized inference is a necessary baseline: in an exploratory 2Wiki pilot it produced eight correct answers among twenty assigned questions, compared with six for partitioned complete In-Star and dense communication, while using 73.3% fewer known logical model-token references across nineteen completed records. This comparison limits the practical case for partitioning in that setting; it does not rule out multi-agent benefits under other deployment constraints.

On the existing 300-question HotpotQA cohort, readout choice accounts for much of the apparent communication gain. Removing abstentions before voting increases partitioned NoComm from 10 to 62 exact matches. Under the same readout, Broadcast reaches 65, leaving an uncertain one-percentage-point advantage. Shared-context comparisons likewise do not establish a routing benefit. We therefore treat readout repair as a required baseline rather than attributing all observed gains to evidence synthesis.

A narrower communication result follows directly from the synchronous one-round implementation. With a fixed sink and matched message cap, an In-Star retains every incoming message used by Broadcast at that sink. Exact request reconstruction verifies this property on all 300 questions in each evidence regime, allowing reuse of the saved sink outputs with two rather than six directed deliveries. This result establishes conditional sink sufficiency, not a general learned-routing advantage or a guarantee across further rounds. The one-sender 2Wiki failures motivate complementary-sender selection as a future test. Historical P4/P5 halting and bandwidth observations remain archival pending resolution of their dataset provenance and token-counter limitations.

## Evidence and reproduction

- [Readout audit, including matched intervals and request identities](../hotpot_readout_20261016/REPORT.md).
- [2Wiki pilot limitations and failure examples](../2wiki_20261016/FINDINGS.md).
- [Original P4 report, preserved](../../../results/p4_definitive_suite/p4_definitive_report.md).
- [Original P5 report, preserved; see the qualifications above](../../../results/p5_communication_control/p5_master_report.md).
- [E2 and other historical paired contrasts](../../analysis/final_findings_20261014/paired_effects.json).
- [Recomputed claim register and consumed-file hashes](claim_checks.json).

Run `python -m WEBExtension.pilots.research_synthesis_20261016.audit_claims` from the repository root. It reads the local cached dataset, validates raw question alignment and recorded contrasts, and leaves source artifacts unchanged. No API credentials or model client are loaded.
