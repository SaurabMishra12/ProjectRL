# Evidence audit informing the new design

Date: 2026-10-16. This is a bounded audit of the paths below, not an exhaustive audit of every historical run.

## 1. HotpotQA readout must be matched

The current [readout reassessment](../causal_study/readout_reassessment/07_FINAL_SCIENTIFIC_DECISION.md) compares the same abstention-aware rule on both endpoints:

- Partitioned NoComm: **62/300 EM**; Broadcast: **65/300 EM**.
- Paired difference: +1.000 EM point, saved 95% CI [-1.333,+3.333], exact McNemar p=0.5811.
- F1 difference: +1.231 points, saved CI [-0.802,+3.389].
- Shared evidence: both abstention-aware endpoints have 73/300 EM.

The counts were independently recomputed in this session using `independent_counts()` in [verify.py](../causal_study/readout_reassessment/verify.py), which checks its reconstruction against the saved independent count artifact. It returned original/abstention-aware counts of 10/62 for partitioned NoComm and 34/65 for partitioned Broadcast. The intervals and tests above were read from the existing reassessment; they were not recomputed here.

Consequently, comparing improved NoComm (62) with original-readout Broadcast (34) does not isolate communication. Excluding abstentions fixes a weak benchmark readout, but a large routing benefit is unestablished. It is also not a safe universal answerability rule for genuinely unanswerable requests.

## 2. Annotated-support absence is a proxy

The historical 224/300 missing-annotated-support statistic is not proof that those questions are impossible from the remaining sources or model knowledge. Supporting labels describe annotated paths, not all semantically sufficient paths. Support-present subgroup contrasts likewise do not establish a population communication effect without matched readout and inference resources.

The [existing E9 protocol](../evidence_value_study/reports/03_E9_CAUSAL_PROTOCOL.md) explicitly separates source access, semantic necessity, actual packet delivery, and supported correctness. This new benchmark proposal is separate from that study and grants no authority to bypass its qualification or execution gates.

## 3. Historical scaling is useful, but not a scaling-law proof

The inspected [summary](../../08-06results/scaling_study_agents.json) contains GSM8K SparseComm records for N=2,4,5, each reporting 150 samples:

- N=2: 98.0% accuracy; 1,399.97 legacy total tokens/question; 13.38 communication tokens/question.
- N=4: 98.67%; 3,642.44 total; 157.24 communication.
- N=5: 98.0%; 4,539.46 total; 246.17 communication.

[generate_scaling_plot.py](../../08-06results/generate_scaling_plot.py) supplies N=3 from the Transfer_GSM8K benchmark and explicitly calls its 1,680 total tokens an estimate. Its four plotted points are therefore not yet a verified common-cohort, common-protocol curve. No such equivalence is assumed here.

[Scale_Experiment.py](../../Scale_Experiment.py) has two further relevant properties:

- `IBController.__init__` uses input dimension `5 + N + N*N + 3` (line 2134); `build_features` flattens `prev_W` (line 2163); `_split_action` materializes `H_src @ H_dst.T` (line 2177). Rank two reduces the edge-output parameterization, but this implementation still uses quadratic state and pairwise computation.
- `compute_token_cost` (line 296) adds generated tokens and transmitted tokens. This is not a complete provider input/output billing ledger: local evidence, repeated source context, instructions, and cached-input treatment are not established by that number.

The named `run_scaling_and_ablation_study` (line 3733) varies training sample counts 100/200/300/400. Those N labels must not be confused with team size. The evidence inspected does not authenticate every legacy raw question record, checkpoint, or dense counterpart.

## 4. Existing Web components to reuse conceptually

Direct source reads found:

- [protocol.py](../protocol.py): three-agent assignment; concise answer/citation prompt; exact tokenizer; source reconstruction; prefix/crop payloads; original majority vote. The old schema does not preserve useful partial facts when an agent returns `noanswer` without citations.
- [experiments.py](../experiments.py): only unique recipients receive update calls; delivered complete source lines determine permissible new citations. This is a useful visibility boundary to retain.
- [learned_controller.py](../learned_controller.py): a three-agent categorical action space and observable feature allowlist. It is distinct from the legacy rank-two controller. Neither existing interface is a drop-in arbitrary-N controller.
- [controller_study.py](../controller_study.py): call and payload penalties, and recipient-based cost accounting. Equal edges need not mean equal recipient calls; preserve the distinction in new controls.

These are targeted source findings, not an assertion that every possible implementation issue has been found. Existing policies trained under old prompts/readout cannot be declared validated under the proposed new contract.

## 5. Benchmark facts checked against primary sources

[2WikiMultihopQA official repository](https://github.com/Alab-NII/2wikimultihop) provides context, support annotations, reasoning triples, and alias-aware evaluation updates. Use context as inference evidence; keep support labels, triples, gold entity identifiers, and aliases in the evaluator. Pin the compatible corrected release and evaluator rather than mixing dataset versions. The [official v1.1 evaluator](https://github.com/Alab-NII/2wikimultihop/blob/main/2wikimultihop_evaluate_v1.1.py) still uses answer EM/token F1; it does not eliminate phrasing effects.

[FinQA official repository](https://github.com/czyssrs/FinQA) distributes financial-report text, tables, questions, and program/reference fields. Its [evaluation implementation](https://github.com/czyssrs/FinQA/blob/main/code/evaluate/evaluate.py) scores executable programs. This is not a raw-HTML browsing benchmark. The official repository documents past retrieval-format leakage fixes, so do not use gold retrieval fields or label-dependent table rendering. See also the [FinQA paper](https://aclanthology.org/2021.emnlp-main.300/) and [2Wiki paper](https://aclanthology.org/2020.coling-main.580/).

Full supplied context is a source-availability condition to verify locally. It is not a guarantee that every model will find the answer, that every reference is correct, or that communication will help. Neither benchmark is live Web search.

## 6. Graph coverage and provenance

Tier 2 graph discovery used project `projectRLMAIN`, generation **2026-10-13T17:28:45Z**, fast index mode. `Scale_Experiment.py` had metadata-matching coverage and no recorded parse gap. A narrowed function search returned all 47 matching function entries; targeted traces of the scaling entry point and sample loop were inspected.

For the cited `WEBExtension` sources/reports, coverage returned freshness **not_tracked**, despite no recorded issue. The new Web symbols were absent from graph searches; direct source reads and targeted `rg` were used. That absence was not treated as proof that the files or functions do not exist. Coverage is a best-effort signal, not a completeness guarantee.

The companion source manifest records hashes of the local artifacts used. Historical evidence remains unchanged. Any future result must have its own frozen input, prompt, code, model, and attempt provenance.
