# SparseComm saved-results audit and publication readiness

Audit snapshot: 2026-10-14T14:16:19.607263+00:00. All analysis uses saved authentic artifacts. Automatic recovery is paused; no experiment services are loaded. This analysis issued **zero inference calls**.

**Verdict: not ready for a strong WWW 2027 research submission as currently framed.** E1–E4 are complete and internally authenticated. E5 has only 129/300 completed questions; no missing results were filled. A null primary result alone does not make research unpublishable. Here the incomplete test, degenerate manipulation and generalizability gaps jointly block readiness. There is useful empirical evidence about evidence availability, but the main interaction remains inconclusive, the content contrast mostly did not change messages, and learned-policy claims need a complete test and careful resource/adaptivity limits.

## 1. Evidence scope and independent verification

- E1–E4 each have exactly 300 unique frozen question IDs. Their saved SHA256 values match completion manifests.
- E5 has exactly 129 unique frozen test IDs, with all 20 required arms per saved question. It is an interrupted prefix, not a completed test.
- All recorded answer F1/EM values in E1–E5 match the provided upstream HotpotQA reference scorer: **zero mismatches**. The reference file is copied and hashed in `sources/`. Supporting-label and joint-label diagnostics were newly derived from the saved ensemble/citation unions and are NOT sentence-entailment evaluations or a leaderboard submission.
- Every unique recorded payload was recounted with the pinned GPT-OSS tokenizer: **zero token-count mismatches**.
- Frozen scientific source hashes and all three selected checkpoint hashes match. No training, data resampling or model invocation was performed during this audit.
- Graph discovery has no current tracked coverage for WEBExtension (generation 2026-10-13T17:28:45Z); direct-source checks support this bounded audit. Index cleanliness is not asserted as proof of completeness.

## 2. Complete experiments: quality and resources

F1 and EM are on a 0–100 scale. R means redundant evidence; P means exclusive partitioned evidence. Logical calls include shared initial inference; they must not be summed across overlapping arms as physical charges.

| Experiment | Condition | n | Mean F1 | EM | Payload tokens/question | Logical calls/question |
|---|---|---:|---:|---:|---:|---:|
| E1 | P_CCW | 300 | 9.040 | 6.333 | 66.800 | 6.000 |
| E1 | P_CW | 300 | 9.133 | 6.667 | 66.800 | 6.000 |
| E1 | P_NoComm | 300 | 5.267 | 3.667 | 0.000 | 3.000 |
| E1 | P_SelfUpdate | 300 | 5.756 | 3.667 | 0.000 | 6.000 |
| E1 | R_CCW | 300 | 32.839 | 24.000 | 145.163 | 6.000 |
| E1 | R_CW | 300 | 32.732 | 24.333 | 145.163 | 6.000 |
| E1 | R_NoComm | 300 | 31.748 | 22.667 | 0.000 | 3.000 |
| E1 | R_SelfUpdate | 300 | 32.089 | 23.667 | 0.000 | 6.000 |
| E2 | R_CCW | 300 | 32.839 | 24.000 | 145.163 | 6.000 |
| E2 | R_CW | 300 | 32.732 | 24.333 | 145.163 | 6.000 |
| E2 | R_NoComm | 300 | 31.748 | 22.667 | 0.000 | 3.000 |
| E3 | R_crop | 300 | 31.490 | 23.333 | 47.277 | 4.000 |
| E3 | R_prefix | 300 | 31.635 | 23.333 | 47.277 | 4.000 |
| E4 | P_final | 300 | 10.772 | 7.667 | 216.310 | 12.000 |
| E4 | P_stopped | 300 | 10.772 | 7.667 | 200.627 | 11.820 |

## 3. Primary and secondary inference

The frozen primary contrast `(CW−NoComm)_P − (CW−NoComm)_R` is **2.882 F1 points**, 95% question-bootstrap CI **[-0.260, 6.011]**, paired sign-randomization **p=0.0713**. The interval includes zero. Do not describe the primary moderator hypothesis as confirmed or replace it with a more favorable subgroup after observing results.

| Contrast | Effect (F1 points) | 95% CI | Raw p | Original Holm p |
|---|---:|---|---:|---:|
| E1 (CW-NoComm)_P - (CW-NoComm)_R | 2.882 | [-0.260, 6.011] | 0.0713 | unknown |
| E1 R_CW - R_NoComm | 0.985 | [-0.768, 2.893] | 0.2955 | 0.858 |
| E1 P_CW - P_NoComm | 3.866 | [1.356, 6.486] | 0.0015 | 0.009 |
| E1 R_CCW - R_NoComm | 1.091 | [-0.801, 3.144] | 0.2860 | 0.858 |
| E1 P_CCW - P_NoComm | 3.773 | [1.421, 6.354] | 0.0024 | 0.012 |
| E1 R_CW - R_SelfUpdate | 0.644 | [-0.871, 2.215] | 0.4210 | 0.858 |
| E1 P_CW - P_SelfUpdate | 3.377 | [0.849, 6.053] | 0.0110 | 0.044 |
| E2 R_CW - R_CCW | -0.107 | [-1.476, 1.147] | 0.8801 | 0.880 |
| E3 R_prefix - R_crop | 0.145 | [-0.136, 0.571] | 1.0000 | 1 |
| E4 P_stopped - P_final | 0.000 | [0.000, 0.000] | 1.0000 | 1 |

Partitioned CW improves on NoComm and SelfUpdate in secondary contrasts. This supports a conditional observed gain within P; it does not prove that P improves communication gains more than R. E2 routing and E3 content estimates do not establish equivalence. Added `holm_across_completed_secondary_contrasts` values are a separately labeled multiplicity sensitivity; they do not silently replace the original analysis families.

## 4. Gold-location strata and retrieval availability

**62 Disjoint-Gold, 14 Co-located-Gold, 224 Retrieval-Missing-Gold.** Both labelled gold articles are present in only 79/300 publisher candidate contexts, and in 76/300 top-6 pools. Thus most absent-gold cases originate in supplied fullwiki candidates, rather than being caused by the additional top-6 selection. These are real publisher fullwiki candidate contexts; the assertion that every question contains two gold articles plus eight distractors does not describe this sample.

A missing gold article does not prove that every possible answer cue is absent: backbone knowledge and non-gold passages may still help. The diagnostic specifically measures availability of annotated gold evidence.

| E1 exploratory stratum contrast | Stratum | n | Effect | 95% CI | Raw p | Report Holm p |
|---|---|---:|---:|---|---:|---:|
| P_CW - NoComm | Disjoint-Gold | 62 | 10.311 | [1.111, 20.167] | 0.0398 | 1.0000 |
| P_CW - NoComm | Co-located-Gold | 14 | 17.857 | [0.000, 39.286] | 0.2502 | 1.0000 |
| P_CW - NoComm | Retrieval-Missing-Gold | 224 | 1.208 | [-0.348, 3.022] | 0.1672 | 1.0000 |
| (CW-NoComm)_P - (CW-NoComm)_R | Disjoint-Gold | 62 | 12.200 | [2.681, 22.301] | 0.0169 | 0.4563 |
| (CW-NoComm)_P - (CW-NoComm)_R | Co-located-Gold | 14 | 10.317 | [-14.286, 34.921] | 0.6211 | 1.0000 |
| (CW-NoComm)_P - (CW-NoComm)_R | Retrieval-Missing-Gold | 224 | -0.163 | [-2.831, 2.451] | 0.9055 | 1.0000 |

The Disjoint-Gold point estimates are larger, but post-hoc subgroup inference has low effective n and does not pass the stored familywise adjustment. Co-location n=14 is particularly uncertain. Stratum membership is observational; its heterogeneity interval does not identify a randomized causal contrast between strata. The real 2/2/2 shard design has a 20% same-shard probability conditional on both gold articles being retained, not the 3/3/4 ten-document probability. Gold labels were used offline here, not to choose controller inputs/actions.

## 5. Manipulation checks and experimental overlap

- **E2:** CW/CCW have identical source payload amounts and calls, isolating one-step destination allocation in R. E2 reuses the same provider keys and scientific outcomes as E1 R; it is a mechanistic reanalysis, not another independent 300-question replication. Execution/cache metadata differs between full records.
- **E3:** 298/300 questions have identical prefix/crop payloads and receiver keys. Only two questions instantiate a content intervention. Exact budget matching passes, but this result cannot show content-insensitivity or compression equivalence. Do not drop the 298 identical questions and relabel the two changing cases as a new primary test.
- **E4:** stopping depths are {'3': 290, '2': 5, '0': 3, '1': 2}; only 10/300 stop before depth3. Six of these10 early consensuses are wrong. Continuing corrects none of those6 in this sample. All300 stopped/final F1 values coincide, but 0 answer strings differ. The total is54/3600 hypothetical calls avoided (**1.5%**) and 7.25% offline payload reduction. Full trajectories were actually generated, so these are replay counterfactuals, not realized monetary or wall-time savings. No justified noninferiority margin was predeclared.
- **Intentional overlap:** the same 300 test questions occur in E1–E4; E5 uses a129-question prefix of the same frozen test population. Pairing improves contrast efficiency. Do not count1200 or900 independent questions. Reusing training/validation questions across epochs and seeds is intentional and disclosed.

## 6. Train/validation/test leakage audit

The active study has100 training,30 validation,300 heldout and12 pilot questions. All6 partition pairs have **zero ID overlap, zero normalized-question overlap, and zero normalized candidate-title overlap**. Within-partition IDs/questions are unique. The100/30 subsets were selected using fixed type-stratified ID hashing without inspecting predictions/scores. Heldout input and gold bytes match the original300/100 study. Type counts: train80 bridge/20 comparison; validation24/6; heldout240/60.

No overt gold/correctness/ID-hash feature is present in `observable_features`. Gold informs offline rewards and validation selection, which is appropriate; inference/actor features are whitelisted. The actual categorical contextual-bandit PPO controller is not an architecture-compatible continuation of the original continuous SparseComm controller and does not train multi-round stopping.

Zero measured overlap is not a guarantee against document aliases, semantic duplicates, older undisclosed evaluations or model pretraining contamination. Title filtering changes representativeness. Training and heldout candidate-retrieval distributions differ. Record these limits rather than claiming universally leakage-free evaluation.

## 7. E5 preliminary trend: explicitly incomplete

All three optimizer seeds completed three epochs, 200 R/P training contexts and60 validation contexts each epoch (1800 train rollouts,540 validation records; only100/30 unique questions). Validation communication utility selects checkpoints, not test F1. Selected epochs: seed7 epoch1, seed19 epoch3, seed31 epoch3.

**The following figures and numbers use only129/300 completed test questions. They cannot support final300-question or publication claims.** Missing rows were not assigned zero, imputed or fabricated. `E5_partial_missing_data_bounds.csv` gives deliberately broad0–100 worst/best-case bounds for the unobserved171 questions.

| Partial contrast, seeds averaged within question | Effect | 95% exploratory CI | n |
|---|---:|---|---:|
| R_seed_mean_Learned - NoComm | -0.543 | [-2.631, 1.470] | 129 |
| R_seed_mean_Learned - SelfUpdate | 0.281 | [-2.117, 2.703] | 129 |
| R_seed_mean_Learned - Static | 0.322 | [-2.085, 2.860] | 129 |
| R_seed_mean_Learned - Broadcast | 0.220 | [-1.699, 2.148] | 129 |
| R_seed_mean_Learned - MatchedRandom | 0.005 | [-1.243, 1.255] | 129 |
| P_seed_mean_Learned - NoComm | 7.836 | [4.439, 11.755] | 129 |
| P_seed_mean_Learned - SelfUpdate | 6.470 | [2.159, 10.799] | 129 |
| P_seed_mean_Learned - Static | 6.479 | [2.707, 10.214] | 129 |
| P_seed_mean_Learned - Broadcast | -5.335 | [-8.337, -2.560] | 129 |
| P_seed_mean_Learned - MatchedRandom | 2.183 | [0.828, 3.918] | 129 |

On this prefix, P learned policies improve over NoComm but their seed mean underperforms Broadcast; R seed-mean quality is slightly below NoComm while using more calls/payload. This is a resource/quality tradeoff, not evidence of blanket dominance. Seed7 P is almost Broadcast; seed31 P chooses a single action on129/129 questions. Do not claim robust task-adaptive routing from three seeds alone.

**MatchedRandom caveat:** all774 learned/random comparisons match declared edge count andquota, but397 differ in logical recipient calls and114 differ in actual transmitted tokens. A fan-in may update one recipient while another two-edge graph updates two. This baseline is not exact recipient-call/token matched and cannot isolate a pure routing effect.

E5 response namespace differs from E1. Same-question baseline answer differences are recorded in `cross_suite_same_question_baselines.csv`; do not pool suites as independent replications or assume identical teacher draws. Controller optimizer seeds share real cached teacher responses where prompts match and are not independent model backbones.

## 8. Physical cost, refusals, timing and unknowns

The original cost reporting function suppresses both input/output totals if any successful output count is missing. This audit preserves the original reports and correctly reports dimensions independently. Known sums are not fabricated complete bills.

| Active pipeline | Returned successful responses | Known input tokens | Output lower bound | Successful responses missing output usage | Original unresolved attempt liabilities |
|---|---:|---:|---:|---:|---:|
| controlled | 6134 | 5,960,203 | 1,582,163 | 29 | 8 |
| controller | 5725 | 5,748,101 | 1,511,822 | 27 | 4 |

Across all preserved WEBExtension ledgers—including archived caps, pilot and prior study—de-duplication by original attempt ID gives 13,666 attempts: {'success': 13632, 'rejected': 21, 'uncertain': 13}. Reported input-token lower bound **13,333,177**, output-token lower bound **3,535,121**. There are 34 missing input-usage records and 91 missing output-usage records. This global scope is not just the active primary protocol and must not be confused with logical per-arm costs.

Refusals remain explicit abstentions with original provider bodies and reported usage; unavailable usage remains unknown. These quality results describe the deployed endpoint including its safety-filter policy, not unconstrained model capability. Call counts differ across some arms, so refusal opportunities may differ as well; the call-matched SelfUpdate control is especially relevant. Provider500/connection/timeouts may have been billed before a response reached disk. Retry authorization never marks those originals as ungenerated. Imported successful responses retain the original cost-owner ledger and are not charged again merely because a new study reads them.

No verified deployment price/billing export is available, so total currency cost and savings are **not calculable** from these artifacts. Do not substitute a public list price for this deployment without verifying billing terms.

Network timing quantiles and worker-session start/end durations are in separate tables. Queue/cache/end-to-end times and nested stage timers overlap; do not sum nested timers as independent wall time. Timings cover captured operational sessions, not unobserved server compute or lost historical spans. No clean arm-randomized end-to-end runtime benchmark exists here. Automatic recovery improved throughput but is not an experimental treatment latency effect.

## 9. Publication readiness and actionable priorities

1. **Complete the frozen E5 heldout evaluation** before reporting a learned-controller result. Keep selected checkpoints unchanged; do not tune after looking at the partial test results.
2. **Repair/reframe the content question.** The current E3 has only2 actual message changes. Preserve it as a failed/degenerate manipulation and, if new experiments are authorized, preregister a separate meaningful content intervention on new authentic data with sufficiently long messages and exact budgets. Do not overwrite the current result.
3. **Make evidence-availability scope explicit.** Only76/300 selected pools contain both gold articles. A fresh authentic distractor/full-availability protocol can test complementary transfer more directly; this requires a separate frozen dataset/run, not gold-aware post-hoc repair of the current primary sample.
4. **Strengthen generalizability.** One model deployment, one benchmark and one primary evidence-partition seed do not establish robustness. Three optimizer seeds do not substitute for independent partition seeds, backbone models or benchmarks. Additional runs are not performed by this audit.
5. **Match resources for causal topology claims.** Either use recipient/call/payload-matched controls or state that E5 compares whole policies with measured resource differences. Keep E2 restricted to one-step destination allocation.
6. **Report known costs and unknown liabilities separately**, publish the correction alongside the original incomplete cost summaries, and verify entitlement/billing before monetary claims.
7. **Align the paper to the actual implementation.** The locally found `PaperOctVersion/Paper6Oct.tex` describes the prior closed-book/continuous SparseComm work; no verified current WWW manuscript tied to these300-question artifacts was identified. The June2026 `SparseComm_Research_Paper.md` also describes continuous edge weights, semantic masking and theoretical/topological claims outside this new categorical-bandit study. Its earlier accuracy/saving claims were not validated by this Web audit and cannot be transferred to these results. Existing dossier headings asserting “100% verified” are claims in those documents, not independent evidence. A submission-ready prose/novelty/page-limit/citation audit of the final manuscript remains outstanding.
8. **Novelty is not certified.** The current blueprint positions an empirical evidence-availability paper, not a state-of-the-art RL router. A full current literature comparison and proof that the Web-specific research challenge is substantive remain necessary.

The official [WWW2027 research-track call](https://www2027.thewebconf.org/research-track-papers/) requires explicit Web/track relevance on the first page and warns that merely using a Web dataset/API is insufficient. Candidate-based multi-hop Web evidence delivery is a plausible research focus, but the abstract must state the scientific challenge, the tested estimand and the actual uncertainty. Listed deadlines were18 October2026 for abstracts and25 October2026 for full papers (AoE), checked during this audit. Acceptance remains a peer-review decision, not something these software checks can guarantee.

## 10. Claim-support map

| Proposed claim | Evidence judgment |
|---|---|
| Communication has a confirmed larger benefit with partitioned evidence | Not confirmed by the pooled primary interaction; report estimate andCI. |
| Communication helps the Disjoint-Gold cases massively | Larger exploratory point estimate, low n, multiplicity adjustment not passed; avoid “massive”. |
| Prefix/crop are equivalent or content does not matter | Unsupported: 298/300 payloads identical; only2 informative pairs. |
| Adaptive stopping preserves correctness and saves real money/time | Unsupported generally; exact observed F1 tie, six wrong early consensuses, small hypothetical savings only. |
| Learned SparseComm robustly outperforms baselines | Incomplete test; partial policies have broadcast tradeoffs and static behavior. |
| MatchedRandom isolates routing at identical resource costs | False for actual calls/tokens; only edgecount/quota matched. |
| Training/testing have no measured overlaps | Supported for exact ID/question/title tests; semantic/pretraining contamination unverified. |
| Results are authentic, internally consistent and reproducible from saved artifacts | Strong internal provenance/scoring/hash evidence; immutable provider snapshot and full external rerun remain unavailable. |

## 11. Available metrics, plots and reproducibility

All recorded scalar quality, grounding-label, transmission, invocation, refusal, cache and utility metrics are in `tables/all_condition_question_metrics.csv`; their per-arm summaries, missingness and full distribution profiles are supplied. Official-style supporting/joint-label metrics are explicitly diagnostic. Paired F1/EM transition records, stratum effects, learning curves, action entropy/dominance, resource matching, source retrieval counts, physical attempt provenance, latency quantiles and missing-test bounds are supplied separately.

The figures include13 core research panels, optimizer diagnostics and individual panels for every additional recorded scalar condition metric. All figures are exported as PNG and vector PDF, with constrained layouts and separate legends. Some exact data points naturally coincide; plotting does not add artificial jitter or change the values. No title/legend/axis overlap was observed in the reviewed core renders.

Run the two saved analysis scripts offline using `HF_HUB_OFFLINE=1`; they do not instantiate a model provider. The companion notebook opens the already-calculated metrics and figures by default. Recalculation is opt-in and writes only derived audit files. Keep the source result/checkpoint artifacts intact. `analysis_artifact_manifest.json` hashes inputs and every derived file.

## 12. Adversarial self-review

- **Contribution:** There is a defensible empirical audit of evidence availability and delivery, but superiority/novelty is not established.
- **Clarity:** Distinguish answerF1 from supporting-label overlap, P fromR, complete tests frompartial prefixes, and actual bills fromlogical/counterfactual costs.
- **Experimental strength:** Complete paired E1–E4 are internally verified; E3 manipulation degeneracy and smallgold strata are substantial limitations.
- **Completeness:** E5 and robustness replication remain incomplete; no final controller claim or publication-readiness certification is justified.
- **Method soundness:** No overt gold-feature leakage and exact split checks pass; controller scope, static degeneration, namespace reuse and imperfect matched controls require explicit disclosure.

## 13. Verified external references

- [Official HotpotQA evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py): answerF1/EM and supporting/joint metric definitions. Local reference hash is recorded in the audit manifest.
- [HotpotQA dataset and official setting descriptions](https://hotpotqa.github.io/): distinguish candidate evaluation from full Wikipedia/live browsing.
- [WWW2027 research-track requirements](https://www2027.thewebconf.org/research-track-papers/): scope/relevance/deadlines, consulted14 October2026.

The supplied blueprint and pasted strategy are treated as research plans and claims to test, not as evidence that experiments ran or permission to fabricate favorable outcomes.
