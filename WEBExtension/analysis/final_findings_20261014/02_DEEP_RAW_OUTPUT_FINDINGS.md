# 2. Findings from the complete raw outputs

Snapshot generated 2026-10-14T17:51:19.720260+00:00. Scope: completed frozen WEBExtension E1–E5, controller training/validation/pilot, operational cost evidence, and the supplied 13-page manuscript. Each experiment uses the same 300 held-out question identities; the unit of inference is the question, not the arm, agent, seed, or repeated epoch. All analysis is offline; no inference calls were issued. Earlier raw artifacts and the 129-question audit are preserved. Display tables round to six significant digits; CSV/JSON files retain machine-readable precision. NA is unknown or inapplicable, never an invented zero.

Run identifiers: `run_f6aa61bac2c3dfae`, `controller_study_a0b3768357484b45`, split `controller_splits_2e025f8ae3d9815e`. Input hashes and final artifact hashes are in `final_manifest.json`.

## Main finding: communication usefulness and sparsity are conditional

In E5, the R learned seed mean has F1 32.1084, compared with 32.3140 for NoComm and 31.6510 for Broadcast. It sends 60.05 payload tokens and uses 3.97778 calls per question, versus 293.293 tokens and 6 calls for Broadcast. This is a 79.5256% payload reduction and 33.7037% logical-call reduction, with an inconclusive F1 difference (+0.45735, CI −0.68787 to 1.60985). It does not establish noninferiority. NoComm already supplies similar quality at zero transmission and three calls; learned utility is significantly worse than NoComm (−0.02002, Holm p=0.0019998).

In P, the learned seed mean reaches F1 10.5604 versus NoComm 5.05979, SelfUpdate 6.21606 and Static 6.42086. The gains are +5.50062, +4.34435 and +4.13955 points, respectively, and pass the stored E5 Holm family. Broadcast reaches 14.9920: learned is worse by 4.43158 points (CI −6.31219 to −2.70728; Holm p=0.0019998), while transmitting 46.2780% fewer payload tokens and requiring 22.3519% fewer logical calls. That is a measured quality/resource tradeoff. Fewer transmitted tokens do not imply the same percent reduction in billed input/output tokens or latency.

E5 utility against NoComm in P is +0.01624 with CI spanning zero; improvement in F1 is therefore stronger evidence than improvement in the prespecified reward. Static means the validation-selected OUTSTAR from only three candidates, not the best of all 15 two-edge supports. A claim of superiority over all static topologies would exceed the experiment.

## The main moderator hypothesis remains inconclusive

E1's frozen primary difference-in-differences is +2.88151 F1 points, CI [−0.259961, 6.01089], p=0.0712929. Communication improves P in secondary comparisons, but the pooled experiment does not confirm that communication benefits P more than R. The distinction matters: a significant effect in one group and a nonsignificant effect in another is not itself a significant interaction. E5 is a different whole-policy comparison and cannot retrospectively replace the E1 primary endpoint.


![E1–E4 paired effects. Intervals and tests retain their original analysis families.](figures/02_paired_effects.png)

E1–E4 paired effects. Intervals and tests retain their original analysis families.

## Evidence availability is a major scope limitation

The fullwiki publisher candidates contain both annotated gold articles in 79/300 questions; BM25 top-six retains both in 76/300. Of the final pools, 62 are Disjoint-Gold, 14 Co-located-Gold and 224 Retrieval-Missing-Gold. Thus 74.6667% lack at least one labelled gold article after selection. Only three additional both-gold cases are lost by top-six selection. Most missing-gold cases are already missing in the supplied candidate set; calling all of this BM25 failure would be inaccurate.

For actual six-document 2/2/2 allocation, the chance that two retained distinct gold articles co-locate under uniform shuffling is 3×C(2,2)/C(6,2)=1/5. For hypothetical 3/3/4 allocation of ten documents it is [C(3,2)+C(3,2)+C(4,2)]/C(10,2)=12/45=26.6667%, rather than exactly 30%. The observed 14/76 co-location rate is descriptive. Fullwiki candidate retrieval is different from the guaranteed two-gold distractor setting.

Gold-missing does not mean no possible answer information exists: alternative passages or model memory can answer questions. Gold labels are used only for offline analysis/reward, not to choose deployment shards. The subgroup findings below are observational and post-hoc, with only 14 co-located cases. Larger point estimates do not establish a causal effect of belonging to a stratum.


![Availability of annotated gold articles before and after BM25 top-six selection.](figures/04_retrieval_audit.png)

Availability of annotated gold articles before and after BM25 top-six selection.
| regime | group_field | group | baseline | n | learned_F1 | baseline_F1 | effect | ci95_low | ci95_high | helped | harmed | ties | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R | stratum | Co-located-Gold | NoComm | 14 | 78.8117 | 83.5441 | -4.73242 | -14.2857 | 0.0884564 | 1 | 1 | 12 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | stratum | Co-located-Gold | SelfUpdate | 14 | 78.8117 | 75.0376 | 3.77407 | -0.100385 | 9.48836 | 3 | 1 | 10 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | stratum | Co-located-Gold | Static | 14 | 78.8117 | 76.4012 | 2.41044 | 0 | 7.20183 | 2 | 0 | 12 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | stratum | Co-located-Gold | Broadcast | 14 | 78.8117 | 74.5758 | 4.23583 | 0.058971 | 9.92063 | 4 | 0 | 10 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | stratum | Disjoint-Gold | NoComm | 62 | 74.4675 | 73.21 | 1.25751 | -0.823919 | 3.60732 | 6 | 3 | 53 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | stratum | Disjoint-Gold | SelfUpdate | 62 | 74.4675 | 74.918 | -0.4505 | -4.07614 | 3.10861 | 5 | 6 | 51 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | stratum | Disjoint-Gold | Static | 62 | 74.4675 | 73.8282 | 0.639226 | -1.6847 | 3.65153 | 4 | 5 | 53 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | stratum | Disjoint-Gold | Broadcast | 62 | 74.4675 | 73.6984 | 0.769096 | -1.65028 | 3.92962 | 4 | 5 | 53 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | stratum | Retrieval-Missing-Gold | NoComm | 224 | 17.465 | 17.7928 | -0.32775 | -1.64731 | 0.884167 | 9 | 6 | 209 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | stratum | Retrieval-Missing-Gold | SelfUpdate | 224 | 17.465 | 17.223 | 0.242042 | -1.09655 | 1.5827 | 9 | 8 | 207 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | stratum | Retrieval-Missing-Gold | Static | 224 | 17.465 | 16.7213 | 0.743678 | -0.666661 | 2.17741 | 12 | 5 | 207 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | stratum | Retrieval-Missing-Gold | Broadcast | 224 | 17.465 | 17.3301 | 0.134905 | -1.13611 | 1.35739 | 9 | 8 | 207 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | type | bridge | NoComm | 240 | 33.8876 | 34.1291 | -0.241448 | -1.57255 | 1.00336 | 11 | 9 | 220 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | type | bridge | SelfUpdate | 240 | 33.8876 | 33.9851 | -0.0974559 | -1.5494 | 1.37708 | 13 | 13 | 214 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | type | bridge | Static | 240 | 33.8876 | 32.8721 | 1.0155 | -0.30923 | 2.41959 | 13 | 9 | 218 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | type | bridge | Broadcast | 240 | 33.8876 | 32.8919 | 0.995687 | -0.0517012 | 2.20646 | 15 | 9 | 216 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | type | comparison | NoComm | 60 | 24.9913 | 25.0539 | -0.0626154 | -2.95156 | 2.17342 | 5 | 1 | 54 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | type | comparison | SelfUpdate | 60 | 24.9913 | 23.2827 | 1.70855 | -0.18034 | 4.19003 | 4 | 2 | 54 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | type | comparison | Static | 60 | 24.9913 | 25.0539 | -0.0626154 | -2.95156 | 2.17342 | 5 | 1 | 54 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| R | type | comparison | Broadcast | 60 | 24.9913 | 26.6873 | -1.69601 | -5.55556 | 1.38413 | 2 | 4 | 54 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | stratum | Co-located-Gold | NoComm | 14 | 26.1905 | 7.14286 | 19.0476 | 4.7619 | 38.0952 | 4 | 0 | 10 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | stratum | Co-located-Gold | SelfUpdate | 14 | 26.1905 | 7.14286 | 19.0476 | 4.7619 | 38.0952 | 4 | 0 | 10 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | stratum | Co-located-Gold | Static | 14 | 26.1905 | 18.5065 | 7.68398 | -10.1732 | 27.9221 | 3 | 3 | 8 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | stratum | Co-located-Gold | Broadcast | 14 | 26.1905 | 35.7143 | -9.52381 | -21.4286 | 0 | 0 | 3 | 11 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | stratum | Disjoint-Gold | NoComm | 62 | 22.0469 | 10.6153 | 11.4316 | 5.31535 | 18.1914 | 17 | 4 | 41 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | stratum | Disjoint-Gold | SelfUpdate | 62 | 22.0469 | 13.9948 | 8.05216 | -0.315744 | 16.2874 | 18 | 6 | 38 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | stratum | Disjoint-Gold | Static | 62 | 22.0469 | 14.6352 | 7.41174 | -0.721569 | 15.2376 | 17 | 5 | 40 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | stratum | Disjoint-Gold | Broadcast | 62 | 22.0469 | 32.0001 | -9.95315 | -16.7121 | -4.04957 | 3 | 15 | 44 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | stratum | Retrieval-Missing-Gold | NoComm | 224 | 6.40422 | 3.3919 | 3.01233 | 1.25787 | 5.00449 | 19 | 2 | 203 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | stratum | Retrieval-Missing-Gold | SelfUpdate | 224 | 6.40422 | 4.0051 | 2.39912 | 0.40398 | 4.48684 | 18 | 5 | 201 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | stratum | Retrieval-Missing-Gold | Static | 224 | 6.40422 | 3.3919 | 3.01233 | 1.58824 | 4.71372 | 19 | 0 | 205 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | stratum | Retrieval-Missing-Gold | Broadcast | 224 | 6.40422 | 8.98925 | -2.58503 | -4.1521 | -1.24562 | 1 | 17 | 206 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | type | bridge | NoComm | 240 | 12.1813 | 6.29785 | 5.88341 | 3.39904 | 8.48411 | 34 | 6 | 200 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | type | bridge | SelfUpdate | 240 | 12.1813 | 7.74031 | 4.44095 | 1.46138 | 7.35957 | 35 | 10 | 195 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | type | bridge | Static | 240 | 12.1813 | 7.58252 | 4.59874 | 1.94379 | 7.28446 | 34 | 7 | 199 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | type | bridge | Broadcast | 240 | 12.1813 | 16.8074 | -4.62618 | -6.68792 | -2.58496 | 4 | 29 | 207 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | type | comparison | NoComm | 60 | 4.07698 | 0.107527 | 3.96945 | 0.63731 | 7.93651 | 6 | 0 | 54 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | type | comparison | SelfUpdate | 60 | 4.07698 | 0.119048 | 3.95793 | 0.634921 | 7.8952 | 5 | 1 | 54 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | type | comparison | Static | 60 | 4.07698 | 1.77419 | 2.30278 | -0.553166 | 5.63731 | 5 | 1 | 54 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |
| P | type | comparison | Broadcast | 60 | 4.07698 | 7.73016 | -3.65318 | -7.38958 | -0.562724 | 0 | 6 | 54 | POST_HOC_DESCRIPTIVE; marginal bootstrap CI, not multiplicity-confirmed |

In P, learned-minus-NoComm is +11.4316 points on Disjoint-Gold and +19.0476 on Co-located-Gold, compared with +3.01233 on Missing-Gold. The data do not support the advance prediction that the disjoint stratum must have the largest improvement. Against Broadcast the learned deficit is −9.95315 on Disjoint-Gold and −9.52381 on Co-located-Gold. Missing evidence and constrained transfer both matter; this experiment cannot isolate their causal shares.


![Post-hoc P subgroup contrasts; marginal bootstrap intervals are not multiplicity-confirmed primary tests.](figures/15_E5_subgroups.png)

Post-hoc P subgroup contrasts; marginal bootstrap intervals are not multiplicity-confirmed primary tests.

## The learned policy is often almost static

P seed 31 selects action 14 on all 300 questions: the two-edge in-star 0→1, 2→1 with quota 128. P seed 7 selects action 54 (six-edge broadcast with quota 256) on 294/300 questions. It produces exactly the same final answer as the Broadcast baseline on all 300 questions. Its high score therefore does not demonstrate a new routing advantage. P seed 19 mostly chooses another in-star. R seed 31 chooses NoComm on 241/300 questions. Across seeds, selected deterministic policies use only one to four actions per regime.

| regime | seed | n | unique_actions | top_action | top_action_fraction | NoComm_fraction | entropy_bits | frequencies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R | 7 | 300 | 4 | 45 | 0.573333 | 0 | 1.15615 | {45: 172, 40: 8, 31: 119, 54: 1} |
| R | 19 | 300 | 3 | 32 | 0.573333 | 0 | 1.36286 | {14: 89, 32: 172, 7: 39} |
| R | 31 | 300 | 2 | 0 | 0.803333 | 0.803333 | 0.715211 | {0: 241, 14: 59} |
| P | 7 | 300 | 3 | 54 | 0.98 | 0 | 0.154441 | {54: 294, 2: 5, 31: 1} |
| P | 19 | 300 | 3 | 32 | 0.653333 | 0 | 1.02749 | {14: 99, 32: 196, 21: 5} |
| P | 31 | 300 | 1 | 14 | 1 | 0 | -0 | {14: 300} |

| regime | seed | answers_equal_to_broadcast | F1_equal_to_broadcast | n |
| --- | --- | --- | --- | --- |
| R | 7 | 273 | 283 | 300 |
| R | 19 | 274 | 283 | 300 |
| R | 31 | 266 | 275 | 300 |
| P | 7 | 300 | 300 | 300 |
| P | 19 | 271 | 275 | 300 |
| P | 31 | 259 | 266 | 300 |

The three-seed mean is a statistical summary of three trained policies, not an implemented per-question mixture, voting ensemble or oracle selector. It should not be presented as a deployable single policy. Do not pick seed 7 post hoc as the headline router because it ties Broadcast. Selection was by validation utility; test-conditioned reselection would invalidate that decision rule.

We independently reconstructed the 53 observable features from saved initial responses and frozen assignments, then reevaluated the selected weights offline. All 1,800 regime×seed×question argmax actions matched their recorded choices. Mean maximum action probability is only 0.01878–0.02144, compared with uniform 1/55=0.01818, and mean entropy is 5.77835–5.78118 bits against log2(55)=5.78136. Thus near-uniform stochastic scores yield concentrated greedy choices. This is evidence of weak probability separation, not proof that training failed or an invitation to tune on test data. The next table distinguishes entropy of the actor's categorical probabilities from entropy of the selected action frequencies. A nearly uniform probability vector can still have the same slightly-largest action on every question; deterministic selection frequency alone does not reveal policy confidence.

| regime | seed | n | mean_max_probability | mean_top2_margin | mean_entropy_bits | min_entropy_bits | max_entropy_bits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P | 7 | 300 | 0.0187767 | 8.89161e-05 | 5.78118 | 5.78094 | 5.78129 |
| P | 19 | 300 | 0.0202479 | 0.000173953 | 5.77984 | 5.77801 | 5.78069 |
| P | 31 | 300 | 0.0210684 | 0.000751184 | 5.77913 | 5.77743 | 5.77982 |
| R | 7 | 300 | 0.0189283 | 5.1572e-05 | 5.78109 | 5.78089 | 5.78129 |
| R | 19 | 300 | 0.0200853 | 9.56446e-05 | 5.77977 | 5.77826 | 5.78055 |
| R | 31 | 300 | 0.0214391 | 0.000426044 | 5.77835 | 5.77708 | 5.77978 |

## Matching edge count and quota does not match resources

Of 1,800 learned/random paired evaluations, edge count and quota match in all 1,800. Actual recipient-call counts differ in 912 (50.6667%) and transmitted token totals differ in 254 (14.1111%). An in-star updates one recipient, whereas another two-edge graph may update two. Short messages also leave different unused quota. Consequently, the P Learned–MatchedRandom F1 difference +1.01577 (raw p=.0456954; Holm p=.502650) is not multiplicity-robust and cannot establish pure routing superiority. Even a significant result would need the actual resource mismatch disclosed.

## E2 and E3 expose the limits of component attribution

E2 reverses one-step cyclic destinations while preserving the exact sender payloads, edge count and receiver calls in shared evidence. The CW–CCW difference is −0.106522 F1 points with a CI spanning zero. This is a valid narrow destination-allocation contrast, not evidence that all graph structures are interchangeable. E2 shares E1's R answers, payloads and provider call keys; it is a reanalysis of overlapping experimental data, not independent replication.

E3 uses a designated receiver and exact token-count matching, but 298/300 prefix/crop inputs are identical because messages fit the 256-token cap. Only two questions undergo an actual content change. The observed +0.144928 receiver F1 difference cannot establish general content-insensitivity, prefix superiority, compression equivalence or safety. The null result mainly diagnoses an ineffective manipulation. Removing the 298 ties after observing results cannot turn the remaining two into a confirmatory experiment.

## Stopping is a small replay saving in this protocol

E4 stops early on 10/300 questions: 3 at depth 0, 2 at depth 1 and 5 at depth 2; 290 run all three rounds. Six early consensuses are wrong, and continuation corrects none of those six here. Stopped and final answers/F1 coincide on all questions. Replay avoids 54/3,600 logical calls (1.5%) and reduces mean transmitted payload from 216.31 to 200.6267 (7.2504%). Full trajectories were actually generated, so this does not measure realized spending/time saved by an online stopping service. Unanimity is not a correctness certificate.


![E4 shared-trajectory replay: stopping depth and answer quality across rounds.](figures/08_stopping_trajectory.png)

E4 shared-trajectory replay: stopping depth and answer quality across rounds.

## Training, seed instability and execution-order drift

Training F1/reward and validation utility are separate quantities. Seed 7 has its best validation utility in epoch 1; later training reward improves without improving the selection criterion. Seeds 19 and 31 select epoch 3. Validation has only 30 unique questions (60 correlated R/P contexts), so selected policy instability is unsurprising but its cause is not established experimentally. All nine collection epochs completed four PPO passes with 28 minibatch updates each and no KL early stop. Training stochastic-action entropy remains near log2(55), while greedy test actions concentrate strongly; no convergence or optimality guarantee follows from completing three epochs.


![Recorded training rewards and validation curves; repeated questions are not new independent samples.](figures/09_controller_training.png)

Recorded training rewards and validation curves; repeated questions are not new independent samples.


The historical 129-question P Learned–NoComm estimate was +7.83555 points; the final effect is +5.50062. The Broadcast deficit changed from −5.33519 to −4.43158. These changes reflect the evaluated prefix, not model learning during testing. Checkpoints were frozen. All prefixes and their construction are preserved below; no stopping rule or model selection used these post-hoc curves.

| prefix_n | regime | baseline | learned_F1 | baseline_F1 | difference |
| --- | --- | --- | --- | --- | --- |
| 50 | R | NoComm | 40.615 | 39.9039 | 0.711111 |
| 50 | R | Broadcast | 40.615 | 42.9039 | -2.28889 |
| 50 | P | NoComm | 9.7013 | 1.66667 | 8.03463 |
| 50 | P | Broadcast | 9.7013 | 15.6277 | -5.92641 |
| 100 | R | NoComm | 34.8408 | 34.9186 | -0.0777778 |
| 100 | R | Broadcast | 34.8408 | 35.0186 | -0.177778 |
| 100 | P | NoComm | 10.4062 | 2.11905 | 8.28716 |
| 100 | P | Broadcast | 10.4062 | 14.9805 | -4.57431 |
| 129 | R | NoComm | 34.274 | 34.8174 | -0.543437 |
| 129 | R | Broadcast | 34.274 | 34.0535 | 0.220463 |
| 129 | P | NoComm | 12.4018 | 4.56626 | 7.83555 |
| 129 | P | Broadcast | 12.4018 | 17.737 | -5.33519 |
| 150 | R | NoComm | 34.1719 | 34.7874 | -0.615504 |
| 150 | R | Broadcast | 34.1719 | 33.5527 | 0.619228 |
| 150 | P | NoComm | 11.8656 | 5.12698 | 6.73858 |
| 150 | P | Broadcast | 11.8656 | 16.4538 | -4.58826 |
| 200 | R | NoComm | 32.4711 | 33.0344 | -0.563285 |
| 200 | R | Broadcast | 32.4711 | 32.0923 | 0.378879 |
| 200 | P | NoComm | 10.4794 | 4.8775 | 5.60191 |
| 200 | P | Broadcast | 10.4794 | 14.0166 | -3.53715 |
| 250 | R | NoComm | 32.1812 | 32.6319 | -0.450628 |
| 250 | R | Broadcast | 32.1812 | 31.4115 | 0.76977 |
| 250 | P | NoComm | 10.6692 | 4.53057 | 6.13867 |
| 250 | P | Broadcast | 10.6692 | 15.0418 | -4.37258 |
| 300 | R | NoComm | 32.1084 | 32.314 | -0.205682 |
| 300 | R | Broadcast | 32.1084 | 31.651 | 0.457348 |
| 300 | P | NoComm | 10.5604 | 5.05979 | 5.50062 |
| 300 | P | Broadcast | 10.5604 | 14.992 | -4.43158 |
![Diagnostic cumulative effects in frozen execution order, not a training curve.](figures/16_prefix_drift.png)

Diagnostic cumulative effects in frozen execution order, not a training curve.

## Real examples: gains, harms and scoring effects

The examples below are deliberately selected from the three largest and three smallest paired seed-mean F1 changes in each listed comparison. They illustrate mechanisms and failure types; they are not a representative sample or causal explanation. Gold answers come from the publisher. Exact question IDs link back to the saved E5 record. No new LLM judge was used. Benchmark F1 changes may reflect answer granularity, not only factual correctness: for example, “American Civil War” versus gold “War” receives lower token-overlap F1 even though the longer phrase may be semantically related.

### P Learned - NoComm — 5a88981c554299206df2b307

Question: What type of conflict does Nathaniel Starbuck and Bernard Cornwell have in common?

Gold: **War**. Type: bridge; stratum: Retrieval-Missing-Gold. Baseline answer: `war` (F1 100); seed-mean change -50 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | American Civil War | 50 | [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]] | 256 | 540 |
| 19 | American Civil War | 50 | [[1, 0], [2, 0]] | 128 | 188 |
| 31 | American Civil War | 50 | [[0, 1], [2, 1]] | 128 | 139 |

### P Learned - NoComm — 5a7db0915542990b8f503a30

Question: What German philosopher, economist and political theorist was Jakob Lukas Schabelitz a friend and associate of?

Gold: **Karl Marx**. Type: bridge; stratum: Disjoint-Gold. Baseline answer: `Karl Marx` (F1 100); seed-mean change -33.3333 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | Karl Marx | 100 | [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]] | 256 | 254 |
| 19 | Karl Marx | 100 | [[1, 0], [2, 0]] | 128 | 69 |
| 31 | noanswer | 0 | [[0, 1], [2, 1]] | 128 | 65 |

### P Learned - NoComm — 5ae655845542991bbc9760c3

Question: An American physicist coined the term "soft energy path" in 1976.  Today he is chairman and chief scientist of what?

Gold: **Rocky Mountain Institute**. Type: bridge; stratum: Disjoint-Gold. Baseline answer: `the Rocky Mountain Institute` (F1 100); seed-mean change -33.3333 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | Rocky Mountain Institute | 100 | [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]] | 256 | 366 |
| 19 | noanswer | 0 | [[1, 0], [2, 0]] | 128 | 71 |
| 31 | the Rocky Mountain Institute | 100 | [[0, 1], [2, 1]] | 128 | 119 |

### P Learned - NoComm — 5ae76dfd5542997b22f6a755

Question: A series of ministerial meetings and agreements between European countries and includes The Swedish higher education system is called what?

Gold: **Bologna Process**. Type: bridge; stratum: Disjoint-Gold. Baseline answer: `noanswer` (F1 0); seed-mean change +100 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | the Bologna Process | 100 | [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]] | 256 | 108 |
| 19 | the Bologna Process | 100 | [[0, 1], [2, 1]] | 128 | 47 |
| 31 | the Bologna Process | 100 | [[0, 1], [2, 1]] | 128 | 47 |

### P Learned - NoComm — 5ab761d955429928e1fe3856

Question: Which Austrian Chancellor does Jörg Leichtfried serve under?

Gold: **Christian Kern**. Type: bridge; stratum: Retrieval-Missing-Gold. Baseline answer: `noanswer` (F1 0); seed-mean change +100 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | Christian Kern | 100 | [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]] | 256 | 174 |
| 19 | Christian Kern | 100 | [[0, 1], [2, 1]] | 128 | 80 |
| 31 | Christian Kern | 100 | [[0, 1], [2, 1]] | 128 | 80 |

### P Learned - NoComm — 5ab67aba55429954757d32e7

Question: The Battle of Nanchang was fought between the Chinese Nationals and what ground-based armed force of Japan?

Gold: **Imperial Japanese Army**. Type: bridge; stratum: Retrieval-Missing-Gold. Baseline answer: `noanswer` (F1 0); seed-mean change +100 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | Imperial Japanese Army | 100 | [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]] | 256 | 196 |
| 19 | Imperial Japanese Army | 100 | [[0, 1], [2, 1]] | 128 | 91 |
| 31 | Imperial Japanese Army | 100 | [[0, 1], [2, 1]] | 128 | 91 |

### P Learned - Broadcast — 5ab7f6d35542993667794070

Question: What is the name of the Australian specialist electronic music magazine that features avant-rock, experimental sound art, and experimental hip hop?

Gold: **Cyclic Defrost**. Type: bridge; stratum: Disjoint-Gold. Baseline answer: `Cyclic Defrost` (F1 100); seed-mean change -66.6667 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | Cyclic Defrost | 100 | [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]] | 256 | 176 |
| 19 | noanswer | 0 | [[1, 0], [2, 0]] | 128 | 81 |
| 31 | noanswer | 0 | [[0, 1], [2, 1]] | 128 | 14 |

### P Learned - Broadcast — 5a87bc3f5542996432c57275

Question: Which is located farther West, the Cascade Locks and Canal or Lake Worth Lagoon?

Gold: **Cascade Locks and Canal**. Type: comparison; stratum: Retrieval-Missing-Gold. Baseline answer: `Cascade Locks and Canal` (F1 100); seed-mean change -66.6667 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | Cascade Locks and Canal | 100 | [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]] | 256 | 334 |
| 19 | noanswer | 0 | [[1, 0], [2, 0]] | 128 | 135 |
| 31 | noanswer | 0 | [[0, 1], [2, 1]] | 128 | 14 |

### P Learned - Broadcast — 5abb73425542996cc5e49ff5

Question: What's the NASDAQ ticker code for Spirit Airlines, based at Detroit Metro Airport among others and an ultra low cost carrier?

Gold: **SAVE**. Type: bridge; stratum: Disjoint-Gold. Baseline answer: `SAVE` (F1 100); seed-mean change -66.6667 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | SAVE | 100 | [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]] | 256 | 124 |
| 19 | noanswer | 0 | [[1, 0], [2, 0]] | 128 | 55 |
| 31 | noanswer | 0 | [[0, 1], [2, 1]] | 128 | 14 |

### P Learned - Broadcast — 5ae0c7b055429945ae95944b

Question: The Admiral's Men occupied which kind of theatre in the 1590s ?

Gold: **Elizabethan theatre**. Type: bridge; stratum: Retrieval-Missing-Gold. Baseline answer: `noanswer` (F1 0); seed-mean change +16.6667 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | noanswer | 0 | [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]] | 256 | 128 |
| 19 | the Rose Theatre | 50 | [[1, 0], [2, 0]] | 128 | 13 |
| 31 | noanswer | 0 | [[0, 1], [2, 1]] | 128 | 58 |

### P Learned - Broadcast — 5adc6ded55429947ff17395d

Question: Which of the two airlines that operate flights into Enrique Malek International Airport is the larger air carrier in Panama?

Gold: **Copa Airlines**. Type: bridge; stratum: Disjoint-Gold. Baseline answer: `Air Panama` (F1 0); seed-mean change +33.3333 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | Air Panama | 0 | [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]] | 256 | 346 |
| 19 | Air Panama | 0 | [[1, 0], [2, 0]] | 128 | 72 |
| 31 | Copa Airlines | 100 | [[0, 1], [2, 1]] | 128 | 166 |

### P Learned - Broadcast — 5ab6aaff55429953192ad35b

Question: What Cayman Islands registered Mandarin and Cantonese-language television broadcaster launched Phoenix Hong Kong Channel on 28 March 2011?

Gold: **Phoenix Television**. Type: bridge; stratum: Disjoint-Gold. Baseline answer: `noanswer` (F1 0); seed-mean change +66.6667 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | noanswer | 0 | [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]] | 256 | 158 |
| 19 | Phoenix Television | 100 | [[0, 1], [2, 1]] | 128 | 72 |
| 31 | Phoenix Television | 100 | [[0, 1], [2, 1]] | 128 | 72 |

### R Learned - NoComm — 5ab6087a554299110f2199be

Question: Do both Korematsu v. United States and Chaplinsky v. New Hampshire cases regard the rights of U. S. citizens?

Gold: **yes**. Type: comparison; stratum: Retrieval-Missing-Gold. Baseline answer: `yes` (F1 100); seed-mean change -66.6667 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | yes | 100 | [[1, 0], [2, 0]] | 64 | 128 |
| 19 | no | 0 | [[0, 1], [2, 1]] | 128 | 205 |
| 31 | no | 0 | [[0, 1], [2, 1]] | 128 | 205 |

### R Learned - NoComm — 5ae6b6065542991bbc976168

Question: Out of the actors who have played the role of Luc Deveraux in the Universal Soldier franchise, which actor has also starred in the movies Holby City, Doctor Strange, the Bourne Ultimatum and Zero Dark Thirty?

Gold: **Scott Adkins**. Type: bridge; stratum: Co-located-Gold. Baseline answer: `Scott Adkins` (F1 100); seed-mean change -66.6667 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | noanswer | 0 | [[1, 0], [2, 0]] | 64 | 71 |
| 19 | noanswer | 0 | [[1, 0], [2, 0]] | 128 | 135 |
| 31 | Scott Adkins | 100 | [[0, 1], [2, 1]] | 128 | 256 |

### R Learned - NoComm — 5a7dc2185542990b8f503a7b

Question: Which county is Jollyville, Texas located in that had a population of 1,024,266 as of the 2010 census?

Gold: **Travis County**. Type: bridge; stratum: Retrieval-Missing-Gold. Baseline answer: `Travis County` (F1 100); seed-mean change -66.6667 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | noanswer | 0 | [[2, 0], [2, 1]] | 256 | 118 |
| 19 | noanswer | 0 | [[0, 1], [1, 2]] | 64 | 118 |
| 31 | Travis County | 100 | [] | 0 | 0 |

### R Learned - NoComm — 5a8f7e785542992414482ae6

Question: Are both Pomeranian and Sporting Lucas Terrier named after a particular person?

Gold: **no**. Type: comparison; stratum: Retrieval-Missing-Gold. Baseline answer: `noanswer` (F1 0); seed-mean change +33.3333 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | no | 100 | [[2, 0], [2, 1]] | 256 | 14 |
| 19 | noanswer | 0 | [[1, 0], [2, 0]] | 128 | 14 |
| 31 | noanswer | 0 | [] | 0 | 0 |

### R Learned - NoComm — 5a8df1a65542995085b3735f

Question: Rossiya Airlines in located in the largest what on earth?

Gold: **largest country**. Type: bridge; stratum: Disjoint-Gold. Baseline answer: `model layout` (F1 0); seed-mean change +44.4444 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | model layout | 0 | [[1, 0], [2, 0]] | 64 | 108 |
| 19 | country | 66.6667 | [[0, 1], [2, 1]] | 128 | 182 |
| 31 | country | 66.6667 | [[0, 1], [2, 1]] | 128 | 182 |

### R Learned - NoComm — 5ac05310554299012d1db5e7

Question: What is the name of the murder victim that worked in a restaurant, in which Mark Fuhrman was the detective?

Gold: **Ron Goldman**. Type: bridge; stratum: Retrieval-Missing-Gold. Baseline answer: `noanswer` (F1 0); seed-mean change +66.6667 points.

| seed | answer | F1 | edges | quota | actual_payload |
| --- | --- | --- | --- | --- | --- |
| 7 | Ron Goldman | 100 | [[1, 0], [2, 0]] | 64 | 61 |
| 19 | Ron Goldman | 100 | [[0, 1], [2, 1]] | 128 | 61 |
| 31 | noanswer | 0 | [] | 0 | 0 |

## What the data cannot settle

Citations were checked as annotated label sets, not entailment. Refusal incidence is retained in primary scores; conditioning on refusal status is treatment-dependent and only descriptive. The deployment alias is not an immutable model snapshot. One benchmark, one evidence partition/role/crop seed and three optimizer seeds do not establish cross-model, cross-partition or live-Web robustness. No new external baseline systems were run by this analysis. These boundaries should shape the paper, not be hidden in a footnote.
