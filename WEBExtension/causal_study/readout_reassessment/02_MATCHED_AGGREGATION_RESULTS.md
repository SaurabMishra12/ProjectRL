# Complete matched aggregation results

**Exploratory post hoc counterfactual readouts; zero new completions.** All 20 arms×300 questions×3 readouts are present. P means partitioned evidence; R means shared evidence. Policy seeds are separate. The same readout is applied to the same saved ordered responses in each compatible arm; readout changes add no requests or tokens.

Readouts: original normalized plurality with earliest-node ties; abstention-aware plurality with the same ties; abstention-aware plurality with normalized lexical ties. The latter is an order-sensitivity control. Every substantive selection is an existing proposal. When all agents abstain, the repaired rules return an explicit abstention marker; its EM is 0 on this selected cohort. The any-correct ceiling therefore bounds these selection readouts here. It is not an upper bound on unrestricted generative synthesis.

Selection% is correct final answers divided by questions with any correct final proposal. Coverage is an oracle diagnostic, never an input to voting. Abstention% is final-output abstention; agent-abstention counts/frequency are also in the full CSV. Changed compares normalized strings with the original readout, including explicit abstention-marker standardization.

The complete joined table, including physical requests and token fields for every arm/readout, is [results/matched_results_with_costs.csv](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/results/matched_results_with_costs.csv).

## original


| Arm | Correct/300 | EM% | Answer F1 | Abstention% | Any-correct/300 | Selection% | Ceiling gap pp | Changed |
|---|---|---|---|---|---|---|---|---|
| P_NoComm | 10 | 3.333 | 5.060 | 90.667 | 67 | 14.925 | 19.000 | 0 |
| P_SelfUpdate | 12 | 4.000 | 6.216 | 88.667 | 68 | 17.647 | 18.667 | 0 |
| P_Static | 13 | 4.333 | 6.421 | 88.333 | 66 | 19.697 | 17.667 | 0 |
| P_Broadcast | 34 | 11.333 | 14.992 | 78.667 | 66 | 51.515 | 10.667 | 0 |
| P_Learned_seed7 | 34 | 11.333 | 14.992 | 78.667 | 66 | 51.515 | 10.667 | 0 |
| P_Learned_seed19 | 20 | 6.667 | 9.165 | 86.000 | 68 | 29.412 | 16.000 | 0 |
| P_Learned_seed31 | 17 | 5.667 | 7.524 | 88.000 | 66 | 25.758 | 16.333 | 0 |
| P_MatchedRandom_seed7 | 34 | 11.333 | 14.992 | 78.667 | 66 | 51.515 | 10.667 | 0 |
| P_MatchedRandom_seed19 | 15 | 5.000 | 7.067 | 87.333 | 66 | 22.727 | 17.000 | 0 |
| P_MatchedRandom_seed31 | 13 | 4.333 | 6.575 | 89.333 | 69 | 18.841 | 18.667 | 0 |
| R_NoComm | 69 | 23.000 | 32.314 | 51.000 | 79 | 87.342 | 3.333 | 0 |
| R_SelfUpdate | 68 | 22.667 | 31.845 | 52.333 | 77 | 88.312 | 3.000 | 0 |
| R_Static | 68 | 22.667 | 31.308 | 52.667 | 79 | 86.076 | 3.667 | 0 |
| R_Broadcast | 70 | 23.333 | 31.651 | 52.667 | 76 | 92.105 | 2.000 | 0 |
| R_Learned_seed7 | 73 | 24.333 | 32.590 | 52.000 | 80 | 91.250 | 2.333 | 0 |
| R_Learned_seed19 | 71 | 23.667 | 31.966 | 51.000 | 80 | 88.750 | 3.000 | 0 |
| R_Learned_seed31 | 68 | 22.667 | 31.770 | 51.000 | 78 | 87.179 | 3.333 | 0 |
| R_MatchedRandom_seed7 | 69 | 23.000 | 32.108 | 51.667 | 77 | 89.610 | 2.667 | 0 |
| R_MatchedRandom_seed19 | 71 | 23.667 | 32.363 | 51.333 | 80 | 88.750 | 3.000 | 0 |
| R_MatchedRandom_seed31 | 67 | 22.333 | 32.056 | 50.667 | 76 | 88.158 | 3.000 | 0 |



## abstention_aware


| Arm | Correct/300 | EM% | Answer F1 | Abstention% | Any-correct/300 | Selection% | Ceiling gap pp | Changed |
|---|---|---|---|---|---|---|---|---|
| P_NoComm | 62 | 20.667 | 28.882 | 52.667 | 67 | 92.537 | 1.667 | 114 |
| P_SelfUpdate | 61 | 20.333 | 28.803 | 53.333 | 68 | 89.706 | 2.333 | 107 |
| P_Static | 60 | 20.000 | 28.026 | 52.000 | 66 | 90.909 | 2.000 | 111 |
| P_Broadcast | 65 | 21.667 | 30.113 | 54.000 | 66 | 98.485 | 0.333 | 76 |
| P_Learned_seed7 | 65 | 21.667 | 30.113 | 54.000 | 66 | 98.485 | 0.333 | 76 |
| P_Learned_seed19 | 65 | 21.667 | 29.947 | 52.667 | 68 | 95.588 | 1.000 | 100 |
| P_Learned_seed31 | 63 | 21.000 | 29.056 | 53.667 | 66 | 95.455 | 1.000 | 105 |
| P_MatchedRandom_seed7 | 65 | 21.667 | 30.113 | 54.000 | 66 | 98.485 | 0.333 | 76 |
| P_MatchedRandom_seed19 | 63 | 21.000 | 29.441 | 52.000 | 66 | 95.455 | 1.000 | 107 |
| P_MatchedRandom_seed31 | 66 | 22.000 | 30.439 | 50.667 | 69 | 95.652 | 1.000 | 118 |
| R_NoComm | 73 | 24.333 | 33.914 | 47.333 | 79 | 92.405 | 2.000 | 13 |
| R_SelfUpdate | 71 | 23.667 | 33.607 | 48.667 | 77 | 92.208 | 2.000 | 12 |
| R_Static | 75 | 25.000 | 34.693 | 47.333 | 79 | 94.937 | 1.333 | 18 |
| R_Broadcast | 73 | 24.333 | 33.273 | 48.667 | 76 | 96.053 | 1.000 | 15 |
| R_Learned_seed7 | 78 | 26.000 | 34.874 | 47.000 | 80 | 97.500 | 0.667 | 17 |
| R_Learned_seed19 | 77 | 25.667 | 34.461 | 46.667 | 80 | 96.250 | 1.000 | 15 |
| R_Learned_seed31 | 72 | 24.000 | 33.731 | 47.333 | 78 | 92.308 | 2.000 | 13 |
| R_MatchedRandom_seed7 | 73 | 24.333 | 34.003 | 47.333 | 77 | 94.805 | 1.333 | 15 |
| R_MatchedRandom_seed19 | 76 | 25.333 | 34.552 | 47.667 | 80 | 95.000 | 1.333 | 13 |
| R_MatchedRandom_seed31 | 69 | 23.000 | 32.951 | 47.667 | 76 | 90.789 | 2.333 | 11 |



## abstention_lexical


| Arm | Correct/300 | EM% | Answer F1 | Abstention% | Any-correct/300 | Selection% | Ceiling gap pp | Changed |
|---|---|---|---|---|---|---|---|---|
| P_NoComm | 64 | 21.333 | 29.461 | 52.667 | 67 | 95.522 | 1.000 | 124 |
| P_SelfUpdate | 61 | 20.333 | 28.787 | 53.333 | 68 | 89.706 | 2.333 | 120 |
| P_Static | 61 | 20.333 | 28.469 | 52.000 | 66 | 92.424 | 1.667 | 119 |
| P_Broadcast | 64 | 21.333 | 29.693 | 54.000 | 66 | 96.970 | 0.667 | 79 |
| P_Learned_seed7 | 64 | 21.333 | 29.688 | 54.000 | 66 | 96.970 | 0.667 | 79 |
| P_Learned_seed19 | 65 | 21.667 | 29.882 | 52.667 | 68 | 95.588 | 1.000 | 106 |
| P_Learned_seed31 | 63 | 21.000 | 29.181 | 53.667 | 66 | 95.455 | 1.000 | 113 |
| P_MatchedRandom_seed7 | 64 | 21.333 | 29.688 | 54.000 | 66 | 96.970 | 0.667 | 79 |
| P_MatchedRandom_seed19 | 64 | 21.333 | 29.911 | 52.000 | 66 | 96.970 | 0.667 | 114 |
| P_MatchedRandom_seed31 | 66 | 22.000 | 30.346 | 50.667 | 69 | 95.652 | 1.000 | 125 |
| R_NoComm | 73 | 24.333 | 33.760 | 47.333 | 79 | 92.405 | 2.000 | 19 |
| R_SelfUpdate | 71 | 23.667 | 33.561 | 48.667 | 77 | 92.208 | 2.000 | 16 |
| R_Static | 76 | 25.333 | 34.758 | 47.333 | 79 | 96.203 | 1.000 | 23 |
| R_Broadcast | 73 | 24.333 | 33.532 | 48.667 | 76 | 96.053 | 1.000 | 20 |
| R_Learned_seed7 | 78 | 26.000 | 34.842 | 47.000 | 80 | 97.500 | 0.667 | 21 |
| R_Learned_seed19 | 77 | 25.667 | 34.328 | 46.667 | 80 | 96.250 | 1.000 | 19 |
| R_Learned_seed31 | 73 | 24.333 | 33.866 | 47.333 | 78 | 93.590 | 1.667 | 17 |
| R_MatchedRandom_seed7 | 74 | 24.667 | 34.095 | 47.333 | 77 | 96.104 | 1.000 | 19 |
| R_MatchedRandom_seed19 | 77 | 25.667 | 34.349 | 47.667 | 80 | 96.250 | 1.000 | 19 |
| R_MatchedRandom_seed31 | 70 | 23.333 | 32.893 | 47.667 | 76 | 92.105 | 2.000 | 15 |


## Actual historical physical requests and tokens

Each row below joins an arm's referenced request keys to original started/success events, dispatch/receipt files and hash-checked provider response caches. Counts are **nonadditive physical request references**, not exclusive incremental spending by that arm. The same initial request can support several policies and all three offline readouts. Summing rows or multiplying by three overcounts spend. Policy-training/search requests outside these referenced keys are not allocated to test-serving rows.

Input and output columns are documented successful-provider usage. Output is a lower bound where usage is absent; unknown-execution billing can add more. Missing usage is never manufactured as complete zero. Readout computation itself adds 0 model calls. Message-token volume is an explicit communication payload measure, not billed inference tokens.


| Arm | Documented attempts | Successful keys | Input tokens | Output tokens LB | Missing output usage | Uncertain attempts | Message tokens |
|---|---|---|---|---|---|---|---|
| P_NoComm | 904 | 900 | 523403 | 199817 | 5 | 1 | 0 |
| P_SelfUpdate | 1804 | 1800 | 1048428 | 386886 | 11 | 1 | 0 |
| P_Static | 1506 | 1500 | 891828 | 348574 | 9 | 3 | 13514 |
| P_Broadcast | 1804 | 1800 | 1092464 | 430548 | 9 | 1 | 39862 |
| P_Learned_seed7 | 1797 | 1793 | 1086617 | 427491 | 9 | 1 | 37526 |
| P_Learned_seed19 | 1204 | 1200 | 715754 | 275666 | 6 | 1 | 14210 |
| P_Learned_seed31 | 1204 | 1200 | 714931 | 277963 | 6 | 1 | 12508 |
| P_MatchedRandom_seed7 | 1797 | 1793 | 1086827 | 428069 | 9 | 1 | 37689 |
| P_MatchedRandom_seed19 | 1454 | 1450 | 859170 | 337903 | 6 | 1 | 13261 |
| P_MatchedRandom_seed31 | 1463 | 1457 | 863437 | 335010 | 8 | 3 | 12451 |
| R_NoComm | 900 | 900 | 1222800 | 256165 | 3 | 0 | 0 |
| R_SelfUpdate | 1803 | 1800 | 2447950 | 501703 | 6 | 1 | 0 |
| R_Static | 1502 | 1500 | 2071208 | 421654 | 5 | 0 | 28584 |
| R_Broadcast | 1802 | 1800 | 2545310 | 499749 | 9 | 1 | 87988 |
| R_Learned_seed7 | 1384 | 1382 | 1901939 | 391259 | 6 | 0 | 22303 |
| R_Learned_seed19 | 1241 | 1239 | 1713358 | 349958 | 6 | 1 | 26088 |
| R_Learned_seed31 | 959 | 959 | 1307814 | 275474 | 3 | 0 | 5654 |
| R_MatchedRandom_seed7 | 1446 | 1445 | 1989248 | 409333 | 6 | 0 | 22387 |
| R_MatchedRandom_seed19 | 1445 | 1443 | 1990350 | 410726 | 5 | 0 | 25890 |
| R_MatchedRandom_seed31 | 1009 | 1007 | 1372530 | 293860 | 3 | 0 | 5219 |


Across the union:7,648 documented attempts,7,634 successful requests,7,786,951 known-success input tokens and at least 1,992,938 output tokens;37 successful responses lack output usage and 5 uncertain attempts retain unknown billing. No cached success lacks a documented dispatch. These totals are not new spend, nor a complete exact invoice. The saved token/call accounting is independent of which readout is displayed.

No per-question ensemble is formed from learned seeds. Their final responses and historical requests often overlap, and even identical accuracy does not create independent replication. The raw row-level data, request/attempt joins and uncertainty flags are in [results/question_readouts.jsonl](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/results/question_readouts.jsonl), [results/physical_accounting.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/results/physical_accounting.json) and [results/attempt_inventory.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/results/attempt_inventory.json).
