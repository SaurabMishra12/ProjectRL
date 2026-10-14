# 1. Complete results and metric register

Snapshot generated 2026-10-14T17:51:19.720260+00:00. Scope: completed frozen WEBExtension E1–E5, controller training/validation/pilot, operational cost evidence, and the supplied 13-page manuscript. Each experiment uses the same 300 held-out question identities; the unit of inference is the question, not the arm, agent, seed, or repeated epoch. All analysis is offline; no inference calls were issued. Earlier raw artifacts and the 129-question audit are preserved. Display tables round to six significant digits; CSV/JSON files retain machine-readable precision. NA is unknown or inapplicable, never an invented zero.

Run identifiers: `run_f6aa61bac2c3dfae`, `controller_study_a0b3768357484b45`, split `controller_splits_2e025f8ae3d9815e`. Input hashes and final artifact hashes are in `final_manifest.json`.

## Evidence status and interpretation

The completed E1–E5 files have 300 distinct test IDs each, identical question sets, matching completion SHA256 values, and unchanged frozen scientific source/checkpoint hashes. Answer F1 and EM were recomputed with the locally preserved official HotpotQA evaluator; all stored scores matched. All distinct transmitted payloads were recounted with the pinned tokenizer. This is strong internal consistency/provenance evidence, not external certification of the Azure deployment snapshot or an independent rerun.

The old paper's P1–P5/W1–W4 numbers are a separate historical evidence tier. The supplied manuscript is preserved in full text and its numerical lines catalogued. Those historical runs have not been newly reproduced or certified by the current E-suite; their numbers must not be relabeled as E1–E5 results. The result catalog below is exhaustive for the completed E-suite's recorded scalar condition fields and the explicitly listed recomputed diagnostics. It cannot supply unmeasured entailment, true provider bills, server compute time, or semantic/pretraining leakage.

## Data, arms and metric units

R means every agent sees the same BM25 top-six candidate articles. P means those six articles are exclusively partitioned across three agents (normally two each). F1 and EM use 0–100 scales. Payload is actual inter-agent token transmission per question; a broadcast copy counts separately per directed edge. Logical calls count the arm's required generations including shared initials, not new provider charges. Supporting-label metrics measure citation-set overlap with annotations, not textual entailment. E3 scores a designated receiver; all other main arms use normalized majority voting.

| experiment | n | target | unique_ids | ids_within_frozen_test | status | sha256 | complete_sha_matches_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 300 | 300 | 300 | True | COMPLETE | 1afe9ce6293a85acdcc743ee18b81c9ae79d57cd8258697cd715e77b4748a608 | True |
| E2 | 300 | 300 | 300 | True | COMPLETE | dfae111876fc8fbcc3aaadd7ddff050909b3e76f0bb2e9ee51db7fa7490ed5c1 | True |
| E3 | 300 | 300 | 300 | True | COMPLETE | 793f10a77da7556cec6adec23e0bcbdd8e99cf5361ca977d202f61229cd2c8c6 | True |
| E4 | 300 | 300 | 300 | True | COMPLETE | 10ec23783228e25ab1002d0aeb241e7dd5f30f0b6e8a67e41d6f5b4cfe72084f | True |
| E5 | 300 | 300 | 300 | True | COMPLETE | cf34e15527704024760f6ceaf11a34f62cfbdf000c08422bd6568f165ef7beca | True |

## All main arm results

| experiment | condition | n | F1 | F1_CI_low | F1_CI_high | EM | EM_Wilson_low | EM_Wilson_high | payload_tokens_known_mean | logical_calls_known_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 9.04001 | 6.1714 | 12.0897 | 6.33333 | 4.09169 | 9.67913 | 66.8 | 6 |
| E1 | P_CW | 300 | 9.13297 | 6.24665 | 12.2831 | 6.66667 | 4.35675 | 10.0723 | 66.8 | 6 |
| E1 | P_NoComm | 300 | 5.26671 | 2.99996 | 7.73516 | 3.66667 | 2.05953 | 6.44538 | 0 | 3 |
| E1 | P_SelfUpdate | 300 | 5.7556 | 3.4541 | 8.25036 | 3.66667 | 2.05953 | 6.44538 | 0 | 6 |
| E1 | R_CCW | 300 | 32.839 | 28.015 | 37.7255 | 24 | 19.5153 | 29.1421 | 145.163 | 6 |
| E1 | R_CW | 300 | 32.7325 | 27.965 | 37.5447 | 24.3333 | 19.8222 | 29.4935 | 145.163 | 6 |
| E1 | R_NoComm | 300 | 31.7477 | 26.9969 | 36.6054 | 22.6667 | 18.292 | 27.7325 | 0 | 3 |
| E1 | R_SelfUpdate | 300 | 32.0889 | 27.3205 | 36.9821 | 23.6667 | 19.2089 | 28.7903 | 0 | 6 |
| E2 | R_CCW | 300 | 32.839 | 28.015 | 37.7255 | 24 | 19.5153 | 29.1421 | 145.163 | 6 |
| E2 | R_CW | 300 | 32.7325 | 27.965 | 37.5447 | 24.3333 | 19.8222 | 29.4935 | 145.163 | 6 |
| E2 | R_NoComm | 300 | 31.7477 | 26.9969 | 36.6054 | 22.6667 | 18.292 | 27.7325 | 0 | 3 |
| E3 | R_crop | 300 | 31.4904 | 26.7383 | 36.3405 | 23.3333 | 18.9028 | 28.4381 | 47.2767 | 4 |
| E3 | R_prefix | 300 | 31.6354 | 26.8561 | 36.4433 | 23.3333 | 18.9028 | 28.4381 | 47.2767 | 4 |
| E4 | P_final | 300 | 10.7717 | 7.66574 | 14.1489 | 7.66667 | 5.16276 | 11.241 | 216.31 | 12 |
| E4 | P_stopped | 300 | 10.7717 | 7.66574 | 14.1489 | 7.66667 | 5.16276 | 11.241 | 200.627 | 11.82 |
| E5 | P_Broadcast | 300 | 14.992 | 11.2729 | 18.9022 | 11.3333 | 8.22445 | 15.4199 | 132.873 | 6 |
| E5 | P_Learned_seed19 | 300 | 9.16516 | 6.25399 | 12.4176 | 6.66667 | 4.35675 | 10.0723 | 47.3667 | 4 |
| E5 | P_Learned_seed31 | 300 | 7.52407 | 4.8273 | 10.5025 | 5.66667 | 3.56775 | 8.8866 | 41.6933 | 4 |
| E5 | P_Learned_seed7 | 300 | 14.992 | 11.2729 | 18.9022 | 11.3333 | 8.22445 | 15.4199 | 125.087 | 5.97667 |
| E5 | P_MatchedRandom_seed19 | 300 | 7.06692 | 4.4529 | 9.93464 | 5 | 3.05316 | 8.0847 | 44.2033 | 4.83333 |
| E5 | P_MatchedRandom_seed31 | 300 | 6.57499 | 4.16531 | 9.17208 | 4.33333 | 2.54964 | 7.27175 | 41.5033 | 4.85667 |
| E5 | P_MatchedRandom_seed7 | 300 | 14.992 | 11.2729 | 18.9022 | 11.3333 | 8.22445 | 15.4199 | 125.63 | 5.97667 |
| E5 | P_NoComm | 300 | 5.05979 | 2.92058 | 7.44005 | 3.33333 | 1.82049 | 6.02618 | 0 | 3 |
| E5 | P_SelfUpdate | 300 | 6.21606 | 3.811 | 8.84132 | 4 | 2.30273 | 6.86042 | 0 | 6 |
| E5 | P_Static | 300 | 6.42086 | 3.96852 | 9.12816 | 4.33333 | 2.54964 | 7.27175 | 45.0467 | 5 |
| E5 | R_Broadcast | 300 | 31.651 | 26.8925 | 36.638 | 23.3333 | 18.9028 | 28.4381 | 293.293 | 6 |
| E5 | R_Learned_seed19 | 300 | 31.9659 | 27.1022 | 36.8779 | 23.6667 | 19.2089 | 28.7903 | 86.96 | 4.13 |
| E5 | R_Learned_seed31 | 300 | 31.7696 | 27.0181 | 36.6592 | 22.6667 | 18.292 | 27.7325 | 18.8467 | 3.19667 |
| E5 | R_Learned_seed7 | 300 | 32.5896 | 27.8183 | 37.492 | 24.3333 | 19.8222 | 29.4935 | 74.3433 | 4.60667 |
| E5 | R_MatchedRandom_seed19 | 300 | 32.3627 | 27.6613 | 37.1855 | 23.6667 | 19.2089 | 28.7903 | 86.3 | 4.81 |
| E5 | R_MatchedRandom_seed31 | 300 | 32.0562 | 27.3532 | 36.9556 | 22.3333 | 17.9871 | 27.3791 | 17.3967 | 3.35667 |
| E5 | R_MatchedRandom_seed7 | 300 | 32.1076 | 27.3209 | 36.9892 | 23 | 18.5972 | 28.0855 | 74.6233 | 4.81667 |
| E5 | R_NoComm | 300 | 32.314 | 27.4669 | 37.2197 | 23 | 18.5972 | 28.0855 | 0 | 3 |
| E5 | R_SelfUpdate | 300 | 31.8446 | 27.0474 | 36.7541 | 22.6667 | 18.292 | 27.7325 | 0 | 6 |
| E5 | R_Static | 300 | 31.3085 | 26.6714 | 36.1364 | 22.6667 | 18.292 | 27.7325 | 95.28 | 5 |

F1 summary intervals use 5,000 question-bootstrap resamples (seed 42); EM uses a Wilson interval. A per-arm interval is not a paired contrast test. Below, canonical completed contrast reports use 10,000 resamples and 10,000 Monte Carlo sign randomizations, with the (1+k)/(B+1) p-value correction. These p-values are not exact enumeration. Original family-specific Holm corrections are preserved.

## E1–E4 canonical paired effects

| experiment | regime | metric | contrast | n | effect | CI95 | raw_p | Holm_p | helped | harmed | ties |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 |  | answer F1 | (CW-NoComm)_P - (CW-NoComm)_R | 300 | 2.88151 | [-0.25996063576945927, 6.010893847437964] | 0.0712929 | NA (unknown) | 25 | 16 | 259 |
| E1 |  | answer F1 | R_CW - R_NoComm | 300 | 0.984757 | [-0.7680588093823387, 2.893296595379929] | 0.29547 | NA (unknown) | 14 | 6 | 280 |
| E1 |  | answer F1 | P_CW - P_NoComm | 300 | 3.86626 | [1.356056786350904, 6.485954184704185] | 0.00149985 | NA (unknown) | 20 | 5 | 275 |
| E1 |  | answer F1 | R_CCW - R_NoComm | 300 | 1.09128 | [-0.8013360790444124, 3.144086659074404] | 0.285971 | NA (unknown) | 14 | 10 | 276 |
| E1 |  | answer F1 | P_CCW - P_NoComm | 300 | 3.7733 | [1.4206812876665817, 6.354451447245564] | 0.00239976 | NA (unknown) | 20 | 4 | 276 |
| E1 |  | answer F1 | R_CW - R_SelfUpdate | 300 | 0.643572 | [-0.870582611832612, 2.2149882209988596] | 0.420958 | NA (unknown) | 13 | 6 | 281 |
| E1 |  | answer F1 | P_CW - P_SelfUpdate | 300 | 3.37737 | [0.8490954715219422, 6.052734487734487] | 0.0109989 | NA (unknown) | 18 | 7 | 275 |
| E2 |  | answer F1 | R_CW - R_CCW | 300 | -0.106524 | [-1.4761482730232731, 1.147305934805935] | 0.880112 | NA (unknown) | 10 | 6 | 284 |
| E3 |  | answer F1 | R_prefix - R_crop | 300 | 0.145022 | [-0.13636363636363635, 0.5714285714285714] | 1 | NA (unknown) | 1 | 1 | 298 |
| E4 |  | answer F1 | P_stopped - P_final | 300 | 0 | [0.0, 0.0] | 1 | NA (unknown) | 0 | 0 | 300 |

## E5 canonical whole-policy effects: all F1 and utility contrasts

| experiment | regime | metric | contrast | n | effect | CI95 | raw_p | Holm_p | helped | harmed | ties |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E5 | R | f1 | seed_mean_learned - Static | 300 | 0.799874 | [-0.3725974025974023, 2.018996188996189] | 0.194781 | 1 | 18 | 10 | 272 |
| E5 | R | f1 | seed_mean_learned - NoComm | 300 | -0.205682 | [-1.4058396863427818, 0.9160068531972552] | 0.728327 | 1 | 16 | 10 | 274 |
| E5 | R | f1 | seed_mean_learned - SelfUpdate | 300 | 0.263745 | [-0.9745510180107698, 1.5131373942248554] | 0.683332 | 1 | 17 | 15 | 268 |
| E5 | R | f1 | seed_mean_learned - Broadcast | 300 | 0.457348 | [-0.6878669581765553, 1.6098516838338817] | 0.439156 | 1 | 17 | 13 | 270 |
| E5 | R | f1 | seed_mean_learned - MatchedRandom | 300 | -0.0671327 | [-0.8712590495571044, 0.7084754329491171] | 0.863514 | 1 | 14 | 9 | 277 |
| E5 | R | utility_gain | seed_mean_learned - Static | 300 | 0.0198058 | [0.00812132339100251, 0.0320216639516201] | 0.00059994 | 0.00839916 | 285 | 15 | 0 |
| E5 | R | utility_gain | seed_mean_learned - NoComm | 300 | -0.0200225 | [-0.031963690317788084, -0.008737465820745737] | 9.999e-05 | 0.0019998 | 14 | 286 | 0 |
| E5 | R | utility_gain | seed_mean_learned - SelfUpdate | 300 | 0.00467175 | [-0.007651433898099584, 0.017134670227165114] | 0.474653 | 1 | 245 | 53 | 2 |
| E5 | R | utility_gain | seed_mean_learned - Broadcast | 300 | 0.0564798 | [0.0450247708078268, 0.0679980477526322] | 9.999e-05 | 0.0019998 | 289 | 11 | 0 |
| E5 | R | utility_gain | seed_mean_learned - MatchedRandom | 300 | 0.00160438 | [-0.006423761881092969, 0.009409661256685072] | 0.693931 | 1 | 212 | 28 | 60 |
| E5 | P | f1 | seed_mean_learned - Static | 300 | 4.13955 | [1.86126018246986, 6.416441100715294] | 0.00039996 | 0.00639936 | 39 | 8 | 253 |
| E5 | P | f1 | seed_mean_learned - NoComm | 300 | 5.50062 | [3.456924045277936, 7.75711375752647] | 9.999e-05 | 0.0019998 | 40 | 6 | 254 |
| E5 | P | f1 | seed_mean_learned - SelfUpdate | 300 | 4.34435 | [1.9230151850574058, 6.868642444595955] | 0.00039996 | 0.00639936 | 40 | 11 | 249 |
| E5 | P | f1 | seed_mean_learned - Broadcast | 300 | -4.43158 | [-6.312186913866229, -2.707283755144286] | 9.999e-05 | 0.0019998 | 4 | 35 | 261 |
| E5 | P | f1 | seed_mean_learned - MatchedRandom | 300 | 1.01577 | [0.07171884455150579, 2.0603352214541726] | 0.0456954 | 0.50265 | 15 | 8 | 277 |
| E5 | P | utility_gain | seed_mean_learned - Static | 300 | 0.0328434 | [0.010490790898433012, 0.055136940415954884] | 0.00359964 | 0.0467953 | 73 | 227 | 0 |
| E5 | P | utility_gain | seed_mean_learned - NoComm | 300 | 0.0162423 | [-0.00416293249170223, 0.03884027789336464] | 0.142386 | 1 | 39 | 261 | 0 |
| E5 | P | utility_gain | seed_mean_learned - SelfUpdate | 300 | 0.0246796 | [0.0004360020625306472, 0.049870481328375627] | 0.049695 | 0.50265 | 41 | 259 | 0 |
| E5 | P | utility_gain | seed_mean_learned - Broadcast | 300 | -0.0131295 | [-0.03193784721437454, 0.004032709495065079] | 0.166383 | 1 | 266 | 34 | 0 |
| E5 | P | utility_gain | seed_mean_learned - MatchedRandom | 300 | 0.0135874 | [0.004281184322821576, 0.023926494155176607] | 0.00639936 | 0.0767923 | 246 | 49 | 5 |

Only the frozen E1 interaction is the primary endpoint. All E5 contrasts are exploratory relative to that primary question, despite the now-complete test. Utility is the prespecified F1-gain fraction minus payload and recipient-call penalties; it is not answer accuracy. Seeds are averaged within each question before uncertainty estimation. No noninferiority conclusion is licensed by a nonsignificant difference: no E-suite margin was predeclared.

## E5 seed-mean resource tradeoffs

| regime | baseline | n | learned_seed_mean_F1 | baseline_F1 | F1_change | learned_mean_payload_tokens | baseline_payload_tokens | payload_reduction_percent | learned_mean_logical_calls | baseline_calls | call_reduction_percent | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R | NoComm | 300 | 32.1084 | 32.314 | -0.205682 | 60.05 | 0 | NA (unknown) | 3.97778 | 3 | -32.5926 | COMPLETE_300 |
| R | SelfUpdate | 300 | 32.1084 | 31.8446 | 0.263745 | 60.05 | 0 | NA (unknown) | 3.97778 | 6 | 33.7037 | COMPLETE_300 |
| R | Static | 300 | 32.1084 | 31.3085 | 0.799874 | 60.05 | 95.28 | 36.9752 | 3.97778 | 5 | 20.4444 | COMPLETE_300 |
| R | Broadcast | 300 | 32.1084 | 31.651 | 0.457348 | 60.05 | 293.293 | 79.5256 | 3.97778 | 6 | 33.7037 | COMPLETE_300 |
| P | NoComm | 300 | 10.5604 | 5.05979 | 5.50062 | 71.3822 | 0 | NA (unknown) | 4.65889 | 3 | -55.2963 | COMPLETE_300 |
| P | SelfUpdate | 300 | 10.5604 | 6.21606 | 4.34435 | 71.3822 | 0 | NA (unknown) | 4.65889 | 6 | 22.3519 | COMPLETE_300 |
| P | Static | 300 | 10.5604 | 6.42086 | 4.13955 | 71.3822 | 45.0467 | -58.4628 | 4.65889 | 5 | 6.82222 | COMPLETE_300 |
| P | Broadcast | 300 | 10.5604 | 14.992 | -4.43158 | 71.3822 | 132.873 | 46.278 | 4.65889 | 6 | 22.3519 | COMPLETE_300 |

## Training and validation: every epoch

| seed | epoch | mean_utility_gain | mean_f1 | n_contexts | optimizer_loss | optimizer_updates | optimizer_early_kl_stop | optimizer_max_kl | train_mean_reward | train_mean_f1 | train_rollouts | selected_epoch | train_action_entropy_bits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 1 | 0.0228066 | 39.8918 | 60 | -0.0256007 | 28 | False | 0.000291981 | -0.0319343 | 45.9436 | 200 | 1 | 5.55474 |
| 7 | 2 | -0.0214084 | 33.3677 | 60 | -0.0405123 | 28 | False | 0.000257187 | -0.0146944 | 47.538 | 200 | 1 | 5.59555 |
| 7 | 3 | -0.00413586 | 35.3717 | 60 | -0.0442788 | 28 | False | 0.00114451 | 0.00688462 | 49.7342 | 200 | 1 | 5.54869 |
| 19 | 1 | -0.0119008 | 34.6825 | 60 | -0.0356855 | 28 | False | 0.000268281 | -0.0101793 | 48.268 | 200 | 3 | 5.55316 |
| 19 | 2 | -0.0343973 | 32.3345 | 60 | -0.0164988 | 28 | False | 0.000244614 | -0.00952488 | 48.2342 | 200 | 3 | 5.49333 |
| 19 | 3 | 0.0274682 | 38.0118 | 60 | -0.0340154 | 28 | False | 0.000523943 | 0.00850476 | 49.8961 | 200 | 3 | 5.53402 |
| 31 | 1 | -0.00479782 | 34.5714 | 60 | -0.0392239 | 28 | False | 0.000238843 | -0.0243962 | 46.5675 | 200 | 3 | 5.60358 |
| 31 | 2 | 0.030153 | 37.8745 | 60 | -0.0234646 | 28 | False | 0.000394352 | -0.0149414 | 47.618 | 200 | 3 | 5.62387 |
| 31 | 3 | 0.035072 | 37.8745 | 60 | -0.034304 | 28 | False | 0.000779569 | -0.0160049 | 47.4704 | 200 | 3 | 5.5429 |

There are 100 unique training and 30 unique validation questions, evaluated in both R and P. Per seed: 3 collection epochs × 200 contexts = 600 training rollouts and 3 × 60 = 180 validation records. Across seeds 7/19/31: 1,800 rollouts, 540 validation records and 252 optimizer minibatch updates (28 per epoch per seed). Validation-selected epochs are 1, 3, 3. These are not 1,800 independently sampled training questions. The prior larger study is archived; it is included only in global cost accounting.

| regime | seed | n | F1 | EM | payload | calls | utility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R | 7 | 300 | 32.5896 | 24.3333 | 74.3433 | 4.60667 | -0.0229094 |
| R | 19 | 300 | 31.9659 | 23.6667 | 86.96 | 4.13 | -0.0271419 |
| R | 31 | 300 | 31.7696 | 22.6667 | 18.8467 | 3.19667 | -0.0100163 |
| P | 7 | 300 | 14.992 | 11.3333 | 125.087 | 5.97667 | 0.0302918 |
| P | 19 | 300 | 9.16516 | 6.66667 | 47.3667 | 4 | 0.0167093 |
| P | 31 | 300 | 7.52407 | 5.66667 | 41.6933 | 4 | 0.00172577 |

## Pilot and static validation (not held-out efficacy results)

| phase | regime | action | n | F1 | EM | payload | calls | utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot | P | CCW | 12 | 16.6667 | 16.6667 | 71.8333 | 6 | 0.0383333 |
| pilot | P | CW | 12 | 28.8889 | 16.6667 | 71.8333 | 6 | 0.160556 |
| pilot | P | OUTSTAR | 12 | 15 | 8.33333 | 46 | 5 | 0.0385947 |
| pilot | R | CCW | 12 | 87.2222 | 75 | 207.833 | 6 | -0.045 |
| pilot | R | CW | 12 | 87.2222 | 75 | 207.833 | 6 | -0.045 |
| pilot | R | OUTSTAR | 12 | 87.2222 | 75 | 128.833 | 5 | -0.0287564 |
| static_validation | P | CCW | 30 | 8.32723 | 6.66667 | 78.8333 | 6 | -0.0617277 |
| static_validation | P | CW | 30 | 14.2857 | 13.3333 | 78.8333 | 6 | -0.00214286 |
| static_validation | P | OUTSTAR | 30 | 13.913 | 13.3333 | 59.8 | 5 | 0.00731248 |
| static_validation | R | CCW | 30 | 57.072 | 56.6667 | 227.833 | 6 | -0.0319227 |
| static_validation | R | CW | 30 | 57.0898 | 56.6667 | 227.833 | 6 | -0.0317447 |
| static_validation | R | OUTSTAR | 30 | 55.8095 | 50 | 146.133 | 5 | -0.0287489 |

The 12-question pilot checks parsing, costs and invariants. Static selection uses 30 validation questions in both regimes; it is not a test-set comparison. Full nested numeric fields and source links are retained in the supplemental tables.

## Frozen selected actor distributions

| regime | seed | n | mean_max_probability | mean_top2_margin | mean_entropy_bits | min_entropy_bits | max_entropy_bits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P | 7 | 300 | 0.0187767 | 8.89161e-05 | 5.78118 | 5.78094 | 5.78129 |
| P | 19 | 300 | 0.0202479 | 0.000173953 | 5.77984 | 5.77801 | 5.78069 |
| P | 31 | 300 | 0.0210684 | 0.000751184 | 5.77913 | 5.77743 | 5.77982 |
| R | 7 | 300 | 0.0189283 | 5.1572e-05 | 5.78109 | 5.78089 | 5.78129 |
| R | 19 | 300 | 0.0200853 | 9.56446e-05 | 5.77977 | 5.77826 | 5.78055 |
| R | 31 | 300 | 0.0214391 | 0.000426044 | 5.77835 | 5.77708 | 5.77978 |

## Split overlap and availability

| pair | id_overlap | normalized_question_overlap | candidate_title_overlap |
| --- | --- | --- | --- |
| train__validation | 0 | 0 | 0 |
| train__heldout | 0 | 0 | 0 |
| train__pilot | 0 | 0 | 0 |
| validation__heldout | 0 | 0 | 0 |
| validation__pilot | 0 | 0 | 0 |
| heldout__pilot | 0 | 0 | 0 |

| stratum | n |
| --- | --- |
| Retrieval-Missing-Gold | 224 |
| Disjoint-Gold | 62 |
| Co-located-Gold | 14 |

Exact IDs, normalized questions and normalized candidate titles have no measured cross-partition overlap. Title aliases, semantic duplicates, pretraining contamination and all historical exposures were not exhaustively detectable. The same test questions are intentionally reused across E1–E5; they must not be counted as 1,500 independent observations.

## Physical usage, refusals and timing

| pipeline | successes | reported_input_tokens | successful_inputs_missing | reported_output_tokens_lower_bound | successful_outputs_missing | original_unresolved_attempts | attempts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| controlled | 6134 | 5960203 | 0 | 1582163 | 29 | 8 | 6153 |
| controller | 10019 | 10130139 | 0 | 2640066 | 37 | 5 | 10037 |

| field | value |
| --- | --- |
| scope | All original attempt IDs in WEBExtension/artifacts, including archived caps, pilots and preserved prior study; ledger copies de-duplicated by attempt ID |
| unique_attempts | 17967 |
| outcomes.success | 17926 |
| outcomes.rejected | 27 |
| outcomes.uncertain | 14 |
| reported_input_tokens_lower_bound | 17715215 |
| reported_output_tokens_lower_bound | 4663365 |
| attempts_missing_input_usage | 41 |
| attempts_missing_output_usage | 108 |
| refusals | 133 |
| monetary_cost | NA (unknown) |
| reason | No verified deployment price; missing/refusal/error usage remains unknown, not zero |

Known input and output dimensions are summed independently. Original report functions withheld both totals when either dimension was incomplete; originals are preserved, and these corrected known lower bounds are supplied alongside them. Unknown potentially billed original attempts remain unknown even when a replacement succeeds. Global attempt-ID-deduplicated totals include archived caps, pilots and the prior controller study; they are not only the final evaluation's costs. Logical arm totals must not be summed into a provider bill because arms share cached responses.

No verified deployment price or billing export is present, so currency spending and monetary savings are unmeasured. Missing token usage is not zero. Refusals remain abstentions in headline quality metrics. Timing covers captured local sessions and network intervals; nested timers overlap and do not estimate server compute or a randomized per-arm latency benefit.

| pipeline | http_status | metric | recorded_n | sum | mean | p50 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| controlled | 200.0 | network_wall_seconds | 5190 | 6301.72 | 1.2142 | 1.0383 | 2.26543 | 15.1097 |
| controlled | 200.0 | end_to_end_wall_seconds | 5190 | 12043.5 | 2.32052 | 1.563 | 7.10177 | 52.4525 |
| controlled | 400.0 | network_wall_seconds | 61 | 64.1927 | 1.05234 | 0.906329 | 1.84668 | 3.24893 |
| controlled | 400.0 | end_to_end_wall_seconds | 61 | 91.0981 | 1.49341 | 1.23387 | 3.11862 | 4.9129 |
| controlled | 429.0 | network_wall_seconds | 10 | 1.70654 | 0.170654 | 0.147195 | 0.27737 | 0.321978 |
| controlled | 429.0 | end_to_end_wall_seconds | 10 | 8.8375 | 0.88375 | 1.02281 | 1.35281 | 1.46534 |
| controlled | 500.0 | network_wall_seconds | 2 | 34.174 | 17.087 | 17.087 | 18.2568 | 18.3868 |
| controlled | 500.0 | end_to_end_wall_seconds | 2 | 34.6153 | 17.3077 | 17.3077 | 18.6239 | 18.7702 |
| controlled | unknown | network_wall_seconds | 5 | 497.011 | 99.4022 | 121.306 | 121.389 | 121.404 |
| controlled | unknown | end_to_end_wall_seconds | 5 | 544.834 | 108.967 | 121.494 | 129.808 | 131.071 |
| controller | 200.0 | network_wall_seconds | 9951 | 11556.9 | 1.16138 | 0.980348 | 2.12323 | 43.3915 |
| controller | 200.0 | end_to_end_wall_seconds | 9951 | 17551.9 | 1.76383 | 1.30604 | 4.1188 | 47.7289 |
| controller | 400.0 | network_wall_seconds | 68 | 61.2307 | 0.900452 | 0.863705 | 1.40957 | 1.84637 |
| controller | 400.0 | end_to_end_wall_seconds | 68 | 77.7532 | 1.14343 | 0.998772 | 1.96794 | 2.55169 |
| controller | 429.0 | network_wall_seconds | 13 | 2.19132 | 0.168563 | 0.149816 | 0.247868 | 0.271259 |
| controller | 429.0 | end_to_end_wall_seconds | 13 | 13.6409 | 1.0493 | 0.319194 | 3.73272 | 4.53289 |
| controller | unknown | network_wall_seconds | 5 | 481.694 | 96.3387 | 120.112 | 121.094 | 121.329 |
| controller | unknown | end_to_end_wall_seconds | 5 | 491.148 | 98.2295 | 120.361 | 126.226 | 126.927 |

| pipeline | session | start_utc | end_utc | status | wall_seconds_observed |
| --- | --- | --- | --- | --- | --- |
| run_f6aa61bac2c3dfae | 20261014T053903.462338+0000_8facdd98 | 2026-10-14T05:39:03.463764+00:00 | 2026-10-14T06:11:50.063951+00:00 | FAILED | 1966.6 |
| run_f6aa61bac2c3dfae | 20261014T070525.518460+0000_f47b4d6b | 2026-10-14T07:05:25.519166+00:00 | 2026-10-14T07:05:44.509884+00:00 | FAILED | 18.9908 |
| run_f6aa61bac2c3dfae | 20261014T070818.416120+0000_a357bbc3 | 2026-10-14T07:08:18.417347+00:00 | 2026-10-14T07:45:56.551488+00:00 | STOPPED | 2258.13 |
| run_f6aa61bac2c3dfae | 20261014T085630.507062+0000_d2e93b65 | 2026-10-14T08:56:30.507662+00:00 | 2026-10-14T09:09:04.980721+00:00 | FAILED | 754.473 |
| run_f6aa61bac2c3dfae | 20261014T091213.356757+0000_93ad24bf | 2026-10-14T09:12:13.357300+00:00 | 2026-10-14T09:28:42.071016+00:00 | FAILED | 988.714 |
| run_f6aa61bac2c3dfae | 20261014T102313.656448+0000_fa737840 | 2026-10-14T10:23:13.656802+00:00 | 2026-10-14T10:37:42.993482+00:00 | STOPPED | 869.337 |
| run_f6aa61bac2c3dfae | 20261014T110033.032359+0000_79069b5f | 2026-10-14T11:00:33.032930+00:00 | 2026-10-14T11:49:10.482501+00:00 | FAILED | 2917.45 |
| run_f6aa61bac2c3dfae | 20261014T121348.993806+0000_8ca8c734 | 2026-10-14T12:13:48.994481+00:00 | 2026-10-14T12:59:03.943359+00:00 | FAILED | 2714.95 |
| run_f6aa61bac2c3dfae | 20261014T130523.356180+0000_988e5ce4 | 2026-10-14T13:05:23.356738+00:00 | 2026-10-14T13:19:46.245010+00:00 | FAILED | 862.888 |
| run_f6aa61bac2c3dfae | 20261014T132135.514964+0000_951b35ea | 2026-10-14T13:21:35.515660+00:00 | 2026-10-14T13:24:41.400174+00:00 | FAILED | 185.885 |
| run_f6aa61bac2c3dfae | 20261014T132657.977912+0000_3169fb38 | 2026-10-14T13:26:57.978839+00:00 | 2026-10-14T13:29:37.251031+00:00 | COMPLETE | 159.272 |
| controller_study_a0b3768357484b45 | 20261014T064438.190745+0000_9e2df11d | 2026-10-14T06:44:38.191679+00:00 | 2026-10-14T07:02:23.834368+00:00 | STOPPED | 1065.64 |
| controller_study_a0b3768357484b45 | 20261014T070525.488913+0000_f6eaf6a0 | 2026-10-14T07:05:25.489511+00:00 | 2026-10-14T07:45:56.221740+00:00 | STOPPED | 2430.73 |
| controller_study_a0b3768357484b45 | 20261014T085630.369897+0000_f03a1064 | 2026-10-14T08:56:30.370258+00:00 | 2026-10-14T10:37:44.429586+00:00 | STOPPED | 6074.06 |
| controller_study_a0b3768357484b45 | 20261014T110032.649668+0000_25b9978d | 2026-10-14T11:00:32.650449+00:00 | 2026-10-14T11:00:59.490555+00:00 | FAILED | 26.8402 |
| controller_study_a0b3768357484b45 | 20261014T121348.285860+0000_ddabbb95 | 2026-10-14T12:13:48.286662+00:00 | 2026-10-14T12:27:10.284127+00:00 | FAILED | 801.998 |
| controller_study_a0b3768357484b45 | 20261014T124730.932952+0000_8f1bc695 | 2026-10-14T12:47:30.934054+00:00 | 2026-10-14T12:59:04.931681+00:00 | FAILED | 693.998 |
| controller_study_a0b3768357484b45 | 20261014T130522.492428+0000_fdf75b51 | 2026-10-14T13:05:22.493895+00:00 | 2026-10-14T13:35:22.474326+00:00 | FAILED | 1799.98 |
| controller_study_a0b3768357484b45 | 20261014T134158.734067+0000_54255982 | 2026-10-14T13:41:58.734736+00:00 | 2026-10-14T13:42:44.048070+00:00 | FAILED | 45.3134 |
| controller_study_a0b3768357484b45 | 20261014T144228.735307+0000_9d9225c5 | 2026-10-14T14:42:28.735803+00:00 | 2026-10-14T16:28:53.779667+00:00 | FAILED | 6385.04 |
| controller_study_a0b3768357484b45 | 20261014T163314.890086+0000_55dd2caa | 2026-10-14T16:33:14.892444+00:00 | 2026-10-14T16:36:12.275215+00:00 | COMPLETE | 177.384 |

## Every scalar metric, distribution and missing count

The following includes all recorded scalar condition/cost fields and the recomputed support/joint-label diagnostics. Boolean means are proportions. `sum_known` for an arbitrary metric is descriptive and not necessarily a scientifically meaningful aggregate. Missingness is explicit.

### all_support_available

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E1 | P_CW | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E1 | R_CCW | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E1 | R_CW | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E1 | R_NoComm | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E2 | R_CCW | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E2 | R_CW | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E2 | R_NoComm | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E3 | R_crop | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E3 | R_prefix | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E4 | P_final | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E4 | P_stopped | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | P_Static | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_NoComm | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_Static | 300 | 300 | 0 | 0.253333 | 0.435647 | 76 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |

### answer_support_precision

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.0842778 | 0.251062 | 25.2833 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E1 | P_CW | 300 | 300 | 0 | 0.0791667 | 0.245843 | 23.75 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.0508333 | 0.203176 | 15.25 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.0456667 | 0.189348 | 13.7 | 0 | 0 | 0 | 0 | 0 | 0.341667 | 1 |
| E1 | R_CCW | 300 | 300 | 0 | 0.400278 | 0.46524 | 120.083 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E1 | R_CW | 300 | 300 | 0 | 0.392722 | 0.46188 | 117.817 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E1 | R_NoComm | 300 | 300 | 0 | 0.386786 | 0.459577 | 116.036 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0.38 | 0.457907 | 114 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E2 | R_CCW | 300 | 300 | 0 | 0.400278 | 0.46524 | 120.083 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E2 | R_CW | 300 | 300 | 0 | 0.392722 | 0.46188 | 117.817 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E2 | R_NoComm | 300 | 300 | 0 | 0.386786 | 0.459577 | 116.036 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E3 | R_crop | 300 | 300 | 0 | 0.370556 | 0.464459 | 111.167 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E3 | R_prefix | 300 | 300 | 0 | 0.370556 | 0.464459 | 111.167 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E4 | P_final | 300 | 300 | 0 | 0.0959444 | 0.273347 | 28.7833 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E4 | P_stopped | 300 | 300 | 0 | 0.0952778 | 0.271033 | 28.5833 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.153833 | 0.341191 | 46.15 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.0915 | 0.270185 | 27.45 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.0745556 | 0.246191 | 22.3667 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.152611 | 0.340865 | 45.7833 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.0636111 | 0.227284 | 19.0833 | 0 | 0 | 0 | 0 | 0 | 0.670833 | 1 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0.0598333 | 0.220327 | 17.95 | 0 | 0 | 0 | 0 | 0 | 0.666667 | 1 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.154 | 0.341226 | 46.2 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.0420556 | 0.17657 | 12.6167 | 0 | 0 | 0 | 0 | 0 | 0.254167 | 1 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0.0597222 | 0.220159 | 17.9167 | 0 | 0 | 0 | 0 | 0 | 0.666667 | 1 |
| E5 | P_Static | 300 | 300 | 0 | 0.0577778 | 0.218085 | 17.3333 | 0 | 0 | 0 | 0 | 0 | 0.508333 | 1 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0.38375 | 0.459843 | 115.125 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0.396574 | 0.464806 | 118.972 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0.378796 | 0.4552 | 113.639 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0.386806 | 0.462042 | 116.042 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 0.389185 | 0.45972 | 116.756 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0.372963 | 0.451059 | 111.889 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 0.376852 | 0.458364 | 113.056 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_NoComm | 300 | 300 | 0 | 0.369907 | 0.451909 | 110.972 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0.369008 | 0.454989 | 110.702 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| E5 | R_Static | 300 | 300 | 0 | 0.37463 | 0.455266 | 112.389 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |

### answer_support_recall

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.0693889 | 0.202642 | 20.8167 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E1 | P_CW | 300 | 300 | 0 | 0.0599444 | 0.181016 | 17.9833 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.0391667 | 0.152911 | 11.75 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.0375 | 0.151194 | 11.25 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E1 | R_CCW | 300 | 300 | 0 | 0.268667 | 0.337474 | 80.6 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E1 | R_CW | 300 | 300 | 0 | 0.268 | 0.341668 | 80.4 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E1 | R_NoComm | 300 | 300 | 0 | 0.254944 | 0.327931 | 76.4833 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0.259389 | 0.335328 | 77.8167 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E2 | R_CCW | 300 | 300 | 0 | 0.268667 | 0.337474 | 80.6 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E2 | R_CW | 300 | 300 | 0 | 0.268 | 0.341668 | 80.4 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E2 | R_NoComm | 300 | 300 | 0 | 0.254944 | 0.327931 | 76.4833 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E3 | R_crop | 300 | 300 | 0 | 0.239278 | 0.327761 | 71.7833 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E3 | R_prefix | 300 | 300 | 0 | 0.239278 | 0.327761 | 71.7833 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E4 | P_final | 300 | 300 | 0 | 0.0755 | 0.213348 | 22.65 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E4 | P_stopped | 300 | 300 | 0 | 0.0755 | 0.213348 | 22.65 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.109111 | 0.24756 | 32.7333 | 0 | 0 | 0 | 0 | 0 | 0.508333 | 1 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.0658333 | 0.19167 | 19.75 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.0511111 | 0.168685 | 15.3333 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.107444 | 0.246601 | 32.2333 | 0 | 0 | 0 | 0 | 0 | 0.508333 | 1 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.0491667 | 0.173533 | 14.75 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0.0419444 | 0.148149 | 12.5833 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.109111 | 0.24756 | 32.7333 | 0 | 0 | 0 | 0 | 0 | 0.508333 | 1 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.0352778 | 0.141595 | 10.5833 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0.0466667 | 0.171605 | 14 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E5 | P_Static | 300 | 300 | 0 | 0.0444444 | 0.165595 | 13.3333 | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0.268556 | 0.344684 | 80.5667 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0.267722 | 0.339467 | 80.3167 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0.258556 | 0.332394 | 77.5667 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0.266056 | 0.344102 | 79.8167 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 0.2655 | 0.336553 | 79.65 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0.254944 | 0.329344 | 76.4833 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 0.264111 | 0.344828 | 79.2333 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E5 | R_NoComm | 300 | 300 | 0 | 0.254667 | 0.332472 | 76.4 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0.253833 | 0.334259 | 76.15 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |
| E5 | R_Static | 300 | 300 | 0 | 0.259389 | 0.338293 | 77.8167 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 |

### cache_reads_this_execution

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 1.96667 | 1.23178 | 590 | 0 | 0 | 1 | 3 | 3 | 3 | 6 |
| E1 | P_CW | 300 | 300 | 0 | 0.106667 | 0.723672 | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.07 | 0.453636 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.126667 | 0.836433 | 38 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E1 | R_CCW | 300 | 300 | 0 | 2.44 | 1.20467 | 732 | 0 | 0 | 1 | 3 | 3 | 3 | 6 |
| E1 | R_CW | 300 | 300 | 0 | 0.15 | 0.888895 | 45 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E1 | R_NoComm | 300 | 300 | 0 | 0.09 | 0.512617 | 27 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0.173333 | 0.989893 | 52 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E2 | R_CCW | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E2 | R_CW | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E2 | R_NoComm | 300 | 300 | 0 | 3 | 0 | 900 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| E3 | R_crop | 300 | 300 | 0 | 3.99333 | 0.081513 | 1198 | 3 | 4 | 4 | 4 | 4 | 4 | 4 |
| E3 | R_prefix | 300 | 300 | 0 | 4 | 0 | 1200 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| E4 | P_final | 300 | 300 | 0 | 11.31 | 1.38091 | 3393 | 6 | 8 | 11 | 12 | 12 | 12 | 12 |
| E4 | P_stopped | 300 | 300 | 0 | 11.2033 | 1.66871 | 3361 | 3 | 7.95 | 11 | 12 | 12 | 12 | 12 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.0333333 | 0.390809 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.98 | 0.32653 | 294 | 0 | 1 | 1 | 1 | 1 | 1 | 4 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 1.00667 | 0.283237 | 302 | 0 | 1 | 1 | 1 | 1 | 1 | 4 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 2.99667 | 0.3322 | 899 | 0 | 3 | 3 | 3 | 3 | 3 | 6 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 1.07333 | 0.607607 | 322 | 0 | 0 | 1 | 1 | 1 | 2 | 4 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 1.43333 | 0.653436 | 430 | 0 | 0.95 | 1 | 1 | 2 | 2 | 5 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 2.97667 | 0.428253 | 893 | 0 | 3 | 3 | 3 | 3 | 3 | 6 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.0233333 | 0.250996 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0.0333333 | 0.390809 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E5 | P_Static | 300 | 300 | 0 | 0.0333333 | 0.373301 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0.0733333 | 0.623901 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0.936667 | 0.535673 | 281 | 0 | 0 | 1 | 1 | 1 | 1 | 4 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0.216667 | 0.500557 | 65 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0.743333 | 0.636768 | 223 | 0 | 0 | 0 | 1 | 1 | 1 | 5 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 1.46333 | 0.798403 | 439 | 0 | 0 | 1 | 2 | 2 | 2 | 5 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0.333333 | 0.719222 | 100 | 0 | 0 | 0 | 0 | 0 | 2 | 3 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 1.32 | 0.898308 | 396 | 0 | 0 | 1 | 1 | 2 | 2 | 5 |
| E5 | R_NoComm | 300 | 300 | 0 | 0.0433333 | 0.349087 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0.0666667 | 0.568595 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E5 | R_Static | 300 | 300 | 0 | 0.07 | 0.576954 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |

### em

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 6.33333 | 24.3968 | 1900 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E1 | P_CW | 300 | 300 | 0 | 6.66667 | 24.9861 | 2000 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E1 | P_NoComm | 300 | 300 | 0 | 3.66667 | 18.8256 | 1100 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 3.66667 | 18.8256 | 1100 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E1 | R_CCW | 300 | 300 | 0 | 24 | 42.7797 | 7200 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E1 | R_CW | 300 | 300 | 0 | 24.3333 | 42.9812 | 7300 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E1 | R_NoComm | 300 | 300 | 0 | 22.6667 | 41.9375 | 6800 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 23.6667 | 42.5746 | 7100 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E2 | R_CCW | 300 | 300 | 0 | 24 | 42.7797 | 7200 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E2 | R_CW | 300 | 300 | 0 | 24.3333 | 42.9812 | 7300 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E2 | R_NoComm | 300 | 300 | 0 | 22.6667 | 41.9375 | 6800 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E3 | R_crop | 300 | 300 | 0 | 23.3333 | 42.3659 | 7000 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E3 | R_prefix | 300 | 300 | 0 | 23.3333 | 42.3659 | 7000 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E4 | P_final | 300 | 300 | 0 | 7.66667 | 26.6506 | 2300 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E4 | P_stopped | 300 | 300 | 0 | 7.66667 | 26.6506 | 2300 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | P_Broadcast | 300 | 300 | 0 | 11.3333 | 31.7529 | 3400 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 6.66667 | 24.9861 | 2000 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 5.66667 | 23.1591 | 1700 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 11.3333 | 31.7529 | 3400 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 5 | 21.8309 | 1500 | 0 | 0 | 0 | 0 | 0 | 5 | 100 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 4.33333 | 20.3947 | 1300 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 11.3333 | 31.7529 | 3400 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | P_NoComm | 300 | 300 | 0 | 3.33333 | 17.9805 | 1000 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 4 | 19.6287 | 1200 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_Static | 300 | 300 | 0 | 4.33333 | 20.3947 | 1300 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | R_Broadcast | 300 | 300 | 0 | 23.3333 | 42.3659 | 7000 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 23.6667 | 42.5746 | 7100 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 22.6667 | 41.9375 | 6800 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 24.3333 | 42.9812 | 7300 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 23.6667 | 42.5746 | 7100 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 22.3333 | 41.7176 | 6700 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 23 | 42.1536 | 6900 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_NoComm | 300 | 300 | 0 | 23 | 42.1536 | 6900 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 22.6667 | 41.9375 | 6800 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_Static | 300 | 300 | 0 | 22.6667 | 41.9375 | 6800 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |

### explicit_empty_abstention

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | P_CW | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | R_CCW | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | R_CW | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | R_NoComm | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E2 | R_CCW | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E2 | R_CW | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E2 | R_NoComm | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E3 | R_crop | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E3 | R_prefix | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E4 | P_final | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E4 | P_stopped | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_Static | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_NoComm | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_Static | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

### f1

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 9.04001 | 26.7949 | 2712 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E1 | P_CW | 300 | 300 | 0 | 9.13297 | 27.1098 | 2739.89 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E1 | P_NoComm | 300 | 300 | 0 | 5.26671 | 21.0747 | 1580.01 | 0 | 0 | 0 | 0 | 0 | 50.3571 | 100 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 5.7556 | 21.6406 | 1726.68 | 0 | 0 | 0 | 0 | 0 | 66.6667 | 100 |
| E1 | R_CCW | 300 | 300 | 0 | 32.839 | 43.4603 | 9851.71 | 0 | 0 | 0 | 0 | 81.4286 | 100 | 100 |
| E1 | R_CW | 300 | 300 | 0 | 32.7325 | 43.5472 | 9819.75 | 0 | 0 | 0 | 0 | 82.7273 | 100 | 100 |
| E1 | R_NoComm | 300 | 300 | 0 | 31.7477 | 42.7223 | 9524.32 | 0 | 0 | 0 | 0 | 80 | 100 | 100 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 32.0889 | 43.0875 | 9626.68 | 0 | 0 | 0 | 0 | 80 | 100 | 100 |
| E2 | R_CCW | 300 | 300 | 0 | 32.839 | 43.4603 | 9851.71 | 0 | 0 | 0 | 0 | 81.4286 | 100 | 100 |
| E2 | R_CW | 300 | 300 | 0 | 32.7325 | 43.5472 | 9819.75 | 0 | 0 | 0 | 0 | 82.7273 | 100 | 100 |
| E2 | R_NoComm | 300 | 300 | 0 | 31.7477 | 42.7223 | 9524.32 | 0 | 0 | 0 | 0 | 80 | 100 | 100 |
| E3 | R_crop | 300 | 300 | 0 | 31.4904 | 43.1422 | 9447.11 | 0 | 0 | 0 | 0 | 80 | 100 | 100 |
| E3 | R_prefix | 300 | 300 | 0 | 31.6354 | 43.1551 | 9490.62 | 0 | 0 | 0 | 0 | 80 | 100 | 100 |
| E4 | P_final | 300 | 300 | 0 | 10.7717 | 29.0131 | 3231.52 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E4 | P_stopped | 300 | 300 | 0 | 10.7717 | 29.0131 | 3231.52 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | P_Broadcast | 300 | 300 | 0 | 14.992 | 33.8172 | 4497.6 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 9.16516 | 27.242 | 2749.55 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 7.52407 | 24.9552 | 2257.22 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 14.992 | 33.8172 | 4497.6 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 7.06692 | 23.9173 | 2120.08 | 0 | 0 | 0 | 0 | 0 | 81 | 100 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 6.57499 | 22.8962 | 1972.5 | 0 | 0 | 0 | 0 | 0 | 67.3333 | 100 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 14.992 | 33.8172 | 4497.6 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | P_NoComm | 300 | 300 | 0 | 5.05979 | 20.1553 | 1517.94 | 0 | 0 | 0 | 0 | 0 | 50 | 100 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 6.21606 | 22.2547 | 1864.82 | 0 | 0 | 0 | 0 | 0 | 66.8333 | 100 |
| E5 | P_Static | 300 | 300 | 0 | 6.42086 | 22.7623 | 1926.26 | 0 | 0 | 0 | 0 | 0 | 75.25 | 100 |
| E5 | R_Broadcast | 300 | 300 | 0 | 31.651 | 43.0231 | 9495.3 | 0 | 0 | 0 | 0 | 80 | 100 | 100 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 31.9659 | 42.9963 | 9589.76 | 0 | 0 | 0 | 0 | 80 | 100 | 100 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 31.7696 | 42.736 | 9530.88 | 0 | 0 | 0 | 0 | 80 | 100 | 100 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 32.5896 | 43.3842 | 9776.88 | 0 | 0 | 0 | 0 | 82.7273 | 100 | 100 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 32.3627 | 43.1583 | 9708.82 | 0 | 0 | 0 | 0 | 80 | 100 | 100 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 32.0562 | 42.6269 | 9616.86 | 0 | 0 | 0 | 0 | 80 | 100 | 100 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 32.1076 | 42.9463 | 9632.27 | 0 | 0 | 0 | 0 | 80 | 100 | 100 |
| E5 | R_NoComm | 300 | 300 | 0 | 32.314 | 42.8917 | 9694.21 | 0 | 0 | 0 | 0 | 80 | 100 | 100 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 31.8446 | 43.0341 | 9553.38 | 0 | 0 | 0 | 0 | 80 | 100 | 100 |
| E5 | R_Static | 300 | 300 | 0 | 31.3085 | 42.6688 | 9392.54 | 0 | 0 | 0 | 0 | 76.25 | 100 | 100 |

### input_tokens

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 3568.62 | 690.863 | 1.07059e+06 | 1125 | 2553.15 | 3129.5 | 3533 | 3966.25 | 4727.35 | 6747 |
| E1 | P_CW | 300 | 300 | 0 | 3568.14 | 690.514 | 1.07044e+06 | 1125 | 2553.15 | 3129.5 | 3533 | 3966.25 | 4727.35 | 6747 |
| E1 | P_NoComm | 300 | 300 | 0 | 1744.66 | 347.014 | 523398 | 549 | 1222.85 | 1522.75 | 1724 | 1931.25 | 2336.55 | 3360 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 3494.67 | 693.639 | 1.0484e+06 | 1101 | 2448.7 | 3047.75 | 3454.5 | 3882.5 | 4676.1 | 6723 |
| E1 | R_CCW | 300 | 300 | 0 | 8320.47 | 2083.6 | 2.49614e+06 | 1125 | 5310.6 | 6964.5 | 8226 | 9423 | 11916.9 | 18282 |
| E1 | R_CW | 300 | 300 | 0 | 8320.47 | 2083.6 | 2.49614e+06 | 1125 | 5310.6 | 6964.5 | 8226 | 9423 | 11916.9 | 18282 |
| E1 | R_NoComm | 300 | 300 | 0 | 4075.69 | 1041.33 | 1.22271e+06 | 549 | 2498.85 | 3406.5 | 4029 | 4612.5 | 5896.35 | 8940 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 8160.25 | 2082.35 | 2.44807e+06 | 1101 | 5000.7 | 6821.25 | 8067 | 9309 | 11798 | 17880 |
| E2 | R_CCW | 300 | 300 | 0 | 8320.47 | 2083.6 | 2.49614e+06 | 1125 | 5310.6 | 6964.5 | 8226 | 9423 | 11916.9 | 18282 |
| E2 | R_CW | 300 | 300 | 0 | 8320.47 | 2083.6 | 2.49614e+06 | 1125 | 5310.6 | 6964.5 | 8226 | 9423 | 11916.9 | 18282 |
| E2 | R_NoComm | 300 | 300 | 0 | 4075.69 | 1041.33 | 1.22271e+06 | 549 | 2498.85 | 3406.5 | 4029 | 4612.5 | 5896.35 | 8940 |
| E3 | R_crop | 300 | 300 | 0 | 5489.5 | 1387.49 | 1.64685e+06 | 741 | 3448.8 | 4590.75 | 5419 | 6266 | 7919.2 | 12054 |
| E3 | R_prefix | 300 | 300 | 0 | 5489.46 | 1387.51 | 1.64684e+06 | 741 | 3448.8 | 4590.75 | 5419 | 6266 | 7919.2 | 12054 |
| E4 | P_final | 300 | 300 | 0 | 7234.26 | 1387.42 | 2.17028e+06 | 2277 | 5186.9 | 6309 | 7137 | 8058.5 | 9482.4 | 13521 |
| E4 | P_stopped | 300 | 300 | 0 | 7127.53 | 1524.82 | 2.13826e+06 | 891 | 5003.4 | 6205.75 | 7077.5 | 8045 | 9482.4 | 13521 |
| E5 | P_Broadcast | 300 | 300 | 0 | 3641.55 | 700.414 | 1.09246e+06 | 1149 | 2638.3 | 3178.5 | 3608 | 4072.5 | 4755.6 | 6771 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 2385.85 | 488.527 | 715754 | 749 | 1721 | 2059.25 | 2345.5 | 2640.25 | 3264.75 | 4260 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 2383.1 | 512.393 | 714931 | 749 | 1663.25 | 2060.75 | 2332.5 | 2655 | 3258.4 | 5762 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 3622.06 | 715.12 | 1.08662e+06 | 1149 | 2621.8 | 3165 | 3592 | 4007 | 4755.6 | 6771 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 2863.9 | 631.325 | 859170 | 749 | 1962.3 | 2462.75 | 2808 | 3220.75 | 3993.6 | 6163 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 2878.12 | 624.761 | 863437 | 933 | 1904.95 | 2463.5 | 2832.5 | 3197.25 | 3951.95 | 6163 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 3622.76 | 714.598 | 1.08683e+06 | 1149 | 2624.95 | 3165 | 3592 | 4007 | 4755.6 | 6771 |
| E5 | P_NoComm | 300 | 300 | 0 | 1744.68 | 347.005 | 523403 | 549 | 1222.85 | 1522.75 | 1724 | 1931.25 | 2336.55 | 3360 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 3494.76 | 694.065 | 1.04843e+06 | 1101 | 2448.7 | 3047.75 | 3455.5 | 3882.5 | 4676.1 | 6723 |
| E5 | P_Static | 300 | 300 | 0 | 2972.76 | 608.833 | 891828 | 933 | 2030.95 | 2593.5 | 2943.5 | 3322.75 | 3907 | 6338 |
| E5 | R_Broadcast | 300 | 300 | 0 | 8484.37 | 2095.46 | 2.54531e+06 | 1149 | 5496.6 | 7138.5 | 8376 | 9543.75 | 12061.4 | 18684 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 5711.19 | 1502.28 | 1.71336e+06 | 749 | 3567.6 | 4723 | 5541 | 6568.25 | 8475.9 | 12188 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 4359.38 | 1230.94 | 1.30781e+06 | 749 | 2600.85 | 3504.75 | 4192.5 | 5025.25 | 6586.95 | 8940 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 6339.8 | 1727.24 | 1.90194e+06 | 1149 | 3942.5 | 5267.25 | 6123 | 7324.75 | 9668.75 | 12056 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 6634.5 | 1735.53 | 1.99035e+06 | 933 | 4052.8 | 5576.5 | 6529 | 7578 | 9427.1 | 15168 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 4575.1 | 1559.91 | 1.37253e+06 | 933 | 2600.85 | 3509.25 | 4239 | 5260.5 | 7529.35 | 10748 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 6630.83 | 1762.8 | 1.98925e+06 | 1149 | 4053.95 | 5527.75 | 6525.5 | 7545.25 | 9712.9 | 15036 |
| E5 | R_NoComm | 300 | 300 | 0 | 4076 | 1041.15 | 1.2228e+06 | 549 | 2498.85 | 3406.5 | 4029 | 4612.5 | 5896.35 | 8940 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 8159.83 | 2082.33 | 2.44795e+06 | 1101 | 5000.7 | 6821.25 | 8066.5 | 9308.25 | 11796.1 | 17880 |
| E5 | R_Static | 300 | 300 | 0 | 6904.03 | 1733.42 | 2.07121e+06 | 933 | 4333.6 | 5788.5 | 6826.5 | 7848 | 9897.45 | 15168 |

### invalid_citation_count

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.00333333 | 0.057735 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | P_CW | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | P_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | R_CCW | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | R_CW | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | R_NoComm | 300 | 300 | 0 | 0.00333333 | 0.057735 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E2 | R_CCW | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E2 | R_CW | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E2 | R_NoComm | 300 | 300 | 0 | 0.00333333 | 0.057735 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E3 | R_crop | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E3 | R_prefix | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E4 | P_final | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E4 | P_stopped | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.00333333 | 0.057735 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.01 | 0.128927 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0.0133333 | 0.163026 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.00333333 | 0.057735 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | P_Static | 300 | 300 | 0 | 0.00333333 | 0.057735 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0.00333333 | 0.057735 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0.00666667 | 0.11547 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0.00666667 | 0.11547 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0.0133333 | 0.23094 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 0.00333333 | 0.057735 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0.00666667 | 0.11547 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 0.00666667 | 0.11547 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| E5 | R_NoComm | 300 | 300 | 0 | 0.00666667 | 0.11547 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_Static | 300 | 300 | 0 | 0.00333333 | 0.057735 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

### label_joint_em

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.333333 | 5.7735 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E1 | P_CW | 300 | 300 | 0 | 0.333333 | 5.7735 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.666667 | 8.1513 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.333333 | 5.7735 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E1 | R_CCW | 300 | 300 | 0 | 6.33333 | 24.3968 | 1900 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E1 | R_CW | 300 | 300 | 0 | 6.33333 | 24.3968 | 1900 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E1 | R_NoComm | 300 | 300 | 0 | 5.33333 | 22.5073 | 1600 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 5 | 21.8309 | 1500 | 0 | 0 | 0 | 0 | 0 | 5 | 100 |
| E2 | R_CCW | 300 | 300 | 0 | 6.33333 | 24.3968 | 1900 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E2 | R_CW | 300 | 300 | 0 | 6.33333 | 24.3968 | 1900 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E2 | R_NoComm | 300 | 300 | 0 | 5.33333 | 22.5073 | 1600 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E3 | R_crop | 300 | 300 | 0 | 5.33333 | 22.5073 | 1600 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E3 | R_prefix | 300 | 300 | 0 | 5.33333 | 22.5073 | 1600 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E4 | P_final | 300 | 300 | 0 | 0.666667 | 8.1513 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E4 | P_stopped | 300 | 300 | 0 | 0.666667 | 8.1513 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_Broadcast | 300 | 300 | 0 | 2 | 14.0234 | 600 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.666667 | 8.1513 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.666667 | 8.1513 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 2 | 14.0234 | 600 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.666667 | 8.1513 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 2 | 14.0234 | 600 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.333333 | 5.7735 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0.333333 | 5.7735 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_Static | 300 | 300 | 0 | 0.666667 | 8.1513 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | R_Broadcast | 300 | 300 | 0 | 7.66667 | 26.6506 | 2300 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 6.66667 | 24.9861 | 2000 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 5.66667 | 23.1591 | 1700 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 6.33333 | 24.3968 | 1900 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 6 | 23.7884 | 1800 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 5.33333 | 22.5073 | 1600 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 7 | 25.5573 | 2100 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_NoComm | 300 | 300 | 0 | 5.66667 | 23.1591 | 1700 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 4.66667 | 21.1276 | 1400 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | R_Static | 300 | 300 | 0 | 6.66667 | 24.9861 | 2000 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |

### label_joint_f1

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 5.29194 | 17.0018 | 1587.58 | 0 | 0 | 0 | 0 | 0 | 50 | 100 |
| E1 | P_CW | 300 | 300 | 0 | 5.0012 | 16.1777 | 1500.36 | 0 | 0 | 0 | 0 | 0 | 50 | 100 |
| E1 | P_NoComm | 300 | 300 | 0 | 2.96092 | 12.9627 | 888.277 | 0 | 0 | 0 | 0 | 0 | 33.3333 | 100 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 3.1093 | 12.8206 | 932.789 | 0 | 0 | 0 | 0 | 0 | 30.8974 | 100 |
| E1 | R_CCW | 300 | 300 | 0 | 21.9904 | 32.5182 | 6597.12 | 0 | 0 | 0 | 0 | 44.697 | 100 | 100 |
| E1 | R_CW | 300 | 300 | 0 | 21.911 | 32.5819 | 6573.29 | 0 | 0 | 0 | 0 | 50 | 100 | 100 |
| E1 | R_NoComm | 300 | 300 | 0 | 20.9177 | 31.3717 | 6275.31 | 0 | 0 | 0 | 0 | 44.4444 | 100 | 100 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 21.2468 | 31.5218 | 6374.04 | 0 | 0 | 0 | 0 | 45.8333 | 81 | 100 |
| E2 | R_CCW | 300 | 300 | 0 | 21.9904 | 32.5182 | 6597.12 | 0 | 0 | 0 | 0 | 44.697 | 100 | 100 |
| E2 | R_CW | 300 | 300 | 0 | 21.911 | 32.5819 | 6573.29 | 0 | 0 | 0 | 0 | 50 | 100 | 100 |
| E2 | R_NoComm | 300 | 300 | 0 | 20.9177 | 31.3717 | 6275.31 | 0 | 0 | 0 | 0 | 44.4444 | 100 | 100 |
| E3 | R_crop | 300 | 300 | 0 | 21.2603 | 32.015 | 6378.08 | 0 | 0 | 0 | 0 | 45.8333 | 100 | 100 |
| E3 | R_prefix | 300 | 300 | 0 | 21.2603 | 32.015 | 6378.08 | 0 | 0 | 0 | 0 | 45.8333 | 100 | 100 |
| E4 | P_final | 300 | 300 | 0 | 6.32237 | 18.9192 | 1896.71 | 0 | 0 | 0 | 0 | 0 | 50.0926 | 100 |
| E4 | P_stopped | 300 | 300 | 0 | 6.30967 | 18.7808 | 1892.9 | 0 | 0 | 0 | 0 | 0 | 51.868 | 100 |
| E5 | P_Broadcast | 300 | 300 | 0 | 9.29634 | 23.1868 | 2788.9 | 0 | 0 | 0 | 0 | 0 | 66.6667 | 100 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 5.42389 | 17.6496 | 1627.17 | 0 | 0 | 0 | 0 | 0 | 50 | 100 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 4.41192 | 16.1087 | 1323.57 | 0 | 0 | 0 | 0 | 0 | 40 | 100 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 9.19555 | 23.1384 | 2758.67 | 0 | 0 | 0 | 0 | 0 | 66.6667 | 100 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 3.93303 | 15.0423 | 1179.91 | 0 | 0 | 0 | 0 | 0 | 38.1905 | 100 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 3.22618 | 12.6741 | 967.853 | 0 | 0 | 0 | 0 | 0 | 33.3333 | 80 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 9.31222 | 23.2017 | 2793.67 | 0 | 0 | 0 | 0 | 0 | 66.6667 | 100 |
| E5 | P_NoComm | 300 | 300 | 0 | 2.62917 | 11.4339 | 788.75 | 0 | 0 | 0 | 0 | 0 | 25 | 100 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 3.55284 | 13.8662 | 1065.85 | 0 | 0 | 0 | 0 | 0 | 33.3333 | 100 |
| E5 | P_Static | 300 | 300 | 0 | 3.62698 | 14.5532 | 1088.1 | 0 | 0 | 0 | 0 | 0 | 33.3333 | 100 |
| E5 | R_Broadcast | 300 | 300 | 0 | 21.8543 | 33.3752 | 6556.3 | 0 | 0 | 0 | 0 | 44.697 | 100 | 100 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 21.8346 | 32.5556 | 6550.38 | 0 | 0 | 0 | 0 | 46.5909 | 100 | 100 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 21.076 | 31.4271 | 6322.79 | 0 | 0 | 0 | 0 | 44.697 | 100 | 100 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 21.7631 | 32.7354 | 6528.92 | 0 | 0 | 0 | 0 | 50 | 100 | 100 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 21.4237 | 32.2057 | 6427.1 | 0 | 0 | 0 | 0 | 50 | 100 | 100 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 20.6911 | 31.0645 | 6207.32 | 0 | 0 | 0 | 0 | 44.4444 | 100 | 100 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 21.7919 | 32.7611 | 6537.56 | 0 | 0 | 0 | 0 | 46.5909 | 100 | 100 |
| E5 | R_NoComm | 300 | 300 | 0 | 21.1767 | 31.4434 | 6353.01 | 0 | 0 | 0 | 0 | 46.5909 | 100 | 100 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 21.1463 | 31.7045 | 6343.88 | 0 | 0 | 0 | 0 | 44.4444 | 85.7143 | 100 |
| E5 | R_Static | 300 | 300 | 0 | 21.2475 | 32.4167 | 6374.26 | 0 | 0 | 0 | 0 | 50 | 100 | 100 |

### logical_calls

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E1 | P_CW | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E1 | P_NoComm | 300 | 300 | 0 | 3 | 0 | 900 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E1 | R_CCW | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E1 | R_CW | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E1 | R_NoComm | 300 | 300 | 0 | 3 | 0 | 900 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E2 | R_CCW | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E2 | R_CW | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E2 | R_NoComm | 300 | 300 | 0 | 3 | 0 | 900 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| E3 | R_crop | 300 | 300 | 0 | 4 | 0 | 1200 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| E3 | R_prefix | 300 | 300 | 0 | 4 | 0 | 1200 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| E4 | P_final | 300 | 300 | 0 | 12 | 0 | 3600 | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| E4 | P_stopped | 300 | 300 | 0 | 11.82 | 1.08236 | 3546 | 3 | 12 | 12 | 12 | 12 | 12 | 12 |
| E5 | P_Broadcast | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 4 | 0 | 1200 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 4 | 0 | 1200 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 5.97667 | 0.171913 | 1793 | 4 | 6 | 6 | 6 | 6 | 6 | 6 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 4.83333 | 0.373301 | 1450 | 4 | 4 | 5 | 5 | 5 | 5 | 5 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 4.85667 | 0.350998 | 1457 | 4 | 4 | 5 | 5 | 5 | 5 | 5 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 5.97667 | 0.171913 | 1793 | 4 | 6 | 6 | 6 | 6 | 6 | 6 |
| E5 | P_NoComm | 300 | 300 | 0 | 3 | 0 | 900 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E5 | P_Static | 300 | 300 | 0 | 5 | 0 | 1500 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| E5 | R_Broadcast | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 4.13 | 0.336865 | 1239 | 4 | 4 | 4 | 4 | 4 | 5 | 5 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 3.19667 | 0.398142 | 959 | 3 | 3 | 3 | 3 | 3 | 4 | 4 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 4.60667 | 0.496094 | 1382 | 4 | 4 | 4 | 5 | 5 | 5 | 6 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 4.81 | 0.392956 | 1443 | 4 | 4 | 5 | 5 | 5 | 5 | 5 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 3.35667 | 0.742491 | 1007 | 3 | 3 | 3 | 3 | 3 | 5 | 5 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 4.81667 | 0.396121 | 1445 | 4 | 4 | 5 | 5 | 5 | 5 | 6 |
| E5 | R_NoComm | 300 | 300 | 0 | 3 | 0 | 900 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 6 | 0 | 1800 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| E5 | R_Static | 300 | 300 | 0 | 5 | 0 | 1500 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |

### new_support_facts_delivered

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.45 | 0.644633 | 135 | 0 | 0 | 0 | 0 | 1 | 2 | 3 |
| E1 | P_CW | 300 | 300 | 0 | 0.45 | 0.644633 | 135 | 0 | 0 | 0 | 0 | 1 | 2 | 3 |
| E1 | P_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | R_CCW | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | R_CW | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | R_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E2 | R_CCW | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E2 | R_CW | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E2 | R_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E3 | R_crop | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E3 | R_prefix | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E4 | P_final | 300 | 300 | 0 | 0.603333 | 0.98449 | 181 | 0 | 0 | 0 | 0 | 1 | 2 | 6 |
| E4 | P_stopped | 300 | 300 | 0 | 0.576667 | 0.945393 | 173 | 0 | 0 | 0 | 0 | 1 | 2 | 6 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.853333 | 1.26378 | 256 | 0 | 0 | 0 | 0 | 2 | 4 | 6 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.343333 | 0.565557 | 103 | 0 | 0 | 0 | 0 | 1 | 1 | 3 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.256667 | 0.508249 | 77 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.813333 | 1.2478 | 244 | 0 | 0 | 0 | 0 | 2 | 4 | 6 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.28 | 0.573903 | 84 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0.246667 | 0.529235 | 74 | 0 | 0 | 0 | 0 | 0 | 1 | 2 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.816667 | 1.24696 | 245 | 0 | 0 | 0 | 0 | 2 | 4 | 6 |
| E5 | P_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | P_Static | 300 | 300 | 0 | 0.28 | 0.750652 | 84 | 0 | 0 | 0 | 0 | 0 | 2 | 4 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_Static | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### noanswer

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.833333 | 0.373301 | 250 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E1 | P_CW | 300 | 300 | 0 | 0.85 | 0.357668 | 255 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.906667 | 0.291385 | 272 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.893333 | 0.309205 | 268 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E1 | R_CCW | 300 | 300 | 0 | 0.5 | 0.500835 | 150 | 0 | 0 | 0 | 0.5 | 1 | 1 | 1 |
| E1 | R_CW | 300 | 300 | 0 | 0.513333 | 0.500657 | 154 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E1 | R_NoComm | 300 | 300 | 0 | 0.503333 | 0.500824 | 151 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0.503333 | 0.500824 | 151 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E2 | R_CCW | 300 | 300 | 0 | 0.5 | 0.500835 | 150 | 0 | 0 | 0 | 0.5 | 1 | 1 | 1 |
| E2 | R_CW | 300 | 300 | 0 | 0.513333 | 0.500657 | 154 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E2 | R_NoComm | 300 | 300 | 0 | 0.503333 | 0.500824 | 151 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E3 | R_crop | 300 | 300 | 0 | 0.523333 | 0.50029 | 157 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E3 | R_prefix | 300 | 300 | 0 | 0.523333 | 0.50029 | 157 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E4 | P_final | 300 | 300 | 0 | 0.826667 | 0.379168 | 248 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E4 | P_stopped | 300 | 300 | 0 | 0.826667 | 0.379168 | 248 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.78 | 0.414938 | 234 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.853333 | 0.354364 | 256 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.873333 | 0.333155 | 262 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.78 | 0.414938 | 234 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.866667 | 0.340503 | 260 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0.886667 | 0.317529 | 266 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.78 | 0.414938 | 234 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.9 | 0.300501 | 270 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0.88 | 0.325504 | 264 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_Static | 300 | 300 | 0 | 0.876667 | 0.329369 | 263 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0.516667 | 0.500557 | 155 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0.503333 | 0.500824 | 151 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0.503333 | 0.500824 | 151 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0.51 | 0.500735 | 153 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 0.503333 | 0.500824 | 151 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0.5 | 0.500835 | 150 | 0 | 0 | 0 | 0.5 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 0.51 | 0.500735 | 153 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E5 | R_NoComm | 300 | 300 | 0 | 0.503333 | 0.500824 | 151 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0.513333 | 0.500657 | 154 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| E5 | R_Static | 300 | 300 | 0 | 0.516667 | 0.500557 | 155 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |

### output_tokens

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 294 | 6 | 1397.93 | 597.425 | 410992 | 497 | 740.65 | 961.75 | 1241 | 1603.75 | 2678.75 | 3798 |
| E1 | P_CW | 300 | 295 | 5 | 1405.48 | 601.206 | 414618 | 497 | 740.7 | 965 | 1236 | 1627 | 2776.9 | 3618 |
| E1 | P_NoComm | 300 | 295 | 5 | 670.22 | 325.809 | 197715 | 239 | 340.1 | 448 | 583 | 801.5 | 1333.7 | 2436 |
| E1 | P_SelfUpdate | 300 | 295 | 5 | 1303.32 | 589.254 | 384478 | 526 | 668.4 | 909.5 | 1158 | 1498 | 2528.7 | 4735 |
| E1 | R_CCW | 300 | 299 | 1 | 1672.17 | 858.04 | 499978 | 497 | 804 | 1106 | 1431 | 1890 | 3381.4 | 5674 |
| E1 | R_CW | 300 | 299 | 1 | 1673.19 | 867.158 | 500285 | 497 | 804 | 1110 | 1424 | 1925 | 3362.9 | 5816 |
| E1 | R_NoComm | 300 | 299 | 1 | 845.88 | 506.613 | 252918 | 239 | 375.8 | 523 | 700 | 999 | 1738.8 | 3942 |
| E1 | R_SelfUpdate | 300 | 299 | 1 | 1647.58 | 873.493 | 492625 | 526 | 755.5 | 1044.5 | 1444 | 1940.5 | 3332.6 | 4967 |
| E2 | R_CCW | 300 | 299 | 1 | 1672.17 | 858.04 | 499978 | 497 | 804 | 1106 | 1431 | 1890 | 3381.4 | 5674 |
| E2 | R_CW | 300 | 299 | 1 | 1673.19 | 867.158 | 500285 | 497 | 804 | 1110 | 1424 | 1925 | 3362.9 | 5816 |
| E2 | R_NoComm | 300 | 299 | 1 | 845.88 | 506.613 | 252918 | 239 | 375.8 | 523 | 700 | 999 | 1738.8 | 3942 |
| E3 | R_crop | 300 | 299 | 1 | 1118.23 | 619.593 | 334351 | 345 | 521.6 | 722 | 942 | 1319 | 2256.2 | 4326 |
| E3 | R_prefix | 300 | 299 | 1 | 1118.95 | 621.411 | 334566 | 345 | 521.6 | 722 | 942 | 1319 | 2256.2 | 4326 |
| E4 | P_final | 300 | 295 | 5 | 2872.72 | 1199.57 | 847451 | 1013 | 1552.4 | 1992 | 2471 | 3429.5 | 5530.9 | 6950 |
| E4 | P_stopped | 300 | 295 | 5 | 2817.98 | 1209.95 | 831304 | 498 | 1479.7 | 1985.5 | 2441 | 3371 | 5530.9 | 6950 |
| E5 | P_Broadcast | 300 | 295 | 5 | 1445.71 | 649.466 | 426484 | 513 | 732.3 | 987 | 1269 | 1739 | 2694.3 | 4837 |
| E5 | P_Learned_seed19 | 300 | 295 | 5 | 925.539 | 427.593 | 273034 | 342 | 476.4 | 626.5 | 807 | 1110.5 | 1648 | 3403 |
| E5 | P_Learned_seed31 | 300 | 295 | 5 | 933.186 | 454.216 | 275290 | 342 | 462.1 | 618.5 | 804 | 1126.5 | 1757.7 | 3403 |
| E5 | P_Learned_seed7 | 300 | 295 | 5 | 1435.35 | 638.686 | 423427 | 513 | 732.3 | 986.5 | 1260 | 1732.5 | 2651.6 | 4837 |
| E5 | P_MatchedRandom_seed19 | 300 | 295 | 5 | 1132.72 | 534.039 | 334153 | 342 | 582 | 767.5 | 966 | 1371 | 2187.1 | 3698 |
| E5 | P_MatchedRandom_seed31 | 300 | 295 | 5 | 1124.78 | 517.733 | 331811 | 444 | 586.2 | 780.5 | 982 | 1336 | 2132.9 | 3990 |
| E5 | P_MatchedRandom_seed7 | 300 | 295 | 5 | 1437.31 | 639.671 | 424005 | 513 | 732.3 | 986 | 1263 | 1732.5 | 2651.6 | 4837 |
| E5 | P_NoComm | 300 | 295 | 5 | 671.068 | 342.386 | 197965 | 251 | 334.7 | 438.5 | 567 | 807 | 1246.1 | 2980 |
| E5 | P_SelfUpdate | 300 | 295 | 5 | 1300.23 | 595.516 | 383569 | 500 | 671.7 | 895.5 | 1114 | 1523.5 | 2477 | 4651 |
| E5 | P_Static | 300 | 295 | 5 | 1170.99 | 544.499 | 345442 | 413 | 604.7 | 785 | 1024 | 1375 | 2262 | 3990 |
| E5 | R_Broadcast | 300 | 298 | 2 | 1674.2 | 838.078 | 498912 | 513 | 838.1 | 1113.75 | 1424 | 1914.5 | 3534.7 | 5700 |
| E5 | R_Learned_seed19 | 300 | 298 | 2 | 1171.55 | 636.154 | 349121 | 342 | 532.35 | 732 | 976 | 1396.5 | 2588.45 | 3729 |
| E5 | R_Learned_seed31 | 300 | 299 | 1 | 921.318 | 572.667 | 275474 | 294 | 379.9 | 548.5 | 765 | 1064.5 | 2172.1 | 3538 |
| E5 | R_Learned_seed7 | 300 | 298 | 2 | 1310.14 | 699.695 | 390422 | 456 | 575.95 | 844.25 | 1153.5 | 1492.75 | 2748.35 | 4642 |
| E5 | R_MatchedRandom_seed19 | 300 | 299 | 1 | 1373.67 | 731.361 | 410726 | 419 | 625.9 | 880.5 | 1167 | 1598.5 | 2791.3 | 4522 |
| E5 | R_MatchedRandom_seed31 | 300 | 299 | 1 | 982.809 | 677.734 | 293860 | 294 | 380.9 | 556.5 | 783 | 1105 | 2467.2 | 4555 |
| E5 | R_MatchedRandom_seed7 | 300 | 298 | 2 | 1370.79 | 723.018 | 408496 | 384 | 591.4 | 878.75 | 1168 | 1644.5 | 2838.8 | 4779 |
| E5 | R_NoComm | 300 | 299 | 1 | 856.739 | 483.296 | 256165 | 251 | 375.6 | 537 | 725 | 983 | 1908.9 | 3472 |
| E5 | R_SelfUpdate | 300 | 299 | 1 | 1677.94 | 908.226 | 501703 | 500 | 751 | 1055.5 | 1453 | 1946 | 3617.5 | 5952 |
| E5 | R_Static | 300 | 299 | 1 | 1410.21 | 711.437 | 421654 | 413 | 676.6 | 917.5 | 1230 | 1615 | 2849 | 4339 |

### payload_tokens

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 66.8 | 64.2865 | 20040 | 19 | 21 | 21 | 21 | 95.25 | 198.3 | 391 |
| E1 | P_CW | 300 | 300 | 0 | 66.8 | 64.2865 | 20040 | 19 | 21 | 21 | 21 | 95.25 | 198.3 | 391 |
| E1 | P_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | R_CCW | 300 | 300 | 0 | 145.163 | 145.791 | 43549 | 18 | 21 | 21 | 100.5 | 234.75 | 404.05 | 768 |
| E1 | R_CW | 300 | 300 | 0 | 145.163 | 145.791 | 43549 | 18 | 21 | 21 | 100.5 | 234.75 | 404.05 | 768 |
| E1 | R_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E2 | R_CCW | 300 | 300 | 0 | 145.163 | 145.791 | 43549 | 18 | 21 | 21 | 100.5 | 234.75 | 404.05 | 768 |
| E2 | R_CW | 300 | 300 | 0 | 145.163 | 145.791 | 43549 | 18 | 21 | 21 | 100.5 | 234.75 | 404.05 | 768 |
| E2 | R_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E3 | R_crop | 300 | 300 | 0 | 47.2767 | 49.5277 | 14183 | 6 | 7 | 7 | 7 | 76.5 | 132.15 | 256 |
| E3 | R_prefix | 300 | 300 | 0 | 47.2767 | 49.5277 | 14183 | 6 | 7 | 7 | 7 | 76.5 | 132.15 | 256 |
| E4 | P_final | 300 | 300 | 0 | 216.31 | 214.256 | 64893 | 56 | 63 | 63 | 119.5 | 299.25 | 663.2 | 1246 |
| E4 | P_stopped | 300 | 300 | 0 | 200.627 | 195.317 | 60188 | 0 | 63 | 63 | 63 | 279.5 | 562 | 1246 |
| E5 | P_Broadcast | 300 | 300 | 0 | 132.873 | 133.879 | 39862 | 40 | 42 | 42 | 42 | 176.5 | 408.4 | 834 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 47.3667 | 49.146 | 14210 | 13 | 14 | 14 | 14 | 71.25 | 135 | 256 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 41.6933 | 47.292 | 12508 | 12 | 14 | 14 | 14 | 64.25 | 135 | 256 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 125.087 | 118.347 | 37526 | 12 | 42 | 42 | 42 | 174 | 366.2 | 788 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 44.2033 | 51.4535 | 13261 | 12 | 14 | 14 | 14 | 66.75 | 135.25 | 256 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 41.5033 | 50.4956 | 12451 | 12 | 14 | 14 | 14 | 59 | 141.4 | 256 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 125.63 | 118.35 | 37689 | 40 | 42 | 42 | 42 | 174 | 366.2 | 788 |
| E5 | P_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | P_Static | 300 | 300 | 0 | 45.0467 | 65.1573 | 13514 | 12 | 14 | 14 | 14 | 14 | 196.3 | 308 |
| E5 | R_Broadcast | 300 | 300 | 0 | 293.293 | 293.909 | 87988 | 36 | 42 | 42 | 198 | 502.5 | 836.1 | 1536 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 86.96 | 80.8007 | 26088 | 12 | 14 | 14 | 71.5 | 128.25 | 250.25 | 256 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 18.8467 | 57.1646 | 5654 | 0 | 0 | 0 | 0 | 0 | 175.35 | 256 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 74.3433 | 69.7888 | 22303 | 12 | 14 | 14 | 58 | 128 | 200.5 | 396 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 86.3 | 81.4705 | 25890 | 12 | 14 | 14 | 71 | 128 | 256 | 256 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 17.3967 | 54.5951 | 5219 | 0 | 0 | 0 | 0 | 0 | 138.7 | 256 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 74.6233 | 67.521 | 22387 | 12 | 14 | 14 | 67.5 | 128 | 200.7 | 332 |
| E5 | R_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_Static | 300 | 300 | 0 | 95.28 | 99.0053 | 28584 | 12 | 14 | 14 | 14 | 157 | 270 | 512 |

### provider_refusal_calls

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.0633333 | 0.439659 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| E1 | P_CW | 300 | 300 | 0 | 0.05 | 0.409951 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.0233333 | 0.171913 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.0566667 | 0.432913 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| E1 | R_CCW | 300 | 300 | 0 | 0.0466667 | 0.48159 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E1 | R_CW | 300 | 300 | 0 | 0.0466667 | 0.48159 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E1 | R_NoComm | 300 | 300 | 0 | 0.0166667 | 0.207845 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0.0433333 | 0.456966 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E2 | R_CCW | 300 | 300 | 0 | 0.0466667 | 0.48159 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E2 | R_CW | 300 | 300 | 0 | 0.0466667 | 0.48159 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E2 | R_NoComm | 300 | 300 | 0 | 0.0166667 | 0.207845 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E3 | R_crop | 300 | 300 | 0 | 0.0266667 | 0.293672 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| E3 | R_prefix | 300 | 300 | 0 | 0.0266667 | 0.293672 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| E4 | P_final | 300 | 300 | 0 | 0.103333 | 0.857383 | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |
| E4 | P_stopped | 300 | 300 | 0 | 0.103333 | 0.857383 | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.05 | 0.409951 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.0333333 | 0.269191 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.0366667 | 0.286815 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.0466667 | 0.380752 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.04 | 0.324681 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0.0466667 | 0.353419 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.0466667 | 0.380752 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.0266667 | 0.198545 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0.06 | 0.436312 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| E5 | P_Static | 300 | 300 | 0 | 0.05 | 0.357668 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0.05 | 0.518068 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0.0333333 | 0.373301 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0.02 | 0.244539 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0.0366667 | 0.386203 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 0.0433333 | 0.426688 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0.02 | 0.244539 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 0.0366667 | 0.411363 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| E5 | R_NoComm | 300 | 300 | 0 | 0.02 | 0.244539 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0.0433333 | 0.464228 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| E5 | R_Static | 300 | 300 | 0 | 0.04 | 0.42308 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |

### provider_refusal_count

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.04 | 0.280468 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E1 | P_CW | 300 | 300 | 0 | 0.0266667 | 0.257247 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.0233333 | 0.171913 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.0333333 | 0.269191 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E1 | R_CCW | 300 | 300 | 0 | 0.03 | 0.298995 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E1 | R_CW | 300 | 300 | 0 | 0.03 | 0.298995 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E1 | R_NoComm | 300 | 300 | 0 | 0.0166667 | 0.207845 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0.0266667 | 0.257247 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E2 | R_CCW | 300 | 300 | 0 | 0.03 | 0.298995 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E2 | R_CW | 300 | 300 | 0 | 0.03 | 0.298995 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E2 | R_NoComm | 300 | 300 | 0 | 0.0166667 | 0.207845 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E3 | R_crop | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E3 | R_prefix | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E4 | P_final | 300 | 300 | 0 | 0.03 | 0.263308 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E4 | P_stopped | 300 | 300 | 0 | 0.03 | 0.263308 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.0233333 | 0.222758 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.0266667 | 0.198545 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.03 | 0.263308 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.0233333 | 0.222758 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.0266667 | 0.229779 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0.03 | 0.236544 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.0233333 | 0.222758 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.0266667 | 0.198545 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0.0333333 | 0.243076 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | P_Static | 300 | 300 | 0 | 0.03 | 0.236544 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0.03 | 0.298995 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0.0233333 | 0.250996 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0.02 | 0.244539 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0.0266667 | 0.269935 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 0.03 | 0.275717 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0.02 | 0.244539 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 0.0233333 | 0.250996 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_NoComm | 300 | 300 | 0 | 0.02 | 0.244539 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0.0233333 | 0.237297 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| E5 | R_Static | 300 | 300 | 0 | 0.0266667 | 0.269935 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |

### provider_refusal_question

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.0266667 | 0.161376 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | P_CW | 300 | 300 | 0 | 0.0133333 | 0.114889 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.02 | 0.140234 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.02 | 0.140234 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | R_CCW | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | R_CW | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | R_NoComm | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0.0133333 | 0.114889 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E2 | R_CCW | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E2 | R_CW | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E2 | R_NoComm | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E3 | R_crop | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E3 | R_prefix | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E4 | P_final | 300 | 300 | 0 | 0.0166667 | 0.128233 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E4 | P_stopped | 300 | 300 | 0 | 0.0166667 | 0.128233 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.0133333 | 0.114889 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.02 | 0.140234 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.0166667 | 0.128233 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.0133333 | 0.114889 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.0166667 | 0.128233 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0.02 | 0.140234 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.0133333 | 0.114889 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.02 | 0.140234 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0.0233333 | 0.151212 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | P_Static | 300 | 300 | 0 | 0.02 | 0.140234 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 0.0133333 | 0.114889 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_NoComm | 300 | 300 | 0 | 0.00666667 | 0.081513 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| E5 | R_Static | 300 | 300 | 0 | 0.01 | 0.099665 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

### split_support_across_nodes

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E1 | P_CW | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E1 | R_CCW | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | R_CW | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | R_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E2 | R_CCW | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E2 | R_CW | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E2 | R_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E3 | R_crop | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E3 | R_prefix | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E4 | P_final | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E4 | P_stopped | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | P_Static | 300 | 300 | 0 | 0.206667 | 0.405591 | 62 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_Static | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### support_available_recall

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E1 | P_CW | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E1 | R_CCW | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E1 | R_CW | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E1 | R_NoComm | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E2 | R_CCW | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E2 | R_CW | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E2 | R_NoComm | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E3 | R_crop | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E3 | R_prefix | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E4 | P_final | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E4 | P_stopped | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | P_Static | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | R_NoComm | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |
| E5 | R_Static | 300 | 300 | 0 | 0.535 | 0.342376 | 160.5 | 0 | 0 | 0.333333 | 0.5 | 1 | 1 | 1 |

### support_label_em

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 1 | 9.9665 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E1 | P_CW | 300 | 300 | 0 | 0.666667 | 8.1513 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.666667 | 8.1513 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.666667 | 8.1513 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E1 | R_CCW | 300 | 300 | 0 | 9.66667 | 29.5997 | 2900 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E1 | R_CW | 300 | 300 | 0 | 10 | 30.0501 | 3000 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E1 | R_NoComm | 300 | 300 | 0 | 8.66667 | 28.1816 | 2600 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 8.33333 | 27.6847 | 2500 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E2 | R_CCW | 300 | 300 | 0 | 9.66667 | 29.5997 | 2900 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E2 | R_CW | 300 | 300 | 0 | 10 | 30.0501 | 3000 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E2 | R_NoComm | 300 | 300 | 0 | 8.66667 | 28.1816 | 2600 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E3 | R_crop | 300 | 300 | 0 | 8.66667 | 28.1816 | 2600 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E3 | R_prefix | 300 | 300 | 0 | 8.66667 | 28.1816 | 2600 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E4 | P_final | 300 | 300 | 0 | 1.33333 | 11.4889 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E4 | P_stopped | 300 | 300 | 0 | 1 | 9.9665 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_Broadcast | 300 | 300 | 0 | 2.66667 | 16.1376 | 800 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.666667 | 8.1513 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.666667 | 8.1513 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 2.66667 | 16.1376 | 800 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 1 | 9.9665 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 2.66667 | 16.1376 | 800 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.333333 | 5.7735 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 1 | 9.9665 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | P_Static | 300 | 300 | 0 | 1.33333 | 11.4889 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| E5 | R_Broadcast | 300 | 300 | 0 | 10.6667 | 30.9205 | 3200 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 10 | 30.0501 | 3000 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 8.66667 | 28.1816 | 2600 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 10 | 30.0501 | 3000 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 9.66667 | 29.5997 | 2900 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 8.33333 | 27.6847 | 2500 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 10.6667 | 30.9205 | 3200 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_NoComm | 300 | 300 | 0 | 9 | 28.666 | 2700 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 8 | 27.1746 | 2400 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |
| E5 | R_Static | 300 | 300 | 0 | 10.3333 | 30.4902 | 3100 | 0 | 0 | 0 | 0 | 0 | 100 | 100 |

### support_label_f1

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 7.20238 | 20.7756 | 2160.71 | 0 | 0 | 0 | 0 | 0 | 66.6667 | 100 |
| E1 | P_CW | 300 | 300 | 0 | 6.43968 | 19.4213 | 1931.9 | 0 | 0 | 0 | 0 | 0 | 50.3571 | 100 |
| E1 | P_NoComm | 300 | 300 | 0 | 4.0627 | 15.6899 | 1218.81 | 0 | 0 | 0 | 0 | 0 | 50 | 100 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 3.77381 | 15.1968 | 1132.14 | 0 | 0 | 0 | 0 | 0 | 33.6667 | 100 |
| E1 | R_CCW | 300 | 300 | 0 | 30.9302 | 36.7444 | 9279.05 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E1 | R_CW | 300 | 300 | 0 | 30.5775 | 36.7953 | 9173.25 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E1 | R_NoComm | 300 | 300 | 0 | 29.4776 | 35.8243 | 8843.27 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 29.569 | 36.2071 | 8870.71 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E2 | R_CCW | 300 | 300 | 0 | 30.9302 | 36.7444 | 9279.05 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E2 | R_CW | 300 | 300 | 0 | 30.5775 | 36.7953 | 9173.25 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E2 | R_NoComm | 300 | 300 | 0 | 29.4776 | 35.8243 | 8843.27 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E3 | R_crop | 300 | 300 | 0 | 27.9778 | 36.1186 | 8393.33 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E3 | R_prefix | 300 | 300 | 0 | 27.9778 | 36.1186 | 8393.33 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E4 | P_final | 300 | 300 | 0 | 7.93254 | 22.1258 | 2379.76 | 0 | 0 | 0 | 0 | 0 | 66.6667 | 100 |
| E4 | P_stopped | 300 | 300 | 0 | 7.90079 | 21.9583 | 2370.24 | 0 | 0 | 0 | 0 | 0 | 66.6667 | 100 |
| E5 | P_Broadcast | 300 | 300 | 0 | 12.0437 | 26.5456 | 3613.1 | 0 | 0 | 0 | 0 | 0 | 67.0833 | 100 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 7.2119 | 20.7347 | 2163.57 | 0 | 0 | 0 | 0 | 0 | 66.6667 | 100 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 5.5627 | 18.0411 | 1668.81 | 0 | 0 | 0 | 0 | 0 | 50 | 100 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 11.9151 | 26.4956 | 3574.52 | 0 | 0 | 0 | 0 | 0 | 67.0833 | 100 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 5.22778 | 18.2659 | 1568.33 | 0 | 0 | 0 | 0 | 0 | 50 | 100 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 4.57381 | 16.1611 | 1372.14 | 0 | 0 | 0 | 0 | 0 | 50 | 80 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 12.0595 | 26.5569 | 3617.86 | 0 | 0 | 0 | 0 | 0 | 67.0833 | 100 |
| E5 | P_NoComm | 300 | 300 | 0 | 3.47381 | 13.9228 | 1042.14 | 0 | 0 | 0 | 0 | 0 | 33.3333 | 100 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 4.86111 | 17.5509 | 1458.33 | 0 | 0 | 0 | 0 | 0 | 50 | 100 |
| E5 | P_Static | 300 | 300 | 0 | 4.68889 | 17.4167 | 1406.67 | 0 | 0 | 0 | 0 | 0 | 50 | 100 |
| E5 | R_Broadcast | 300 | 300 | 0 | 30.3698 | 37.0993 | 9110.95 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 30.6367 | 36.7737 | 9191.01 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 29.4208 | 35.8972 | 8826.25 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 30.2389 | 37.0411 | 9071.67 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 30.2661 | 36.4273 | 9079.82 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 28.9272 | 35.5175 | 8678.15 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 29.797 | 37.0183 | 8939.1 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E5 | R_NoComm | 300 | 300 | 0 | 28.8716 | 35.8878 | 8661.48 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 28.7593 | 35.9709 | 8627.79 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |
| E5 | R_Static | 300 | 300 | 0 | 29.5422 | 36.6477 | 8862.67 | 0 | 0 | 0 | 0 | 66.6667 | 100 | 100 |

### usage_complete

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 300 | 0 | 0.98 | 0.140234 | 294 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E1 | P_CW | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E1 | P_NoComm | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E1 | P_SelfUpdate | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E1 | R_CCW | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E1 | R_CW | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E1 | R_NoComm | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E1 | R_SelfUpdate | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E2 | R_CCW | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E2 | R_CW | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E2 | R_NoComm | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E3 | R_crop | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E3 | R_prefix | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E4 | P_final | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E4 | P_stopped | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_NoComm | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | P_Static | 300 | 300 | 0 | 0.983333 | 0.128233 | 295 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | R_Broadcast | 300 | 300 | 0 | 0.993333 | 0.081513 | 298 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | 0.993333 | 0.081513 | 298 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | 0.993333 | 0.081513 | 298 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | 0.993333 | 0.081513 | 298 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | R_NoComm | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| E5 | R_Static | 300 | 300 | 0 | 0.996667 | 0.057735 | 299 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |

### utility_gain

| experiment | condition | n | known_n | missing_n | mean | std | sum_known | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | P_CCW | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E1 | P_CW | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E1 | P_NoComm | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E1 | P_SelfUpdate | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E1 | R_CCW | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E1 | R_CW | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E1 | R_NoComm | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E1 | R_SelfUpdate | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E2 | R_CCW | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E2 | R_CW | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E2 | R_NoComm | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E3 | R_crop | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E3 | R_prefix | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E4 | P_final | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E4 | P_stopped | 300 | 0 | 300 | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) | NA (unknown) |
| E5 | P_Broadcast | 300 | 300 | 0 | 0.0293718 | 0.294331 | 8.81154 | -0.57 | -0.07 | -0.07 | -0.07 | -0.07 | 0.93 | 0.93 |
| E5 | P_Learned_seed19 | 300 | 300 | 0 | 0.0167093 | 0.212579 | 5.01279 | -1.01637 | -0.0300633 | -0.0287068 | -0.0233333 | -0.0233333 | 0.485914 | 0.971574 |
| E5 | P_Learned_seed31 | 300 | 300 | 0 | 0.00172577 | 0.190323 | 0.51773 | -1.01946 | -0.0300173 | -0.0233333 | -0.0233333 | -0.0233333 | -0.00894466 | 0.971574 |
| E5 | P_Learned_seed7 | 300 | 300 | 0 | 0.0302918 | 0.294091 | 9.08755 | -0.57 | -0.07 | -0.07 | -0.07 | -0.07 | 0.93 | 0.93 |
| E5 | P_MatchedRandom_seed19 | 300 | 300 | 0 | -0.00897388 | 0.163371 | -2.69216 | -1.02613 | -0.036922 | -0.03 | -0.03 | -0.0233333 | -0.0121456 | 0.971351 |
| E5 | P_MatchedRandom_seed31 | 300 | 300 | 0 | -0.0132547 | 0.168173 | -3.97641 | -1.03696 | -0.0368527 | -0.03 | -0.03 | -0.0233333 | -0.0158226 | 0.970345 |
| E5 | P_MatchedRandom_seed7 | 300 | 300 | 0 | 0.0301932 | 0.29411 | 9.05796 | -0.57 | -0.07 | -0.07 | -0.07 | -0.07 | 0.93 | 0.93 |
| E5 | P_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | P_SelfUpdate | 300 | 300 | 0 | -0.00843728 | 0.120179 | -2.53118 | -0.591429 | -0.02 | -0.02 | -0.02 | -0.02 | -0.02 | 0.98 |
| E5 | P_Static | 300 | 300 | 0 | -0.0166011 | 0.181062 | -4.98032 | -1.03617 | -0.0558105 | -0.03 | -0.03 | -0.0189785 | -0.0154205 | 0.94963 |
| E5 | R_Broadcast | 300 | 300 | 0 | -0.0765023 | 0.157391 | -22.9507 | -1.07 | -0.07 | -0.07 | -0.07 | -0.07 | -0.07 | 0.93 |
| E5 | R_Learned_seed19 | 300 | 300 | 0 | -0.0271419 | 0.13951 | -8.14256 | -1.03 | -0.03 | -0.0233333 | -0.0233333 | -0.0233333 | -0.0195021 | 0.970907 |
| E5 | R_Learned_seed31 | 300 | 300 | 0 | -0.0100163 | 0.089569 | -3.00489 | -1.01893 | -0.0233395 | 0 | 0 | 0 | 0 | 0.643394 |
| E5 | R_Learned_seed7 | 300 | 300 | 0 | -0.0229094 | 0.165311 | -6.87281 | -1.03426 | -0.03 | -0.03 | -0.03 | -0.0214307 | -0.0120616 | 0.980734 |
| E5 | R_MatchedRandom_seed19 | 300 | 300 | 0 | -0.0275457 | 0.164525 | -8.26371 | -1.03695 | -0.0330201 | -0.03 | -0.03 | -0.0233333 | -0.0177411 | 0.979375 |
| E5 | R_MatchedRandom_seed31 | 300 | 300 | 0 | -0.00801553 | 0.104192 | -2.40466 | -1.02579 | -0.03 | 0 | 0 | 0 | 0 | 0.964633 |
| E5 | R_MatchedRandom_seed7 | 300 | 300 | 0 | -0.0293195 | 0.11851 | -8.79584 | -1.03 | -0.0301498 | -0.03 | -0.03 | -0.0233333 | -0.0180017 | 0.965313 |
| E5 | R_NoComm | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E5 | R_SelfUpdate | 300 | 300 | 0 | -0.0246943 | 0.144187 | -7.40828 | -1.02 | -0.02 | -0.02 | -0.02 | -0.02 | -0.02 | 0.98 |
| E5 | R_Static | 300 | 300 | 0 | -0.0398283 | 0.143657 | -11.9485 | -1.03769 | -0.0363652 | -0.03 | -0.03 | -0.03 | -0.0205592 | 0.948531 |

## Every original report field

[Full-precision field register](tables/all_original_report_fields.csv) retains all pooled/stratified effects, heterogeneity checks, action frequencies, diagnostics and metadata without selectively omitting inconvenient contrasts. The entire original report tree is copied under `original_reports/`.

### original_reports/controlled/E1/gold_strata/statistics.json

```json
{
  "status": "COMPLETE",
  "experiment": "E1",
  "n": 300,
  "counts": {
    "Disjoint-Gold": 62,
    "Co-located-Gold": 14,
    "Retrieval-Missing-Gold": 224,
    "Undefined-Gold": 0
  },
  "effects": [
    {
      "contrast": "R_CW - NoComm",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": -1.8894009216589862,
      "ci95": [
        -6.08294930875576,
        0.7373271889400921
      ],
      "randomization_p": 0.5005499450054994,
      "helped": 1,
      "harmed": 2,
      "unchanged": 59,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CW - NoComm",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 7.5396825396825395,
      "ci95": [
        0.0,
        22.22222222222222
      ],
      "randomization_p": 0.5010498950104989,
      "helped": 2,
      "harmed": 0,
      "unchanged": 12,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CW - NoComm",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 1.3705998619338955,
      "ci95": [
        -0.4952389573253544,
        3.5278331624510706
      ],
      "randomization_p": 0.19488051194880512,
      "helped": 11,
      "harmed": 4,
      "unchanged": 209,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CW - NoComm",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "R_CW - SelfUpdate",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": -1.8894009216589862,
      "ci95": [
        -6.08294930875576,
        0.7373271889400921
      ],
      "randomization_p": 0.5005499450054994,
      "helped": 1,
      "harmed": 2,
      "unchanged": 59,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CW - SelfUpdate",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": -0.24943310657596335,
      "ci95": [
        -1.938775510204081,
        1.190476190476191
      ],
      "randomization_p": 1.0,
      "helped": 1,
      "harmed": 1,
      "unchanged": 12,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CW - SelfUpdate",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 1.4004754216570325,
      "ci95": [
        -0.22866796482392984,
        3.2880709109574995
      ],
      "randomization_p": 0.1344865513448655,
      "helped": 11,
      "harmed": 3,
      "unchanged": 210,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CW - SelfUpdate",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "R_CCW - NoComm",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": -1.2442396313364055,
      "ci95": [
        -4.838709677419355,
        1.105990783410138
      ],
      "randomization_p": 1.0,
      "helped": 1,
      "harmed": 1,
      "unchanged": 60,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CCW - NoComm",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 4.628942486085343,
      "ci95": [
        -5.371057513914657,
        20.408163265306122
      ],
      "randomization_p": 1.0,
      "helped": 1,
      "harmed": 3,
      "unchanged": 10,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CCW - NoComm",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 1.516615800360548,
      "ci95": [
        -0.6334549760560789,
        3.99843795630626
      ],
      "randomization_p": 0.19928007199280073,
      "helped": 12,
      "harmed": 6,
      "unchanged": 206,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CCW - NoComm",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "R_CCW - SelfUpdate",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": -1.2442396313364055,
      "ci95": [
        -4.838709677419355,
        1.105990783410138
      ],
      "randomization_p": 1.0,
      "helped": 1,
      "harmed": 1,
      "unchanged": 60,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CCW - SelfUpdate",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": -3.160173160173161,
      "ci95": [
        -6.938775510204083,
        0.0
      ],
      "randomization_p": 0.24897510248975102,
      "helped": 0,
      "harmed": 3,
      "unchanged": 11,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CCW - SelfUpdate",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 1.5464913600836852,
      "ci95": [
        -0.0666743970315399,
        3.546370110544218
      ],
      "randomization_p": 0.09659034096590341,
      "helped": 8,
      "harmed": 3,
      "unchanged": 213,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CCW - SelfUpdate",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "P_CW - NoComm",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 10.310854547097433,
      "ci95": [
        1.1111350698637235,
        20.16721497983357
      ],
      "randomization_p": 0.0397960203979602,
      "helped": 10,
      "harmed": 3,
      "unchanged": 49,
      "holm_p": 1.0
    },
    {
      "contrast": "P_CW - NoComm",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 17.857142857142858,
      "ci95": [
        0.0,
        39.285714285714285
      ],
      "randomization_p": 0.2501749825017498,
      "helped": 3,
      "harmed": 0,
      "unchanged": 11,
      "holm_p": 1.0
    },
    {
      "contrast": "P_CW - NoComm",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 1.2080627705627707,
      "ci95": [
        -0.3477408008658009,
        3.0222199675324672
      ],
      "randomization_p": 0.16718328167183283,
      "helped": 7,
      "harmed": 2,
      "unchanged": 215,
      "holm_p": 1.0
    },
    {
      "contrast": "P_CW - NoComm",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "P_CW - SelfUpdate",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 9.23558572989313,
      "ci95": [
        -1.0621282970666273,
        19.719461831653486
      ],
      "randomization_p": 0.09169083091690831,
      "helped": 10,
      "harmed": 4,
      "unchanged": 48,
      "holm_p": 1.0
    },
    {
      "contrast": "P_CW - SelfUpdate",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 17.857142857142858,
      "ci95": [
        0.0,
        39.285714285714285
      ],
      "randomization_p": 0.2501749825017498,
      "helped": 3,
      "harmed": 0,
      "unchanged": 11,
      "holm_p": 1.0
    },
    {
      "contrast": "P_CW - SelfUpdate",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 0.8509199134199134,
      "ci95": [
        -0.3801406926406928,
        2.441930465367964
      ],
      "randomization_p": 0.27077292270772924,
      "helped": 5,
      "harmed": 3,
      "unchanged": 216,
      "holm_p": 1.0
    },
    {
      "contrast": "P_CW - SelfUpdate",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "P_CCW - NoComm",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 9.45063949333399,
      "ci95": [
        0.21778506123857164,
        19.06731776288227
      ],
      "randomization_p": 0.0522947705229477,
      "helped": 9,
      "harmed": 3,
      "unchanged": 50,
      "holm_p": 1.0
    },
    {
      "contrast": "P_CCW - NoComm",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 11.554621848739496,
      "ci95": [
        0.0,
        28.571428571428573
      ],
      "randomization_p": 0.24597540245975402,
      "helped": 3,
      "harmed": 0,
      "unchanged": 11,
      "holm_p": 1.0
    },
    {
      "contrast": "P_CCW - NoComm",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 1.715561224489796,
      "ci95": [
        0.13818027210884357,
        3.5799319727891152
      ],
      "randomization_p": 0.05299470052994701,
      "helped": 8,
      "harmed": 1,
      "unchanged": 215,
      "holm_p": 1.0
    },
    {
      "contrast": "P_CCW - NoComm",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "P_CCW - SelfUpdate",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 8.37537067612969,
      "ci95": [
        0.11482745590904991,
        17.2151836541068
      ],
      "randomization_p": 0.06129387061293871,
      "helped": 8,
      "harmed": 3,
      "unchanged": 51,
      "holm_p": 1.0
    },
    {
      "contrast": "P_CCW - SelfUpdate",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 11.554621848739496,
      "ci95": [
        0.0,
        28.571428571428573
      ],
      "randomization_p": 0.24597540245975402,
      "helped": 3,
      "harmed": 0,
      "unchanged": 11,
      "holm_p": 1.0
    },
    {
      "contrast": "P_CCW - SelfUpdate",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 1.3584183673469388,
      "ci95": [
        -0.22337372448979584,
        3.2164115646258504
      ],
      "randomization_p": 0.1277872212778722,
      "helped": 7,
      "harmed": 3,
      "unchanged": 214,
      "holm_p": 1.0
    },
    {
      "contrast": "P_CCW - SelfUpdate",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "(CW-NoComm)_P - (CW-NoComm)_R",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 12.200255468756419,
      "ci95": [
        2.680804726582716,
        22.300595238095234
      ],
      "randomization_p": 0.0168983101689831,
      "helped": 12,
      "harmed": 3,
      "unchanged": 47,
      "holm_p": 0.45625437456254375
    },
    {
      "contrast": "(CW-NoComm)_P - (CW-NoComm)_R",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 10.317460317460316,
      "ci95": [
        -14.285714285714286,
        34.920634920634924
      ],
      "randomization_p": 0.6211378862113789,
      "helped": 3,
      "harmed": 1,
      "unchanged": 10,
      "holm_p": 1.0
    },
    {
      "contrast": "(CW-NoComm)_P - (CW-NoComm)_R",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": -0.16253709137112507,
      "ci95": [
        -2.8307845634854565,
        2.4508057010683055
      ],
      "randomization_p": 0.9055094490550945,
      "helped": 10,
      "harmed": 12,
      "unchanged": 202,
      "holm_p": 1.0
    },
    {
      "contrast": "(CW-NoComm)_P - (CW-NoComm)_R",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    }
  ],
  "heterogeneity": [
    {
      "contrast": "R_CW - NoComm",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -9.429083461341525,
      "ci95": [
        -25.07936507936508,
        0.0
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "R_CW - SelfUpdate",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -1.6399678150830228,
      "ci95": [
        -5.880696364567332,
        1.5138157413503022
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "R_CCW - NoComm",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -5.873182117421749,
      "ci95": [
        -22.244897959183675,
        4.7193129451193965
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "R_CCW - SelfUpdate",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": 1.9159335288367554,
      "ci95": [
        -2.857142857142857,
        6.346479940949989
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "P_CW - NoComm",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -7.5462883100454246,
      "ci95": [
        -30.043980568767093,
        11.366721908704834
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "P_CW - SelfUpdate",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -8.621557127249728,
      "ci95": [
        -31.630245940018238,
        10.945485965647244
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "P_CCW - NoComm",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -2.103982355405506,
      "ci95": [
        -21.173296224648215,
        13.53331464550629
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "P_CCW - SelfUpdate",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -3.1792511726098045,
      "ci95": [
        -21.816025530446783,
        12.005458488380688
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "(CW-NoComm)_P - (CW-NoComm)_R",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": 1.8827951512961043,
      "ci95": [
        -24.408793649424577,
        28.048249676214564
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    }
  ],
  "analysis_source_sha256": "e92948795db042457359bb8931413a6974ea4d00399d99087c18e45978c2ef50",
  "result_sha256": "1afe9ce6293a85acdcc743ee18b81c9ae79d57cd8258697cd715e77b4748a608",
  "limitations": [
    "Post-hoc diagnostic stratification; pooled primary estimand is retained",
    "Gold labels never select documents, partitions, prompts, actions or checkpoints",
    "Strata describe P initial evidence; R results use these same question groups for paired interaction",
    "Retrieval-Missing-Gold remains in pooled analysis and is reported separately",
    "Seed means use questions as independent units; no guaranteed positive or massive effect",
    "Bootstrap heterogeneity intervals are exploratory and unadjusted; effect p-values Holm-adjusted within this report"
  ]
}
```

### original_reports/controlled/E1/statistics.json

```json
{
  "experiment": "E1",
  "effects": [
    {
      "contrast": "(CW-NoComm)_P - (CW-NoComm)_R",
      "primary": true,
      "n": 300,
      "effect": 2.8815065834673677,
      "ci95": [
        -0.25996063576945927,
        6.010893847437964
      ],
      "randomization_p": 0.07129287071292871,
      "helped": 25,
      "harmed": 16,
      "unchanged": 259
    },
    {
      "contrast": "R_CW - R_NoComm",
      "primary": false,
      "n": 300,
      "effect": 0.9847568916196368,
      "ci95": [
        -0.7680588093823387,
        2.893296595379929
      ],
      "randomization_p": 0.29547045295470453,
      "helped": 14,
      "harmed": 6,
      "unchanged": 280,
      "holm_p_within_experiment": 0.857914208579142
    },
    {
      "contrast": "P_CW - P_NoComm",
      "primary": false,
      "n": 300,
      "effect": 3.8662634750870044,
      "ci95": [
        1.356056786350904,
        6.485954184704185
      ],
      "randomization_p": 0.0014998500149985001,
      "helped": 20,
      "harmed": 5,
      "unchanged": 275,
      "holm_p_within_experiment": 0.008999100089991
    },
    {
      "contrast": "R_CCW - R_NoComm",
      "primary": false,
      "n": 300,
      "effect": 1.0912809231436682,
      "ci95": [
        -0.8013360790444124,
        3.144086659074404
      ],
      "randomization_p": 0.285971402859714,
      "helped": 14,
      "harmed": 10,
      "unchanged": 276,
      "holm_p_within_experiment": 0.857914208579142
    },
    {
      "contrast": "P_CCW - P_NoComm",
      "primary": false,
      "n": 300,
      "effect": 3.7733002291825826,
      "ci95": [
        1.4206812876665817,
        6.354451447245564
      ],
      "randomization_p": 0.0023997600239976003,
      "helped": 20,
      "harmed": 4,
      "unchanged": 276,
      "holm_p_within_experiment": 0.011998800119988001
    },
    {
      "contrast": "R_CW - R_SelfUpdate",
      "primary": false,
      "n": 300,
      "effect": 0.6435719127208491,
      "ci95": [
        -0.870582611832612,
        2.2149882209988596
      ],
      "randomization_p": 0.420957904209579,
      "helped": 13,
      "harmed": 6,
      "unchanged": 281,
      "holm_p_within_experiment": 0.857914208579142
    },
    {
      "contrast": "P_CW - P_SelfUpdate",
      "primary": false,
      "n": 300,
      "effect": 3.3773745861981155,
      "ci95": [
        0.8490954715219422,
        6.052734487734487
      ],
      "randomization_p": 0.010998900109989001,
      "helped": 18,
      "harmed": 7,
      "unchanged": 275,
      "holm_p_within_experiment": 0.043995600439956005
    }
  ],
  "scale": "F1/EM percentage points",
  "provider_refusal_policy": "Explicit provider refusals are abstentions with preserved raw evidence and reported costs; malformed or truncated responses still stop the run",
  "citation_caveat": "Supporting-label overlap is a diagnostic; it does not prove entailment",
  "cost_caveat": "Condition cost includes shared initial calls. Physical measured usage is separate.",
  "stopping_caveat": "Replay savings are hypothetical, not observed bill reductions"
}
```

### original_reports/controlled/E2/gold_strata/statistics.json

```json
{
  "status": "COMPLETE",
  "experiment": "E2",
  "n": 300,
  "counts": {
    "Disjoint-Gold": 62,
    "Co-located-Gold": 14,
    "Retrieval-Missing-Gold": 224,
    "Undefined-Gold": 0
  },
  "effects": [
    {
      "contrast": "R_CW - R_CCW",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": -0.6451612903225806,
      "ci95": [
        -1.935483870967742,
        0.0
      ],
      "randomization_p": 1.0,
      "helped": 0,
      "harmed": 1,
      "unchanged": 61,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CW - R_CCW",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 2.910740053597197,
      "ci95": [
        0.12987012987013,
        6.507936507936508
      ],
      "randomization_p": 0.12248775122487751,
      "helped": 4,
      "harmed": 0,
      "unchanged": 10,
      "holm_p": 0.3674632536746325
    },
    {
      "contrast": "R_CW - R_CCW",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": -0.14601593842665275,
      "ci95": [
        -1.9430409000721502,
        1.5170361942125334
      ],
      "randomization_p": 0.8744125587441256,
      "helped": 6,
      "harmed": 5,
      "unchanged": 213,
      "holm_p": 1.0
    },
    {
      "contrast": "R_CW - R_CCW",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    }
  ],
  "heterogeneity": [
    {
      "contrast": "R_CW - R_CCW",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -3.5559013439197775,
      "ci95": [
        -7.273624992519003,
        -0.526695526695527
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    }
  ],
  "analysis_source_sha256": "e92948795db042457359bb8931413a6974ea4d00399d99087c18e45978c2ef50",
  "result_sha256": "dfae111876fc8fbcc3aaadd7ddff050909b3e76f0bb2e9ee51db7fa7490ed5c1",
  "limitations": [
    "Post-hoc diagnostic stratification; pooled primary estimand is retained",
    "Gold labels never select documents, partitions, prompts, actions or checkpoints",
    "Strata describe P initial evidence; R results use these same question groups for paired interaction",
    "Retrieval-Missing-Gold remains in pooled analysis and is reported separately",
    "Seed means use questions as independent units; no guaranteed positive or massive effect",
    "Bootstrap heterogeneity intervals are exploratory and unadjusted; effect p-values Holm-adjusted within this report"
  ]
}
```

### original_reports/controlled/E2/statistics.json

```json
{
  "experiment": "E2",
  "effects": [
    {
      "contrast": "R_CW - R_CCW",
      "primary": false,
      "n": 300,
      "effect": -0.1065240315240315,
      "ci95": [
        -1.4761482730232731,
        1.147305934805935
      ],
      "randomization_p": 0.8801119888011198,
      "helped": 10,
      "harmed": 6,
      "unchanged": 284,
      "holm_p_within_experiment": 0.8801119888011198
    }
  ],
  "scale": "F1/EM percentage points",
  "provider_refusal_policy": "Explicit provider refusals are abstentions with preserved raw evidence and reported costs; malformed or truncated responses still stop the run",
  "citation_caveat": "Supporting-label overlap is a diagnostic; it does not prove entailment",
  "cost_caveat": "Condition cost includes shared initial calls. Physical measured usage is separate.",
  "stopping_caveat": "Replay savings are hypothetical, not observed bill reductions"
}
```

### original_reports/controlled/E3/gold_strata/statistics.json

```json
{
  "status": "COMPLETE",
  "experiment": "E3",
  "n": 300,
  "counts": {
    "Disjoint-Gold": 62,
    "Co-located-Gold": 14,
    "Retrieval-Missing-Gold": 224,
    "Undefined-Gold": 0
  },
  "effects": [
    {
      "contrast": "R_prefix - R_crop",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "randomization_p": 1.0,
      "helped": 0,
      "harmed": 0,
      "unchanged": 62,
      "holm_p": 1.0
    },
    {
      "contrast": "R_prefix - R_crop",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "randomization_p": 1.0,
      "helped": 0,
      "harmed": 0,
      "unchanged": 14,
      "holm_p": 1.0
    },
    {
      "contrast": "R_prefix - R_crop",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 0.19422541743970315,
      "ci95": [
        -0.18262987012987011,
        0.7653061224489796
      ],
      "randomization_p": 1.0,
      "helped": 1,
      "harmed": 1,
      "unchanged": 222,
      "holm_p": 1.0
    },
    {
      "contrast": "R_prefix - R_crop",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    }
  ],
  "heterogeneity": [
    {
      "contrast": "R_prefix - R_crop",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    }
  ],
  "analysis_source_sha256": "e92948795db042457359bb8931413a6974ea4d00399d99087c18e45978c2ef50",
  "result_sha256": "793f10a77da7556cec6adec23e0bcbdd8e99cf5361ca977d202f61229cd2c8c6",
  "limitations": [
    "Post-hoc diagnostic stratification; pooled primary estimand is retained",
    "Gold labels never select documents, partitions, prompts, actions or checkpoints",
    "Strata describe P initial evidence; R results use these same question groups for paired interaction",
    "Retrieval-Missing-Gold remains in pooled analysis and is reported separately",
    "Seed means use questions as independent units; no guaranteed positive or massive effect",
    "Bootstrap heterogeneity intervals are exploratory and unadjusted; effect p-values Holm-adjusted within this report"
  ]
}
```

### original_reports/controlled/E3/statistics.json

```json
{
  "experiment": "E3",
  "effects": [
    {
      "contrast": "R_prefix - R_crop",
      "primary": false,
      "n": 300,
      "effect": 0.14502164502164502,
      "ci95": [
        -0.13636363636363635,
        0.5714285714285714
      ],
      "randomization_p": 1.0,
      "helped": 1,
      "harmed": 1,
      "unchanged": 298,
      "holm_p_within_experiment": 1
    }
  ],
  "scale": "F1/EM percentage points",
  "provider_refusal_policy": "Explicit provider refusals are abstentions with preserved raw evidence and reported costs; malformed or truncated responses still stop the run",
  "citation_caveat": "Supporting-label overlap is a diagnostic; it does not prove entailment",
  "cost_caveat": "Condition cost includes shared initial calls. Physical measured usage is separate.",
  "stopping_caveat": "Replay savings are hypothetical, not observed bill reductions",
  "identical_prefix_crop_questions": 298
}
```

### original_reports/controlled/E4/gold_strata/statistics.json

```json
{
  "status": "COMPLETE",
  "experiment": "E4",
  "n": 300,
  "counts": {
    "Disjoint-Gold": 62,
    "Co-located-Gold": 14,
    "Retrieval-Missing-Gold": 224,
    "Undefined-Gold": 0
  },
  "effects": [
    {
      "contrast": "P_stopped - P_final",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "randomization_p": 1.0,
      "helped": 0,
      "harmed": 0,
      "unchanged": 62,
      "holm_p": 1.0
    },
    {
      "contrast": "P_stopped - P_final",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "randomization_p": 1.0,
      "helped": 0,
      "harmed": 0,
      "unchanged": 14,
      "holm_p": 1.0
    },
    {
      "contrast": "P_stopped - P_final",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "randomization_p": 1.0,
      "helped": 0,
      "harmed": 0,
      "unchanged": 224,
      "holm_p": 1.0
    },
    {
      "contrast": "P_stopped - P_final",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    }
  ],
  "heterogeneity": [
    {
      "contrast": "P_stopped - P_final",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    }
  ],
  "analysis_source_sha256": "e92948795db042457359bb8931413a6974ea4d00399d99087c18e45978c2ef50",
  "result_sha256": "10ec23783228e25ab1002d0aeb241e7dd5f30f0b6e8a67e41d6f5b4cfe72084f",
  "limitations": [
    "Post-hoc diagnostic stratification; pooled primary estimand is retained",
    "Gold labels never select documents, partitions, prompts, actions or checkpoints",
    "Strata describe P initial evidence; R results use these same question groups for paired interaction",
    "Retrieval-Missing-Gold remains in pooled analysis and is reported separately",
    "Seed means use questions as independent units; no guaranteed positive or massive effect",
    "Bootstrap heterogeneity intervals are exploratory and unadjusted; effect p-values Holm-adjusted within this report"
  ]
}
```

### original_reports/controlled/E4/statistics.json

```json
{
  "experiment": "E4",
  "effects": [
    {
      "contrast": "P_stopped - P_final",
      "primary": false,
      "n": 300,
      "effect": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "randomization_p": 1.0,
      "helped": 0,
      "harmed": 0,
      "unchanged": 300,
      "holm_p_within_experiment": 1
    }
  ],
  "scale": "F1/EM percentage points",
  "provider_refusal_policy": "Explicit provider refusals are abstentions with preserved raw evidence and reported costs; malformed or truncated responses still stop the run",
  "citation_caveat": "Supporting-label overlap is a diagnostic; it does not prove entailment",
  "cost_caveat": "Condition cost includes shared initial calls. Physical measured usage is separate.",
  "stopping_caveat": "Replay savings are hypothetical, not observed bill reductions",
  "wrong_unanimity": 6,
  "continuation_corrected_wrong_stop": 0,
  "hypothetical_calls_avoided": 54,
  "stop_depth_by_question": {
    "5ae326e95542990afbd1e0b6": 3,
    "5a8f21c755429918e830d195": 3,
    "5adc6ded55429947ff17395d": 3,
    "5a793bc95542994bb9456fc3": 3,
    "5ae064985542993d6555eb47": 3,
    "5a790364554299029c4b5ee6": 3,
    "5a81e107554299676cceb12a": 3,
    "5ac05310554299012d1db5e7": 3,
    "5add352b5542990dbb2f7dc4": 3,
    "5a834cea5542990548d0b1c6": 3,
    "5a8c616f5542995e66a475de": 3,
    "5ab81cd955429916710eaff7": 3,
    "5a901f735542990a98493591": 3,
    "5a72bb045542994cef4bc3e4": 3,
    "5adf315c5542995ec70e8f8b": 3,
    "5ae7766455429952e35ea920": 3,
    "5a806aca5542996402f6a503": 3,
    "5a72239a55429971e9dc92cf": 3,
    "5a7506f455429916b0164240": 3,
    "5adcd7325542994d58a2f6f6": 3,
    "5a8361f35542996488c2e431": 3,
    "5a84a8ce5542997175ce1f1c": 3,
    "5ab2958c554299449642c912": 3,
    "5a8b120755429949d91db4a4": 3,
    "5a758aff5542992d0ec05fb2": 3,
    "5a82e25555429954d2e2ebbf": 3,
    "5ab95dd8554299743d22eb13": 3,
    "5a8e82ff5542995a26add4e2": 3,
    "5ae327e05542992f92d82220": 3,
    "5ab6087a554299110f2199be": 3,
    "5adfa6db55429942ec259ad6": 3,
    "5a721bbc55429971e9dc9279": 3,
    "5ae0120155429925eb1afbfb": 3,
    "5ab6a2be554299710c8d1f08": 3,
    "5a8e07b25542995085b3738d": 3,
    "5a7b76225542997c3ec9719c": 3,
    "5ae5d63b5542996de7b71a2d": 3,
    "5a734a205542992359bc32e2": 3,
    "5ae6d3345542996d980e7c85": 3,
    "5a7e14505542995ed0d16685": 3,
    "5ab3f8c25542992ade7c6f02": 3,
    "5a72b9155542992359bc3187": 3,
    "5ab659e355429954757d324e": 3,
    "5ae76dfd5542997b22f6a755": 3,
    "5a758ba75542992db9473677": 3,
    "5ae49bb755429970de88d9d7": 3,
    "5a80427c5542996402f6a49f": 3,
    "5ae0156455429942ec259c15": 3,
    "5abd0eae5542996e802b46d4": 3,
    "5ab3460055429976abd1bc80": 3,
    "5a7b55375542995eb53be91f": 3,
    "5a7c2819554299683c1c62db": 3,
    "5ab8f53355429916710eb0d1": 3,
    "5a80fd17554299260e20a1f1": 3,
    "5a8c49655542995e66a47598": 3,
    "5ab3cecd5542992ade7c6eae": 3,
    "5ae7535c5542997b22f6a6d8": 3,
    "5ac55ba05542993e66e8233d": 3,
    "5ab859a955429934fafe6d7b": 3,
    "5a865d11554299211dda2b0d": 3,
    "5a7b47765542992d025e67d8": 3,
    "5a72c9e85542991f9a20c595": 3,
    "5adcc6765542994d58a2f6c9": 3,
    "5abba4b4554299642a094af3": 3,
    "5adcf3a45542994ed6169c38": 3,
    "5ae27b535542996483e649b6": 3,
    "5abbc7a055429931dba1451b": 3,
    "5ab8ee035542991b5579f04b": 3,
    "5adc8d4f554299473435373d": 3,
    "5a8babf05542996e8ac8899e": 3,
    "5abd2cde5542993062266bb1": 3,
    "5a7f75e95542992e7d278d09": 3,
    "5a77857155429949eeb29ebf": 3,
    "5a7d1e685542995ed0d165f7": 3,
    "5ab8effd5542991b5579f062": 3,
    "5a79f6825542996c55b2dcb4": 3,
    "5adf5c175542995ec70e8fd0": 3,
    "5a78c56455429974737f7876": 3,
    "5ac19b2c55429963665198b6": 3,
    "5ab2b7405542991669774069": 3,
    "5ab67aba55429954757d32e7": 3,
    "5a7dc2185542990b8f503a7b": 3,
    "5ab48c325542996a3a969f93": 3,
    "5a8e8bed5542990e94052b47": 3,
    "5ae3776e5542992e3233c40c": 3,
    "5a7de63c5542995f4f4022f5": 3,
    "5ae5736e5542990ba0bbb2b3": 3,
    "5a7cf59b55429909bec768c7": 3,
    "5ac15f6c5542994d76dccdf3": 3,
    "5a80b9ae5542992bc0c4a7eb": 3,
    "5ae370295542992e3233c402": 3,
    "5a75d8dc5542992db94736f0": 3,
    "5a89e2ba5542992e4fca8441": 3,
    "5a8530cc5542992a431d1aeb": 3,
    "5adfc35755429906c02daa30": 3,
    "5ac2659b55429963665199a1": 3,
    "5a7d1bf65542995ed0d165f0": 3,
    "5ab761d955429928e1fe3856": 3,
    "5a729fa15542994cef4bc397": 3,
    "5ae0159855429925eb1afc09": 3,
    "5a7e4414554299495941995c": 3,
    "5ae71bee554299572ea546ea": 3,
    "5ac3d2ac554299741d48a330": 3,
    "5ab6be2f55429954757d335c": 2,
    "5a8d0be85542994ba4e3dbb2": 3,
    "5a77c1d95542995d831812a6": 3,
    "5a81d92f554299676cceb0f9": 3,
    "5ab3e45655429976abd1bcd4": 3,
    "5a8d8abf5542994ba4e3dca1": 3,
    "5a8ac4085542992d82986f74": 3,
    "5ae0c29c5542993d6555ec46": 3,
    "5ae54c085542992663a4f1c4": 3,
    "5a7fc5875542992e7d278d67": 3,
    "5a7a103b5542990783324e27": 3,
    "5a8bbb8c5542995d1e6f142e": 3,
    "5ae2c6bf554299492dc91c6c": 3,
    "5a7568175542992d0ec05f88": 3,
    "5a80956e5542996402f6a579": 3,
    "5a89d8d055429946c8d6e9df": 3,
    "5a7507455542993748c8979d": 3,
    "5ae6793e5542991bbc9760e2": 3,
    "5a85ecf25542991dd0999ec6": 3,
    "5a7db0915542990b8f503a30": 3,
    "5ae1801955429901ffe4aec4": 3,
    "5a7b40cd5542995eb53be8d4": 3,
    "5ab8111c5542991d3222382b": 0,
    "5ae765185542995703ce8c0f": 3,
    "5a77ddb855429967ab1052d8": 1,
    "5ab2d61155429916697740ff": 3,
    "5abdd8d95542991f6610606a": 3,
    "5ab2eff05542991669774140": 3,
    "5adfa8d0554299025d62a32c": 3,
    "5ac55ea55542993e66e82377": 3,
    "5adf53dc5542995534e8c791": 3,
    "5a8a39195542992d82986f04": 3,
    "5a89e85c5542992e4fca846f": 3,
    "5a746abf55429929fddd8417": 3,
    "5ac45cfd554299076e296db5": 3,
    "5abd2af955429933744ab74b": 3,
    "5a8747915542994775f607bb": 3,
    "5ae7db435542993210983f9a": 3,
    "5a8b59a65542995d1e6f137c": 3,
    "5a712beb5542994082a3e61c": 3,
    "5ae5aba0554299546bf82f17": 3,
    "5ae4627d5542991a06ce9a4d": 3,
    "5a7a4f405542994f819ef1c0": 3,
    "5a77510b5542994aec3b728b": 3,
    "5a88abf75542997e5c09a6a1": 3,
    "5a8d0efd5542994ba4e3dbc6": 3,
    "5ab978855542996be2020512": 3,
    "5a90580655429933b8a204ef": 3,
    "5ab5cd0d554299488d4d9a30": 3,
    "5ae095f055429924de1b7103": 3,
    "5ae0818e55429924de1b70e9": 3,
    "5ac4ec9955429924173fb4da": 3,
    "5ae1b2b9554299422ee99684": 3,
    "5ac02c6b554299294b218f38": 3,
    "5adf5fa65542993a75d264db": 3,
    "5ab6993f554299110f219a2b": 3,
    "5a7e23655542997cc2c47528": 3,
    "5ae4cda05542990ba0bbb148": 3,
    "5a72e4375542992359bc31bf": 3,
    "5ae1fa2b5542997f29b3c1df": 3,
    "5adfd42555429942ec259b52": 3,
    "5ab7f7465542993667794071": 3,
    "5a772c395542993735360207": 3,
    "5a8ec4875542995a26add508": 3,
    "5abd1daa55429933744ab72e": 3,
    "5abfab8d5542997ec76fd42d": 0,
    "5ab6aaff55429953192ad35b": 3,
    "5a74f8235542993748c89766": 3,
    "5ae8096f5542994a481bbe99": 3,
    "5a7def0d5542990b8f503af9": 3,
    "5aba77be55429955dce3ee44": 3,
    "5ab56e32554299637185c594": 3,
    "5ab514c05542991779162d72": 3,
    "5ae527085542993aec5ec165": 3,
    "5add97c15542997545bbbd80": 3,
    "5a7b971255429927d897bff3": 3,
    "5a7a34bf5542994f819ef151": 3,
    "5ab67c0055429954757d32f6": 3,
    "5a79ca585542996c55b2dc71": 3,
    "5ac15de55542991316484afb": 3,
    "5a7a1cd85542990783324e65": 3,
    "5a820f2f554299676cceb1ff": 3,
    "5ae0968955429924de1b7105": 3,
    "5abb93275542993f40c73b3c": 3,
    "5a8b86645542997f31a41d62": 3,
    "5a8b7cb95542995d1e6f13bb": 3,
    "5a8986fd55429938390d4046": 3,
    "5a73830d55429908901be2da": 3,
    "5a8f791c5542992414482ad9": 3,
    "5a71446b5542994082a3e72a": 3,
    "5a8cfee555429941ae14df5c": 3,
    "5ac2acff55429921a00ab02b": 3,
    "5a84c4b05542991dd0999deb": 3,
    "5aded04755429975fa854fa7": 3,
    "5a8c9641554299585d9e36f5": 3,
    "5ae08dad55429945ae9593b9": 3,
    "5a8fa89c55429918e830d29f": 3,
    "5ac15c1c5542994ab5c67cf7": 3,
    "5ae34a225542992e3233c370": 3,
    "5a7e02b75542997cc2c474f3": 3,
    "5a86e6f65542994775f6077c": 3,
    "5a8463d05542990548d0b2a5": 3,
    "5a7d3470554299452d57bb56": 3,
    "5ae6b6065542991bbc976168": 3,
    "5ab7c1a955429928e1fe38c8": 3,
    "5a8df1a65542995085b3735f": 2,
    "5a85fe2f5542996432c57165": 3,
    "5a77ba6c55429967ab10524c": 3,
    "5a7c4a6455429907fabeef53": 3,
    "5adfde09554299025d62a369": 3,
    "5a7d2afd5542995f4f402221": 3,
    "5a893fc855429951533612f0": 3,
    "5a803d695542996402f6a494": 3,
    "5a775f615542993735360250": 3,
    "5a7a1d305542990783324e66": 3,
    "5ae0bc185542993d6555ec29": 3,
    "5a7d7d895542995f4f402281": 3,
    "5ade15a45542997545bbbe46": 3,
    "5ae5c27b554299546bf82f4c": 3,
    "5a7f6f4b5542994857a76740": 3,
    "5a7cb9b95542990527d55515": 3,
    "5a85e24a5542994775f6067f": 3,
    "5a8c4c5f554299585d9e3651": 3,
    "5ae347fb5542992e3233c368": 3,
    "5ac4e476554299076e296e34": 2,
    "5ac0c84a554299012d1db633": 3,
    "5ae54e005542993aec5ec19a": 3,
    "5ae75e335542991e8301cc70": 3,
    "5ae4bc7955429913cc2044e9": 3,
    "5a87bc3f5542996432c57275": 3,
    "5ae658855542992ae0d162e3": 3,
    "5ac5188255429924173fb5a3": 3,
    "5a87964c5542996e4f30885f": 3,
    "5abcf84d55429959677d6b86": 3,
    "5a86ebac55429960ec39b6d6": 3,
    "5a8b3fe85542995d1e6f1323": 3,
    "5a7f18a35542994959419aa6": 3,
    "5a8cd381554299653c1aa12b": 3,
    "5ac2da27554299657fa2909f": 3,
    "5ac170af5542994d76dcce20": 3,
    "5abb73425542996cc5e49ff5": 3,
    "5ae5e8c85542996de7b71a69": 3,
    "5ab72a025542992aa3b8c7b8": 3,
    "5addf9e05542997dc79070f9": 3,
    "5ae10df355429920d52342ac": 3,
    "5ade26db5542997c77aded77": 3,
    "5adde99b5542992200553ba1": 3,
    "5a79ecd05542996c55b2dca1": 3,
    "5ab7f6d35542993667794070": 3,
    "5ac5391e5542994611c8b439": 3,
    "5abbd85a5542993f40c73be1": 3,
    "5ae6453e55429908198fa56c": 3,
    "5ae3d67f5542992f92d8238f": 3,
    "5a75e33855429976ec32bc74": 3,
    "5ae494575542995dadf2434d": 3,
    "5ac507725542994611c8b32a": 3,
    "5a8f7e785542992414482ae6": 3,
    "5ade89d855429975fa854ef7": 3,
    "5ae04ea555429924de1b709e": 3,
    "5a7c54eb554299683c1c62e8": 3,
    "5a7a88e455429941d65f268c": 3,
    "5a82a91155429940e5e1a911": 3,
    "5ac00f795542996f0d89cb16": 3,
    "5a78f26055429970f5fffdeb": 3,
    "5adef1b35542993a75d263af": 3,
    "5ae4c96355429913cc204506": 3,
    "5a8ef1e35542995a26add57a": 3,
    "5a7aa5e75542992d025e66d9": 3,
    "5a8b8f5b55429949d91db5e1": 3,
    "5ade914b55429939a52fe8f9": 3,
    "5a7455eb55429979e2882908": 2,
    "5a87560d5542994775f607d9": 3,
    "5ade79335542997c77adee38": 3,
    "5ae1dac95542997283cd229b": 3,
    "5ae1e4095542997283cd22bf": 0,
    "5ae46c9d5542996836b02c9d": 3,
    "5ae141475542997b2ef7d185": 3,
    "5ae0c7b055429945ae95944b": 3,
    "5ae31d5a5542990afbd1e0b1": 3,
    "5a7da9ae5542995f4f4022b1": 3,
    "5ae655845542991bbc9760c3": 1,
    "5a8cbda955429941ae14dee5": 3,
    "5a8de5d55542995085b3734d": 3,
    "5ae4c7c65542990ba0bbb12b": 3,
    "5ab1d983554299340b52540a": 3,
    "5ae0d7bf5542990adbacf67b": 3,
    "5abaf4535542996cc5e49f35": 3,
    "5a7eacd855429930675135c3": 3,
    "5abae3205542996cc5e49edc": 3,
    "5ae5df84554299546bf82fa8": 3,
    "5adee8ff55429975fa854fb8": 3,
    "5a88981c554299206df2b307": 2,
    "5a8208ee5542990a1d231f1e": 3,
    "5ac2455e55429951e9e68512": 3,
    "5addce005542997dc790703c": 3,
    "5a8740ee5542994846c1cd35": 3,
    "5ae09ce855429924de1b7113": 3
  }
}
```

### original_reports/controller/E5/gold_strata/statistics.json

```json
{
  "status": "COMPLETE",
  "experiment": "E5",
  "n": 300,
  "counts": {
    "Disjoint-Gold": 62,
    "Co-located-Gold": 14,
    "Retrieval-Missing-Gold": 224,
    "Undefined-Gold": 0
  },
  "effects": [
    {
      "contrast": "R_seed_mean_Learned - NoComm",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 1.25750593492529,
      "ci95": [
        -0.8490667039054135,
        3.618029372061629
      ],
      "randomization_p": 0.29817018298170184,
      "helped": 6,
      "harmed": 3,
      "unchanged": 53,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - NoComm",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": -4.732419283502874,
      "ci95": [
        -14.285714285714283,
        0.0884564352056616
      ],
      "randomization_p": 1.0,
      "helped": 1,
      "harmed": 1,
      "unchanged": 12,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - NoComm",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": -0.3277501268572695,
      "ci95": [
        -1.65564358756323,
        0.9061731299454511
      ],
      "randomization_p": 0.6312368763123688,
      "helped": 9,
      "harmed": 6,
      "unchanged": 209,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - NoComm",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "R_seed_mean_Learned - SelfUpdate",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": -0.4504998480330549,
      "ci95": [
        -3.9152755188096737,
        3.233770340980966
      ],
      "randomization_p": 0.8121187881211879,
      "helped": 5,
      "harmed": 6,
      "unchanged": 51,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - SelfUpdate",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 3.774074222990632,
      "ci95": [
        -0.10038465146824281,
        9.553295002211412
      ],
      "randomization_p": 0.37866213378662134,
      "helped": 3,
      "harmed": 1,
      "unchanged": 10,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - SelfUpdate",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 0.2420421840064699,
      "ci95": [
        -1.0922874868187364,
        1.5733337842712845
      ],
      "randomization_p": 0.7381261873812619,
      "helped": 9,
      "harmed": 8,
      "unchanged": 207,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - SelfUpdate",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "R_seed_mean_Learned - Static",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 0.6392263650328164,
      "ci95": [
        -1.7268625192012295,
        3.623935204580366
      ],
      "randomization_p": 0.6975302469753024,
      "helped": 4,
      "harmed": 5,
      "unchanged": 53,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - Static",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 2.410437859354268,
      "ci95": [
        0.0,
        7.201828099660917
      ],
      "randomization_p": 0.5014498550144986,
      "helped": 2,
      "harmed": 0,
      "unchanged": 12,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - Static",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 0.7436784445713018,
      "ci95": [
        -0.6613178488178485,
        2.154057685865721
      ],
      "randomization_p": 0.30726927307269275,
      "helped": 12,
      "harmed": 5,
      "unchanged": 207,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - Static",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "R_seed_mean_Learned - Broadcast",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 0.7690964949029464,
      "ci95": [
        -1.671824814969976,
        3.963764953684308
      ],
      "randomization_p": 0.641035896410359,
      "helped": 4,
      "harmed": 5,
      "unchanged": 53,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - Broadcast",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 4.235834684751094,
      "ci95": [
        0.0589709568037744,
        9.950120399036809
      ],
      "randomization_p": 0.12438756124387561,
      "helped": 4,
      "harmed": 0,
      "unchanged": 10,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - Broadcast",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 0.13490477776192084,
      "ci95": [
        -1.1677269902716325,
        1.3985924789496218
      ],
      "randomization_p": 0.8417158284171583,
      "helped": 9,
      "harmed": 8,
      "unchanged": 207,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - Broadcast",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "R_seed_mean_Learned - MatchedRandom",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 0.42498719918074734,
      "ci95": [
        -0.9575012800819254,
        2.0225294418842803
      ],
      "randomization_p": 0.609039096090391,
      "helped": 4,
      "harmed": 3,
      "unchanged": 55,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - MatchedRandom",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 1.0207336523125994,
      "ci95": [
        0.0,
        2.9938482570061504
      ],
      "randomization_p": 0.506049395060494,
      "helped": 2,
      "harmed": 0,
      "unchanged": 12,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - MatchedRandom",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": -0.2713361302258817,
      "ci95": [
        -1.2574677306820163,
        0.6422127357201113
      ],
      "randomization_p": 0.6200379962003799,
      "helped": 8,
      "harmed": 6,
      "unchanged": 210,
      "holm_p": 1.0
    },
    {
      "contrast": "R_seed_mean_Learned - MatchedRandom",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "P_seed_mean_Learned - NoComm",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 11.4315765406847,
      "ci95": [
        5.075911596310078,
        18.294858143648465
      ],
      "randomization_p": 0.0008999100089991,
      "helped": 17,
      "harmed": 4,
      "unchanged": 41,
      "holm_p": 0.0242975702429757
    },
    {
      "contrast": "P_seed_mean_Learned - NoComm",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 19.047619047619047,
      "ci95": [
        4.761904761904762,
        38.095238095238095
      ],
      "randomization_p": 0.12758724127587243,
      "helped": 4,
      "harmed": 0,
      "unchanged": 10,
      "holm_p": 1.0
    },
    {
      "contrast": "P_seed_mean_Learned - NoComm",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 3.0123278539177156,
      "ci95": [
        1.2650137920515225,
        5.02739448051948
      ],
      "randomization_p": 0.0004999500049995,
      "helped": 19,
      "harmed": 2,
      "unchanged": 203,
      "holm_p": 0.013998600139986002
    },
    {
      "contrast": "P_seed_mean_Learned - NoComm",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "P_seed_mean_Learned - SelfUpdate",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 8.052160258042612,
      "ci95": [
        -0.293653018392107,
        16.298262340126666
      ],
      "randomization_p": 0.05829417058294171,
      "helped": 18,
      "harmed": 6,
      "unchanged": 38,
      "holm_p": 1.0
    },
    {
      "contrast": "P_seed_mean_Learned - SelfUpdate",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 19.047619047619047,
      "ci95": [
        4.761904761904762,
        38.095238095238095
      ],
      "randomization_p": 0.12758724127587243,
      "helped": 4,
      "harmed": 0,
      "unchanged": 10,
      "holm_p": 1.0
    },
    {
      "contrast": "P_seed_mean_Learned - SelfUpdate",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 2.3991228945146,
      "ci95": [
        0.37775098125760587,
        4.53638762750946
      ],
      "randomization_p": 0.020997900209979003,
      "helped": 18,
      "harmed": 5,
      "unchanged": 201,
      "holm_p": 0.5249475052494751
    },
    {
      "contrast": "P_seed_mean_Learned - SelfUpdate",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "P_seed_mean_Learned - Static",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 7.41174280073711,
      "ci95": [
        -0.7216273606216673,
        15.606971038659838
      ],
      "randomization_p": 0.07999200079992001,
      "helped": 17,
      "harmed": 5,
      "unchanged": 40,
      "holm_p": 1.0
    },
    {
      "contrast": "P_seed_mean_Learned - Static",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 7.683982683982685,
      "ci95": [
        -9.523809523809522,
        27.922077922077925
      ],
      "randomization_p": 0.49495050494950504,
      "helped": 3,
      "harmed": 3,
      "unchanged": 8,
      "holm_p": 1.0
    },
    {
      "contrast": "P_seed_mean_Learned - Static",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 3.0123278539177156,
      "ci95": [
        1.5950883194186767,
        4.709484835600908
      ],
      "randomization_p": 9.999000099990002e-05,
      "helped": 19,
      "harmed": 0,
      "unchanged": 205,
      "holm_p": 0.0029997000299970007
    },
    {
      "contrast": "P_seed_mean_Learned - Static",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "P_seed_mean_Learned - Broadcast",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": -9.953150370607675,
      "ci95": [
        -16.45608292192732,
        -3.742494065074709
      ],
      "randomization_p": 0.0030996900309969004,
      "helped": 3,
      "harmed": 15,
      "unchanged": 44,
      "holm_p": 0.08059194080591942
    },
    {
      "contrast": "P_seed_mean_Learned - Broadcast",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": -9.523809523809522,
      "ci95": [
        -21.428571428571423,
        0.0
      ],
      "randomization_p": 0.25107489251074894,
      "helped": 0,
      "harmed": 3,
      "unchanged": 11,
      "holm_p": 1.0
    },
    {
      "contrast": "P_seed_mean_Learned - Broadcast",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": -2.5850298574952952,
      "ci95": [
        -4.180875212377227,
        -1.235098890484835
      ],
      "randomization_p": 9.999000099990002e-05,
      "helped": 1,
      "harmed": 17,
      "unchanged": 206,
      "holm_p": 0.0029997000299970007
    },
    {
      "contrast": "P_seed_mean_Learned - Broadcast",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    },
    {
      "contrast": "P_seed_mean_Learned - MatchedRandom",
      "stratum": "Disjoint-Gold",
      "status": "ESTIMATED",
      "n": 62,
      "effect": 1.891321044546851,
      "ci95": [
        -1.6225038402457759,
        5.7606086789554505
      ],
      "randomization_p": 0.3354664533546645,
      "helped": 7,
      "harmed": 5,
      "unchanged": 50,
      "holm_p": 1.0
    },
    {
      "contrast": "P_seed_mean_Learned - MatchedRandom",
      "stratum": "Co-located-Gold",
      "status": "ESTIMATED",
      "n": 14,
      "effect": 2.380952380952381,
      "ci95": [
        0.0,
        7.142857142857143
      ],
      "randomization_p": 1.0,
      "helped": 1,
      "harmed": 0,
      "unchanged": 13,
      "holm_p": 1.0
    },
    {
      "contrast": "P_seed_mean_Learned - MatchedRandom",
      "stratum": "Retrieval-Missing-Gold",
      "status": "ESTIMATED",
      "n": 224,
      "effect": 0.6881113118512311,
      "ci95": [
        -0.02931096681096698,
        1.6098925785795737
      ],
      "randomization_p": 0.10328967103289671,
      "helped": 7,
      "harmed": 3,
      "unchanged": 214,
      "holm_p": 1.0
    },
    {
      "contrast": "P_seed_mean_Learned - MatchedRandom",
      "stratum": "Undefined-Gold",
      "n": 0,
      "status": "UNESTIMABLE_EMPTY_STRATUM"
    }
  ],
  "heterogeneity": [
    {
      "contrast": "R_seed_mean_Learned - NoComm",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": 5.989925218428164,
      "ci95": [
        -0.3588156464417328,
        16.498933024874294
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "R_seed_mean_Learned - SelfUpdate",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -4.224574071023686,
      "ci95": [
        -10.602315762956929,
        1.3445976309441252
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "R_seed_mean_Learned - Static",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -1.7712114943214519,
      "ci95": [
        -7.621125482399577,
        2.827618873602845
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "R_seed_mean_Learned - Broadcast",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -3.4667381898481473,
      "ci95": [
        -9.433712076176873,
        1.867694805194801
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "R_seed_mean_Learned - MatchedRandom",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -0.5957464531318519,
      "ci95": [
        -3.038142766495906,
        1.5581677801541969
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "P_seed_mean_Learned - NoComm",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -7.616042506934347,
      "ci95": [
        -27.209032073239865,
        9.197774539923492
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "P_seed_mean_Learned - SelfUpdate",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -10.995458789576436,
      "ci95": [
        -31.250572443053468,
        6.772180157497996
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "P_seed_mean_Learned - Static",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -0.27223988324557524,
      "ci95": [
        -22.130985580938148,
        18.74777183600713
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "P_seed_mean_Learned - Broadcast",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -0.4293408467981532,
      "ci95": [
        -11.233215182456169,
        12.924323199465512
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    },
    {
      "contrast": "P_seed_mean_Learned - MatchedRandom",
      "heterogeneity": "Disjoint-Gold minus Co-located-Gold",
      "status": "ESTIMATED",
      "n_disjoint": 62,
      "n_colocated": 14,
      "effect": -0.48963133640553025,
      "ci95": [
        -6.816916282642089,
        4.800547235023037
      ],
      "interpretation": "Observational effect heterogeneity; stratum membership was not randomized"
    }
  ],
  "analysis_source_sha256": "e92948795db042457359bb8931413a6974ea4d00399d99087c18e45978c2ef50",
  "result_sha256": "cf34e15527704024760f6ceaf11a34f62cfbdf000c08422bd6568f165ef7beca",
  "limitations": [
    "Post-hoc diagnostic stratification; pooled primary estimand is retained",
    "Gold labels never select documents, partitions, prompts, actions or checkpoints",
    "Strata describe P initial evidence; R results use these same question groups for paired interaction",
    "Retrieval-Missing-Gold remains in pooled analysis and is reported separately",
    "Seed means use questions as independent units; no guaranteed positive or massive effect",
    "Bootstrap heterogeneity intervals are exploratory and unadjusted; effect p-values Holm-adjusted within this report"
  ]
}
```

### original_reports/controller/E5_statistics.json

```json
{
  "status": "COMPLETE",
  "heldout_questions": 300,
  "optimizer_seeds": [
    7,
    19,
    31
  ],
  "effects": [
    {
      "regime": "R",
      "metric": "f1",
      "contrast": "seed_mean_learned - Static",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 0.7998737874898867,
      "ci95": [
        -0.3725974025974023,
        2.018996188996189
      ],
      "randomization_p": 0.1947805219478052,
      "helped": 18,
      "harmed": 10,
      "unchanged": 272,
      "holm_p": 1.0
    },
    {
      "regime": "R",
      "metric": "f1",
      "contrast": "seed_mean_learned - NoComm",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": -0.20568176806566887,
      "ci95": [
        -1.4058396863427818,
        0.9160068531972552
      ],
      "randomization_p": 0.7283271672832716,
      "helped": 16,
      "harmed": 10,
      "unchanged": 274,
      "holm_p": 1.0
    },
    {
      "regime": "R",
      "metric": "f1",
      "contrast": "seed_mean_learned - SelfUpdate",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 0.2637449925375624,
      "ci95": [
        -0.9745510180107698,
        1.5131373942248554
      ],
      "randomization_p": 0.6833316668333167,
      "helped": 17,
      "harmed": 15,
      "unchanged": 268,
      "holm_p": 1.0
    },
    {
      "regime": "R",
      "metric": "f1",
      "contrast": "seed_mean_learned - Broadcast",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 0.4573477949638942,
      "ci95": [
        -0.6878669581765553,
        1.6098516838338817
      ],
      "randomization_p": 0.43915608439156084,
      "helped": 17,
      "harmed": 13,
      "unchanged": 270,
      "holm_p": 1.0
    },
    {
      "regime": "R",
      "metric": "f1",
      "contrast": "seed_mean_learned - MatchedRandom",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": -0.06713271896338265,
      "ci95": [
        -0.8712590495571044,
        0.7084754329491171
      ],
      "randomization_p": 0.8635136486351365,
      "helped": 14,
      "harmed": 9,
      "unchanged": 277,
      "holm_p": 1.0
    },
    {
      "regime": "R",
      "metric": "utility_gain",
      "contrast": "seed_mean_learned - Static",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 0.019805825407866094,
      "ci95": [
        0.00812132339100251,
        0.0320216639516201
      ],
      "randomization_p": 0.0005999400059994001,
      "helped": 285,
      "harmed": 15,
      "unchanged": 0,
      "holm_p": 0.008399160083991601
    },
    {
      "regime": "R",
      "metric": "utility_gain",
      "contrast": "seed_mean_learned - NoComm",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": -0.020022517734700603,
      "ci95": [
        -0.031963690317788084,
        -0.008737465820745737
      ],
      "randomization_p": 9.999000099990002e-05,
      "helped": 14,
      "harmed": 286,
      "unchanged": 0,
      "holm_p": 0.001999800019998
    },
    {
      "regime": "R",
      "metric": "utility_gain",
      "contrast": "seed_mean_learned - SelfUpdate",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 0.004671749871331712,
      "ci95": [
        -0.007651433898099584,
        0.017134670227165114
      ],
      "randomization_p": 0.47465253474652536,
      "helped": 245,
      "harmed": 53,
      "unchanged": 2,
      "holm_p": 1.0
    },
    {
      "regime": "R",
      "metric": "utility_gain",
      "contrast": "seed_mean_learned - Broadcast",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 0.056479812924531886,
      "ci95": [
        0.0450247708078268,
        0.0679980477526322
      ],
      "randomization_p": 9.999000099990002e-05,
      "helped": 289,
      "harmed": 11,
      "unchanged": 0,
      "holm_p": 0.001999800019998
    },
    {
      "regime": "R",
      "metric": "utility_gain",
      "contrast": "seed_mean_learned - MatchedRandom",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 0.0016043845687694683,
      "ci95": [
        -0.006423761881092969,
        0.009409661256685072
      ],
      "randomization_p": 0.6939306069393061,
      "helped": 212,
      "harmed": 28,
      "unchanged": 60,
      "holm_p": 1.0
    },
    {
      "regime": "P",
      "metric": "f1",
      "contrast": "seed_mean_learned - Static",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 4.139550834996756,
      "ci95": [
        1.86126018246986,
        6.416441100715294
      ],
      "randomization_p": 0.00039996000399960006,
      "helped": 39,
      "harmed": 8,
      "unchanged": 253,
      "holm_p": 0.006399360063993601
    },
    {
      "regime": "P",
      "metric": "f1",
      "contrast": "seed_mean_learned - NoComm",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 5.500619504888955,
      "ci95": [
        3.456924045277936,
        7.75711375752647
      ],
      "randomization_p": 9.999000099990002e-05,
      "helped": 40,
      "harmed": 6,
      "unchanged": 254,
      "holm_p": 0.001999800019998
    },
    {
      "regime": "P",
      "metric": "f1",
      "contrast": "seed_mean_learned - SelfUpdate",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 4.344347103455263,
      "ci95": [
        1.9230151850574058,
        6.868642444595955
      ],
      "randomization_p": 0.00039996000399960006,
      "helped": 40,
      "harmed": 11,
      "unchanged": 249,
      "holm_p": 0.006399360063993601
    },
    {
      "regime": "P",
      "metric": "f1",
      "contrast": "seed_mean_learned - Broadcast",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": -4.431584481299851,
      "ci95": [
        -6.312186913866229,
        -2.707283755144286
      ],
      "randomization_p": 9.999000099990002e-05,
      "helped": 4,
      "harmed": 35,
      "unchanged": 261,
      "holm_p": 0.001999800019998
    },
    {
      "regime": "P",
      "metric": "f1",
      "contrast": "seed_mean_learned - MatchedRandom",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 1.0157739064997129,
      "ci95": [
        0.07171884455150579,
        2.0603352214541726
      ],
      "randomization_p": 0.0456954304569543,
      "helped": 15,
      "harmed": 8,
      "unchanged": 277,
      "holm_p": 0.5026497350264973
    },
    {
      "regime": "P",
      "metric": "utility_gain",
      "contrast": "seed_mean_learned - Static",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 0.032843363821293665,
      "ci95": [
        0.010490790898433012,
        0.055136940415954884
      ],
      "randomization_p": 0.0035996400359964,
      "helped": 73,
      "harmed": 227,
      "unchanged": 0,
      "holm_p": 0.046795320467953205
    },
    {
      "regime": "P",
      "metric": "utility_gain",
      "contrast": "seed_mean_learned - NoComm",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 0.016242301586375893,
      "ci95": [
        -0.00416293249170223,
        0.03884027789336464
      ],
      "randomization_p": 0.14238576142385762,
      "helped": 39,
      "harmed": 261,
      "unchanged": 0,
      "holm_p": 1.0
    },
    {
      "regime": "P",
      "metric": "utility_gain",
      "contrast": "seed_mean_learned - SelfUpdate",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 0.024679577572038987,
      "ci95": [
        0.0004360020625306472,
        0.049870481328375627
      ],
      "randomization_p": 0.0496950304969503,
      "helped": 41,
      "harmed": 259,
      "unchanged": 0,
      "holm_p": 0.5026497350264973
    },
    {
      "regime": "P",
      "metric": "utility_gain",
      "contrast": "seed_mean_learned - Broadcast",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": -0.013129495676131142,
      "ci95": [
        -0.03193784721437454,
        0.004032709495065079
      ],
      "randomization_p": 0.1663833616638336,
      "helped": 266,
      "harmed": 34,
      "unchanged": 0,
      "holm_p": 1.0
    },
    {
      "regime": "P",
      "metric": "utility_gain",
      "contrast": "seed_mean_learned - MatchedRandom",
      "inference": "exploratory; paired by question, seeds averaged within question",
      "n": 300,
      "effect": 0.013587427444340919,
      "ci95": [
        0.004281184322821576,
        0.023926494155176607
      ],
      "randomization_p": 0.006399360063993601,
      "helped": 246,
      "harmed": 49,
      "unchanged": 5,
      "holm_p": 0.07679232076792321
    }
  ],
  "action_frequencies": {
    "P_seed19": {
      "14": 99,
      "32": 196,
      "21": 5
    },
    "P_seed31": {
      "14": 300
    },
    "P_seed7": {
      "54": 294,
      "2": 5,
      "31": 1
    },
    "R_seed19": {
      "14": 89,
      "32": 172,
      "7": 39
    },
    "R_seed31": {
      "0": 241,
      "14": 59
    },
    "R_seed7": {
      "45": 172,
      "40": 8,
      "31": 119,
      "54": 1
    }
  },
  "unique_actions": {
    "P_seed19": 3,
    "P_seed31": 1,
    "P_seed7": 3,
    "R_seed19": 3,
    "R_seed31": 2,
    "R_seed7": 4
  },
  "no_comm_rates": {
    "P_seed19": 0.0,
    "P_seed31": 0.0,
    "P_seed7": 0.0,
    "R_seed19": 0.0,
    "R_seed31": 0.8033333333333333,
    "R_seed7": 0.0
  },
  "physical_usage": {
    "attempted_calls": 10037,
    "successful_calls": 10019,
    "provider_refusal_calls": 68,
    "error_calls": 18,
    "usage_complete_for_successes": false,
    "input_tokens": null,
    "output_tokens": null,
    "caveat": "Errors/in-flight calls may be billable and have unknown usage. No price or monetary saving inferred."
  },
  "provider_refusal_policy": "abstain_with_reported_usage",
  "limits": [
    "This is a whole-policy comparison; call/recipient/payload differences are measured, not controlled away",
    "Seed averaging does not create more independent questions or independent model backbones",
    "All contrasts here are exploratory relative to the separate E1 primary interaction",
    "A single selected support or NoComm dominance must be reported without post-hoc retraining on held-out outcomes"
  ]
}
```

## Full-precision files and original paper numbers

- [E2_E1_shared_outcome_audit.csv](tables/E2_E1_shared_outcome_audit.csv)
- [E5_complete_action_adaptivity.csv](tables/E5_complete_action_adaptivity.csv)
- [E5_complete_missing_data_bounds.csv](tables/E5_complete_missing_data_bounds.csv)
- [E5_complete_quality_resource_tradeoffs.csv](tables/E5_complete_quality_resource_tradeoffs.csv)
- [E5_complete_resource_matching.csv](tables/E5_complete_resource_matching.csv)
- [E5_complete_seed_averages.csv](tables/E5_complete_seed_averages.csv)
- [E5_subgroup_diagnostics.csv](tables/E5_subgroup_diagnostics.csv)
- [active_pipeline_physical_usage.csv](tables/active_pipeline_physical_usage.csv)
- [all_arms_by_stratum.csv](tables/all_arms_by_stratum.csv)
- [all_arms_by_type.csv](tables/all_arms_by_type.csv)
- [all_condition_question_metrics.csv](tables/all_condition_question_metrics.csv)
- [all_original_report_fields.csv](tables/all_original_report_fields.csv)
- [all_raw_numeric_fields.csv](tables/all_raw_numeric_fields.csv)
- [all_scalar_distribution_metrics.csv](tables/all_scalar_distribution_metrics.csv)
- [all_training_validation_metrics.csv](tables/all_training_validation_metrics.csv)
- [checkpoint_training_artifact_inventory.csv](tables/checkpoint_training_artifact_inventory.csv)
- [condition_summary.csv](tables/condition_summary.csv)
- [controller_learning_curves.csv](tables/controller_learning_curves.csv)
- [cross_suite_same_question_baselines.csv](tables/cross_suite_same_question_baselines.csv)
- [execution_prefix_trends.csv](tables/execution_prefix_trends.csv)
- [frozen_actor_probability_diagnostics.csv](tables/frozen_actor_probability_diagnostics.csv)
- [frozen_actor_probability_summary.csv](tables/frozen_actor_probability_summary.csv)
- [gold_retrieval_question_audit.csv](tables/gold_retrieval_question_audit.csv)
- [learned_broadcast_identity.csv](tables/learned_broadcast_identity.csv)
- [manuscript_numeric_claim_inventory.csv](tables/manuscript_numeric_claim_inventory.csv)
- [network_timing_summary.csv](tables/network_timing_summary.csv)
- [network_timings.csv](tables/network_timings.csv)
- [paired_question_transitions.csv](tables/paired_question_transitions.csv)
- [payload_token_verification_errors.csv](tables/payload_token_verification_errors.csv)
- [physical_attempts_deduplicated.csv](tables/physical_attempts_deduplicated.csv)
- [pilot_static_all_numeric_fields.csv](tables/pilot_static_all_numeric_fields.csv)
- [pilot_static_question_metrics.csv](tables/pilot_static_question_metrics.csv)
- [pilot_static_summary.csv](tables/pilot_static_summary.csv)
- [refusal_descriptive_groups.csv](tables/refusal_descriptive_groups.csv)
- [selected_seed_performance.csv](tables/selected_seed_performance.csv)
- [worker_session_timings.csv](tables/worker_session_timings.csv)

[Supplied paper full extraction](sources/current_paper_extracted.txt) preserves all paper tables/figures text and numbers, including P1–P5/W1–W4. The numeric claim inventory labels them manuscript-reported. This report does not blend them into verified E-suite evidence.
