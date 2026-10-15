# Communication effect reassessment

**The large partitioned gain weakens sharply under matched abstention handling; it does not become a demonstrated communication harm.** Original Broadcast−NoComm EM is+8 points, aware is+1 point, and lexical-aware is 0. Every comparison below holds the readout fixed. The same 300 questions are paired, not pooled across policies, readouts or evidence regimes.

Intervals are 95% paired question bootstraps:10,000 draws, seed 20261016. W/L/T counts are binary EM paired wins/losses/ties. P is the exact two-sided conditional McNemar test. All 108 declared communication comparisons (including learned versus Static/Broadcast/same-seed MatchedRandom) form one Holm diagnostic family; all 40 readout-versus-original comparisons form a separate family. The present tables show the 54 comparisons against NoComm; all 108 and every F1 result are retained in [results/communication_effects.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/results/communication_effects.json). No interval or corrected p-value turns post hoc selection of a readout into a confirmatory superiority claim.

![Matched original and repaired readouts for NoComm and Broadcast](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/figures/matched_broadcast_no_comm.png)

## Matched communication effects: original


| Arm−same-regime NoComm | EM Δpp [CI] | W/L/T | Exact p | Holm p | F1 Δpp [CI] |
|---|---|---|---|---|---|
| P_SelfUpdate | +0.667 [+0.000, +1.667] | 2/0/298 | 0.5 | 1 | +1.156 [-0.114, +2.606] |
| P_Static | +1.000 [-1.000, +3.000] | 6/3/291 | 0.507812 | 1 | +1.361 [-0.639, +3.524] |
| P_Broadcast | +8.000 [+5.000, +11.333] | 25/1/274 | 8.04663e-07 | 8.69036e-05 | +9.932 [+6.681, +13.423] |
| P_Learned_seed7 | +8.000 [+5.000, +11.333] | 25/1/274 | 8.04663e-07 | 8.69036e-05 | +9.932 [+6.681, +13.423] |
| P_Learned_seed19 | +3.333 [+1.000, +5.667] | 12/2/286 | 0.0129395 | 1 | +4.105 [+1.804, +6.582] |
| P_Learned_seed31 | +2.333 [+0.333, +4.667] | 9/2/289 | 0.0654297 | 1 | +2.464 [+0.417, +4.715] |
| P_MatchedRandom_seed7 | +8.000 [+5.000, +11.333] | 25/1/274 | 8.04663e-07 | 8.69036e-05 | +9.932 [+6.681, +13.423] |
| P_MatchedRandom_seed19 | +1.667 [+0.000, +3.333] | 6/1/293 | 0.125 | 1 | +2.007 [+0.325, +3.939] |
| P_MatchedRandom_seed31 | +1.000 [-1.000, +3.000] | 6/3/291 | 0.507812 | 1 | +1.515 [-0.318, +3.515] |
| R_SelfUpdate | -0.333 [-2.000, +1.333] | 3/4/293 | 1 | 1 | -0.469 [-2.173, +1.106] |
| R_Static | -0.333 [-2.000, +1.000] | 2/3/295 | 1 | 1 | -1.006 [-2.711, +0.583] |
| R_Broadcast | +0.333 [-1.667, +2.333] | 5/4/291 | 1 | 1 | -0.663 [-2.486, +1.108] |
| R_Learned_seed7 | +1.333 [-0.667, +3.333] | 7/3/290 | 0.34375 | 1 | +0.276 [-1.593, +2.152] |
| R_Learned_seed19 | +0.667 [-1.000, +2.667] | 5/3/292 | 0.726562 | 1 | -0.348 [-1.992, +1.187] |
| R_Learned_seed31 | -0.333 [-1.000, +0.000] | 0/1/299 | 1 | 1 | -0.544 [-1.633, +0.334] |
| R_MatchedRandom_seed7 | +0.000 [-1.333, +1.333] | 2/2/296 | 1 | 1 | -0.206 [-1.559, +1.106] |
| R_MatchedRandom_seed19 | +0.667 [-1.333, +2.667] | 5/3/292 | 0.726562 | 1 | +0.049 [-1.835, +1.937] |
| R_MatchedRandom_seed31 | -0.667 [-2.000, +0.667] | 1/3/296 | 0.625 | 1 | -0.258 [-1.511, +0.880] |



## Matched communication effects: abstention_aware


| Arm−same-regime NoComm | EM Δpp [CI] | W/L/T | Exact p | Holm p | F1 Δpp [CI] |
|---|---|---|---|---|---|
| P_SelfUpdate | -0.333 [-2.333, +2.000] | 5/6/289 | 1 | 1 | -0.080 [-2.132, +1.954] |
| P_Static | -0.667 [-2.667, +1.333] | 4/6/290 | 0.753906 | 1 | -0.857 [-2.796, +1.015] |
| P_Broadcast | +1.000 [-1.333, +3.333] | 8/5/287 | 0.581055 | 1 | +1.231 [-0.802, +3.389] |
| P_Learned_seed7 | +1.000 [-1.333, +3.333] | 8/5/287 | 0.581055 | 1 | +1.231 [-0.802, +3.389] |
| P_Learned_seed19 | +1.000 [-0.667, +2.667] | 5/2/293 | 0.453125 | 1 | +1.065 [-0.395, +2.740] |
| P_Learned_seed31 | +0.333 [-1.333, +2.000] | 4/3/293 | 1 | 1 | +0.174 [-1.294, +1.668] |
| P_MatchedRandom_seed7 | +1.000 [-1.333, +3.333] | 8/5/287 | 0.581055 | 1 | +1.231 [-0.802, +3.389] |
| P_MatchedRandom_seed19 | +0.333 [-1.000, +1.667] | 3/2/295 | 1 | 1 | +0.558 [-0.917, +2.031] |
| P_MatchedRandom_seed31 | +1.333 [-0.667, +3.333] | 7/3/290 | 0.34375 | 1 | +1.557 [-0.296, +3.546] |
| R_SelfUpdate | -0.667 [-2.667, +1.000] | 3/5/292 | 0.726562 | 1 | -0.308 [-2.042, +1.354] |
| R_Static | +0.667 [+0.000, +1.667] | 2/0/298 | 0.5 | 1 | +0.779 [+0.106, +1.695] |
| R_Broadcast | +0.000 [-2.000, +2.000] | 5/5/290 | 1 | 1 | -0.641 [-2.437, +1.046] |
| R_Learned_seed7 | +1.667 [-0.333, +3.667] | 7/2/291 | 0.179688 | 1 | +0.960 [-0.648, +2.613] |
| R_Learned_seed19 | +1.333 [+0.000, +3.000] | 5/1/294 | 0.21875 | 1 | +0.547 [-0.566, +1.728] |
| R_Learned_seed31 | -0.333 [-1.000, +0.000] | 0/1/299 | 1 | 1 | -0.183 [-1.111, +0.635] |
| R_MatchedRandom_seed7 | +0.000 [-1.667, +1.667] | 3/3/294 | 1 | 1 | +0.089 [-1.292, +1.394] |
| R_MatchedRandom_seed19 | +1.000 [+0.000, +2.333] | 3/0/297 | 0.25 | 1 | +0.638 [-0.167, +1.741] |
| R_MatchedRandom_seed31 | -1.333 [-2.667, -0.333] | 0/4/296 | 0.125 | 1 | -0.963 [-2.269, +0.029] |



## Matched communication effects: abstention_lexical


| Arm−same-regime NoComm | EM Δpp [CI] | W/L/T | Exact p | Holm p | F1 Δpp [CI] |
|---|---|---|---|---|---|
| P_SelfUpdate | -1.000 [-3.333, +1.000] | 4/7/289 | 0.548828 | 1 | -0.673 [-2.826, +1.415] |
| P_Static | -1.000 [-3.333, +1.000] | 4/7/289 | 0.548828 | 1 | -0.992 [-3.116, +0.977] |
| P_Broadcast | +0.000 [-2.333, +2.333] | 6/6/288 | 1 | 1 | +0.232 [-1.711, +2.195] |
| P_Learned_seed7 | +0.000 [-2.333, +2.333] | 6/6/288 | 1 | 1 | +0.227 [-1.714, +2.187] |
| P_Learned_seed19 | +0.333 [-1.000, +2.000] | 3/2/295 | 1 | 1 | +0.421 [-0.744, +1.651] |
| P_Learned_seed31 | -0.333 [-2.000, +1.333] | 3/4/293 | 1 | 1 | -0.280 [-1.907, +1.269] |
| P_MatchedRandom_seed7 | +0.000 [-2.333, +2.333] | 6/6/288 | 1 | 1 | +0.227 [-1.714, +2.187] |
| P_MatchedRandom_seed19 | +0.000 [-1.667, +1.667] | 3/3/294 | 1 | 1 | +0.450 [-1.210, +2.054] |
| P_MatchedRandom_seed31 | +0.667 [-1.000, +2.667] | 5/3/292 | 0.726562 | 1 | +0.885 [-0.777, +2.651] |
| R_SelfUpdate | -0.667 [-2.667, +1.000] | 3/5/292 | 0.726562 | 1 | -0.199 [-1.918, +1.474] |
| R_Static | +1.000 [+0.000, +2.333] | 3/0/297 | 0.25 | 1 | +0.998 [+0.097, +2.063] |
| R_Broadcast | +0.000 [-2.000, +2.000] | 5/5/290 | 1 | 1 | -0.228 [-2.055, +1.513] |
| R_Learned_seed7 | +1.667 [-0.333, +3.667] | 7/2/291 | 0.179688 | 1 | +1.082 [-0.499, +2.700] |
| R_Learned_seed19 | +1.333 [+0.000, +3.000] | 5/1/294 | 0.21875 | 1 | +0.568 [-0.545, +1.741] |
| R_Learned_seed31 | +0.000 [-1.000, +1.000] | 1/1/298 | 1 | 1 | +0.106 [-0.783, +0.884] |
| R_MatchedRandom_seed7 | +0.333 [-1.333, +2.000] | 4/3/293 | 1 | 1 | +0.335 [-1.054, +1.635] |
| R_MatchedRandom_seed19 | +1.333 [+0.333, +2.667] | 4/0/296 | 0.125 | 1 | +0.589 [-0.346, +1.740] |
| R_MatchedRandom_seed31 | -1.000 [-2.667, +0.333] | 1/4/295 | 0.375 | 1 | -0.867 [-2.157, +0.148] |


## Difference between communication effects

The following within-question contrast is (communication−NoComm under aware) minus (communication−NoComm under original). It is not binary, so no single-pair McNemar p-value is assigned. It is a deterministic readout-by-condition contrast, not identified causal mediation through abstentions.


| Communication arm | EM interaction pp [CI] | F1 interaction pp [CI] |
|---|---|---|
| P_SelfUpdate | -1.000 [-3.333, +1.333] | -1.236 [-3.763, +1.221] |
| P_Static | -1.667 [-4.333, +1.000] | -2.218 [-5.121, +0.554] |
| P_Broadcast | -7.000 [-10.333, -4.000] | -8.701 [-12.295, -5.341] |
| P_Learned_seed7 | -7.000 [-10.333, -4.000] | -8.701 [-12.295, -5.341] |
| P_Learned_seed19 | -2.333 [-5.000, +0.000] | -3.040 [-5.789, -0.425] |
| P_Learned_seed31 | -2.000 [-4.333, +0.333] | -2.290 [-4.690, -0.056] |
| P_MatchedRandom_seed7 | -7.000 [-10.333, -4.000] | -8.701 [-12.295, -5.341] |
| P_MatchedRandom_seed19 | -1.333 [-3.333, +0.667] | -1.449 [-3.703, +0.632] |
| P_MatchedRandom_seed31 | +0.333 [-2.333, +3.000] | +0.041 [-2.481, +2.567] |
| R_SelfUpdate | -0.333 [-1.667, +1.000] | +0.162 [-1.400, +1.724] |
| R_Static | +1.000 [-0.333, +2.667] | +1.784 [+0.151, +3.533] |
| R_Broadcast | -0.333 [-1.667, +1.000] | +0.022 [-1.667, +1.622] |
| R_Learned_seed7 | +0.333 [-1.667, +2.333] | +0.684 [-1.356, +2.748] |
| R_Learned_seed19 | +0.667 [-0.667, +2.000] | +0.895 [-0.467, +2.391] |
| R_Learned_seed31 | +0.000 [+0.000, +0.000] | +0.362 [+0.000, +0.990] |
| R_MatchedRandom_seed7 | +0.000 [-1.333, +1.333] | +0.295 [-1.133, +1.690] |
| R_MatchedRandom_seed19 | +0.333 [-1.000, +2.000] | +0.589 [-1.000, +2.233] |
| R_MatchedRandom_seed31 | -0.667 [-2.000, +0.667] | -0.705 [-2.133, +0.533] |


For partitioned Broadcast, the 21-question differential readout gain (52 NoComm gains versus 31 Broadcast gains) reduces a net 24-question communication gain to 3. The point estimate attenuates 87.5% in EM, with interaction−7.000pp [−10.333,−4.000]; F1 interaction is−8.701pp [−12.295,−5.341]. These percentages describe observed arithmetic, not the proportion of a causally isolated mechanism. Shared-evidence EM is tied under both repaired readouts.

The Static partitioned point estimate reverses from+1.000 to−0.667EM points, with wide intervals. Learned partitioned effects shrink but remain slightly positive in each seed. Shared learned seed 7/19 point effects remain positive; seed 31 does not improve NoComm. The original qualitative claim of a large general partitioned communication benefit is weakened; a universal harm, benefit, equality or noninferiority conclusion is unsupported.

## Quality–cost interpretation

Partitioned seed 19 retains a descriptive trade-off:65 correct at 1,200 successful request references versus Broadcast 65 at 1,800; F1 difference−0.166pp [−1.597,+1.229]. Its 3 EM wins/3 losses against Broadcast produce p=1. Seed 7 nearly duplicates Broadcast in both quality and request demand; seed 31 is cheaper with 2 fewer correct endpoints. Shared seeds 7/19 have favorable point estimates against Broadcast with fewer requests, but intervals and seed dependence prevent a general superiority claim. No post hoc best-seed recommendation is made.

MatchedRandom is edge/quota-matched but not reliably recipient-call-matched. It matches recipient counts on only 888/1,800 pairs, with 912 mismatches. For P seed 19 it matches on 50/300 and for P seed 31 on 43/300. Therefore learned versus MatchedRandom cannot isolate a topology effect under equal receiver inference. All paired outcomes remain valid descriptive system-policy comparisons with their respective resource demands; selecting only matched rows would condition on policy-dependent behavior and is not used to manufacture a causal result. Counts and exact resource checks are in [results/physical_accounting.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/results/physical_accounting.json).

These policies were trained/selected under the original reward/readout. They were not retrained for the repaired evaluation. The study evaluates existing frozen policies, not the best possible policy under a new objective. All pre-treatment strata (type, initial unanimity, number of substantive strings and annotated-support availability) are exhaustively saved in [results/all_pre_treatment_subgroups.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/results/all_pre_treatment_subgroups.json). No post-treatment subgroup is promoted to a causal effect.
