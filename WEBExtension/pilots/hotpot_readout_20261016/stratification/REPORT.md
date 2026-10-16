# Gold-evidence stratification of saved E1/E5 readouts

Zero provider calls. All 300 question identities retained. E1 initial partitioned evidence defines the strata: **Disjoint-Gold 62**, **Co-located-Gold 14**, **Retrieval-Missing-Gold 224**, and their combined **Gold-Complete 76**. E5 flags agree for all 300.

**Measured conclusion:** under abstention-aware readout, E5 Disjoint-Gold NoComm reaches 29/62 correct (F1 63.6753), versus Broadcast 31/62 (F1 67.1837). Broadcast minus NoComm is +3.2258 pp EM, 95% paired interval [-4.8387, +11.2903], exact McNemar p=0.6875; F1 is +3.5084 pp, interval [-2.9434, +10.5364]. Gold-complete absolute performance is much stronger than missing-gold performance, but a communication advantage remains uncertain after readout repair.

Membership uses annotated supporting sentence locators, not answer correctness. Gold completeness does not establish semantic necessity or actual use of evidence. A missing annotated source does not make correct answering impossible: the model may know a fact or find redundant evidence. Shared-R rows use E1-P membership for a common question slice; R itself does not partition those sources.

## Partitioned E5: exact EM counts and mean F1

Each cell is `correct / n; mean F1`. F1 is on the 0–100 scale. InStar256 is an exact-request projection reusing the saved Broadcast Hub output and initial non-hub outputs, not a new inference run.

### Disjoint-Gold

| Arm | Original vote | Abstention-aware vote | Fixed Hub 1 |
|---|---:|---:|---:|
| P_NoComm | 4/62; 10.6153 | 29/62; 63.6753 | 14/62; 29.4163 |
| P_Broadcast | 15/62; 32.0001 | 31/62; 67.1837 | 18/62; 37.9992 |
| P_InStar256_reused | 8/62; 17.9694 | 31/62; 66.4403 | 18/62; 37.9992 |
| P_Learned_seed31 | 8/62; 16.8941 | 31/62; 66.4403 | 18/62; 37.0776 |

### Co-located-Gold

| Arm | Original vote | Abstention-aware vote | Fixed Hub 1 |
|---|---:|---:|---:|
| P_NoComm | 1/14; 7.1429 | 9/14; 68.5065 | 3/14; 22.1805 |
| P_Broadcast | 5/14; 35.7143 | 9/14; 68.4783 | 5/14; 41.9388 |
| P_InStar256_reused | 3/14; 21.4286 | 9/14; 68.5065 | 5/14; 41.9388 |
| P_Learned_seed31 | 3/14; 21.4286 | 9/14; 68.5065 | 5/14; 42.2222 |

### Retrieval-Missing-Gold

| Arm | Original vote | Abstention-aware vote | Fixed Hub 1 |
|---|---:|---:|---:|
| P_NoComm | 5/224; 3.3919 | 24/224; 16.7756 | 14/224; 9.3786 |
| P_Broadcast | 14/224; 8.9893 | 25/224; 17.4549 | 14/224; 10.1164 |
| P_InStar256_reused | 6/224; 4.3050 | 23/224; 16.4866 | 14/224; 10.1164 |
| P_Learned_seed31 | 6/224; 4.0615 | 23/224; 16.2431 | 13/224; 9.4265 |

### Gold-Complete

| Arm | Original vote | Abstention-aware vote | Fixed Hub 1 |
|---|---:|---:|---:|
| P_NoComm | 5/76; 9.9757 | 38/76; 64.5653 | 17/76; 28.0834 |
| P_Broadcast | 20/76; 32.6843 | 40/76; 67.4222 | 23/76; 38.7249 |
| P_InStar256_reused | 11/76; 18.6066 | 40/76; 66.8209 | 23/76; 38.7249 |
| P_Learned_seed31 | 11/76; 17.7294 | 40/76; 66.8209 | 23/76; 38.0253 |

## Partitioned paired contrasts

All differences are test minus baseline in percentage points. Parentheses contain paired 95% bootstrap intervals. P-values are exact McNemar tests for EM. These are exploratory unadjusted comparisons.

### Disjoint-Gold

| Contrast | Readout | EM difference (95% CI) | F1 difference (95% CI) | EM p |
|---|---|---:|---:|---:|
| E5:P_Broadcast-NoComm | original | +17.742 [+8.065, +27.419] | +21.385 [+11.796, +31.632] | 0.0009766 |
| E5:P_Broadcast-NoComm | abstention_aware | +3.226 [-4.839, +11.290] | +3.508 [-2.943, +10.536] | 0.6875 |
| E5:P_Broadcast-NoComm | sink1 | +6.452 [-1.613, +16.129] | +8.583 [-0.403, +17.972] | 0.2891 |
| E5:P_Learned_seed31-NoComm | original | +6.452 [+0.000, +14.516] | +6.279 [-1.325, +14.689] | 0.2188 |
| E5:P_Learned_seed31-NoComm | abstention_aware | +3.226 [-3.226, +9.677] | +2.765 [-3.226, +8.756] | 0.625 |
| E5:P_Learned_seed31-NoComm | sink1 | +6.452 [-1.613, +16.129] | +7.661 [-1.613, +17.339] | 0.2891 |
| E5_projection:P_InStar256-NoComm | original | +6.452 [+0.000, +14.516] | +7.354 [-0.403, +15.861] | 0.2188 |
| E5_projection:P_InStar256-NoComm | abstention_aware | +3.226 [-3.226, +9.677] | +2.765 [-3.226, +8.756] | 0.625 |
| E5_projection:P_InStar256-NoComm | sink1 | +6.452 [-1.613, +16.129] | +8.583 [-0.403, +17.972] | 0.2891 |
| E5:P_Learned_seed31-Broadcast | original | -11.290 [-20.968, -1.613] | -15.106 [-25.943, -4.622] | 0.06543 |
| E5:P_Learned_seed31-Broadcast | abstention_aware | +0.000 [-6.452, +6.452] | -0.743 [-6.452, +4.665] | 1 |
| E5:P_Learned_seed31-Broadcast | sink1 | +0.000 [+0.000, +0.000] | -0.922 [-2.765, +0.000] | 1 |
| E5_projection:P_InStar256-Broadcast | original | -11.290 [-20.968, -1.613] | -14.031 [-24.913, -3.578] | 0.06543 |
| E5_projection:P_InStar256-Broadcast | abstention_aware | +0.000 [-6.452, +6.452] | -0.743 [-6.452, +4.665] | 1 |
| E5_projection:P_InStar256-Broadcast | sink1 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1 |
| E1:P_CW-NoComm | original | +8.065 [+0.000, +16.129] | +10.311 [+0.818, +19.989] | 0.125 |
| E1:P_CW-NoComm | abstention_aware | +3.226 [-3.226, +9.677] | +0.766 [-3.578, +4.699] | 0.625 |
| E1:P_CW-NoComm | sink1 | +3.226 [-3.226, +9.677] | +5.535 [-2.158, +13.452] | 0.625 |
| E1:P_CCW-NoComm | original | +8.065 [+0.000, +16.129] | +9.451 [+0.218, +19.147] | 0.125 |
| E1:P_CCW-NoComm | abstention_aware | +1.613 [+0.000, +4.839] | +1.541 [+0.000, +3.886] | 1 |
| E1:P_CCW-NoComm | sink1 | +1.613 [-3.226, +6.452] | +0.147 [-6.158, +6.452] | 1 |
| E1:P_SelfUpdate-NoComm | original | +0.000 [-4.839, +4.839] | +1.075 [-3.763, +6.452] | 1 |
| E1:P_SelfUpdate-NoComm | abstention_aware | +1.613 [-3.226, +8.065] | +0.143 [-4.839, +5.161] | 1 |
| E1:P_SelfUpdate-NoComm | sink1 | +0.000 [-4.839, +4.839] | -0.591 [-4.624, +2.473] | 1 |

### Co-located-Gold

| Contrast | Readout | EM difference (95% CI) | F1 difference (95% CI) | EM p |
|---|---|---:|---:|---:|
| E5:P_Broadcast-NoComm | original | +28.571 [+7.143, +57.143] | +28.571 [+7.143, +57.143] | 0.125 |
| E5:P_Broadcast-NoComm | abstention_aware | +0.000 [+0.000, +0.000] | -0.028 [-0.085, +0.000] | 1 |
| E5:P_Broadcast-NoComm | sink1 | +14.286 [+0.000, +35.714] | +19.758 [+0.000, +41.187] | 0.5 |
| E5:P_Learned_seed31-NoComm | original | +14.286 [+0.000, +35.714] | +14.286 [+0.000, +35.714] | 0.5 |
| E5:P_Learned_seed31-NoComm | abstention_aware | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1 |
| E5:P_Learned_seed31-NoComm | sink1 | +14.286 [+0.000, +35.714] | +20.042 [+0.084, +41.470] | 0.5 |
| E5_projection:P_InStar256-NoComm | original | +14.286 [+0.000, +35.714] | +14.286 [+0.000, +35.714] | 0.5 |
| E5_projection:P_InStar256-NoComm | abstention_aware | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1 |
| E5_projection:P_InStar256-NoComm | sink1 | +14.286 [+0.000, +35.714] | +19.758 [+0.000, +41.187] | 0.5 |
| E5:P_Learned_seed31-Broadcast | original | -14.286 [-35.714, +0.000] | -14.286 [-35.714, +0.000] | 0.5 |
| E5:P_Learned_seed31-Broadcast | abstention_aware | +0.000 [+0.000, +0.000] | +0.028 [+0.000, +0.085] | 1 |
| E5:P_Learned_seed31-Broadcast | sink1 | +0.000 [+0.000, +0.000] | +0.283 [+0.000, +0.850] | 1 |
| E5_projection:P_InStar256-Broadcast | original | -14.286 [-35.714, +0.000] | -14.286 [-35.714, +0.000] | 0.5 |
| E5_projection:P_InStar256-Broadcast | abstention_aware | +0.000 [+0.000, +0.000] | +0.028 [+0.000, +0.085] | 1 |
| E5_projection:P_InStar256-Broadcast | sink1 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1 |
| E1:P_CW-NoComm | original | +14.286 [+0.000, +35.714] | +17.857 [+0.000, +39.286] | 0.5 |
| E1:P_CW-NoComm | abstention_aware | +0.000 [+0.000, +0.000] | -0.817 [-2.386, +0.000] | 1 |
| E1:P_CW-NoComm | sink1 | +7.143 [+0.000, +21.429] | +16.470 [+0.084, +35.000] | 1 |
| E1:P_CCW-NoComm | original | +7.143 [+0.000, +21.429] | +11.555 [+0.000, +28.571] | 1 |
| E1:P_CCW-NoComm | abstention_aware | +0.000 [+0.000, +0.000] | +0.024 [-0.195, +0.265] | 1 |
| E1:P_CCW-NoComm | sink1 | +7.143 [+0.000, +21.429] | +12.820 [-0.075, +32.708] | 1 |
| E1:P_SelfUpdate-NoComm | original | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1 |
| E1:P_SelfUpdate-NoComm | abstention_aware | +0.000 [+0.000, +0.000] | +0.042 [+0.000, +0.125] | 1 |
| E1:P_SelfUpdate-NoComm | sink1 | +0.000 [+0.000, +0.000] | +0.042 [+0.000, +0.125] | 1 |

### Retrieval-Missing-Gold

| Contrast | Readout | EM difference (95% CI) | F1 difference (95% CI) | EM p |
|---|---|---:|---:|---:|
| E5:P_Broadcast-NoComm | original | +4.018 [+1.339, +7.143] | +5.597 [+2.764, +8.620] | 0.01172 |
| E5:P_Broadcast-NoComm | abstention_aware | +0.446 [-1.786, +2.679] | +0.679 [-1.364, +2.731] | 1 |
| E5:P_Broadcast-NoComm | sink1 | +0.000 [-2.232, +2.232] | +0.738 [-1.295, +2.877] | 1 |
| E5:P_Learned_seed31-NoComm | original | +0.446 [-0.893, +2.232] | +0.670 [-0.670, +2.232] | 1 |
| E5:P_Learned_seed31-NoComm | abstention_aware | -0.446 [-2.232, +0.893] | -0.532 [-1.725, +0.374] | 1 |
| E5:P_Learned_seed31-NoComm | sink1 | -0.446 [-2.679, +1.786] | +0.048 [-2.183, +2.321] | 1 |
| E5_projection:P_InStar256-NoComm | original | +0.446 [-0.893, +2.232] | +0.913 [-0.446, +2.516] | 1 |
| E5_projection:P_InStar256-NoComm | abstention_aware | -0.446 [-2.232, +0.893] | -0.289 [-1.531, +0.770] | 1 |
| E5_projection:P_InStar256-NoComm | sink1 | +0.000 [-2.232, +2.232] | +0.738 [-1.295, +2.877] | 1 |
| E5:P_Learned_seed31-Broadcast | original | -3.571 [-6.250, -1.339] | -4.928 [-7.572, -2.530] | 0.007812 |
| E5:P_Learned_seed31-Broadcast | abstention_aware | -0.893 [-2.679, +0.893] | -1.212 [-3.051, +0.445] | 0.625 |
| E5:P_Learned_seed31-Broadcast | sink1 | -0.446 [-1.339, +0.000] | -0.690 [-1.826, +0.000] | 1 |
| E5_projection:P_InStar256-Broadcast | original | -3.571 [-6.250, -1.339] | -4.684 [-7.327, -2.325] | 0.007812 |
| E5_projection:P_InStar256-Broadcast | abstention_aware | -0.893 [-2.679, +0.893] | -0.968 [-2.753, +0.595] | 0.625 |
| E5_projection:P_InStar256-Broadcast | sink1 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1 |
| E1:P_CW-NoComm | original | +0.893 [+0.000, +2.232] | +1.208 [-0.337, +2.971] | 0.5 |
| E1:P_CW-NoComm | abstention_aware | +0.000 [-1.786, +1.786] | -0.183 [-1.759, +1.304] | 1 |
| E1:P_CW-NoComm | sink1 | -1.339 [-3.571, +0.446] | -1.453 [-3.232, -0.010] | 0.375 |
| E1:P_CCW-NoComm | original | +0.893 [+0.000, +2.232] | +1.716 [+0.166, +3.559] | 0.5 |
| E1:P_CCW-NoComm | abstention_aware | -0.446 [-2.232, +1.339] | -0.995 [-2.910, +0.795] | 1 |
| E1:P_CCW-NoComm | sink1 | -0.893 [-3.125, +1.339] | -0.739 [-2.689, +1.114] | 0.6875 |
| E1:P_SelfUpdate-NoComm | original | +0.000 [+0.000, +0.000] | +0.357 [-0.595, +1.310] | 1 |
| E1:P_SelfUpdate-NoComm | abstention_aware | -1.339 [-3.571, +0.446] | -1.545 [-3.581, +0.234] | 0.375 |
| E1:P_SelfUpdate-NoComm | sink1 | -1.339 [-3.571, +0.446] | -1.792 [-3.705, -0.203] | 0.375 |

### Gold-Complete

| Contrast | Readout | EM difference (95% CI) | F1 difference (95% CI) | EM p |
|---|---|---:|---:|---:|
| E5:P_Broadcast-NoComm | original | +19.737 [+11.842, +28.947] | +22.709 [+13.633, +32.217] | 6.104e-05 |
| E5:P_Broadcast-NoComm | abstention_aware | +2.632 [-3.947, +9.211] | +2.857 [-2.411, +8.732] | 0.6875 |
| E5:P_Broadcast-NoComm | sink1 | +7.895 [+0.000, +15.789] | +10.642 [+2.368, +19.448] | 0.1094 |
| E5:P_Learned_seed31-NoComm | original | +7.895 [+1.316, +15.789] | +7.754 [+0.799, +15.321] | 0.07031 |
| E5:P_Learned_seed31-NoComm | abstention_aware | +2.632 [-2.632, +7.895] | +2.256 [-2.632, +7.519] | 0.625 |
| E5:P_Learned_seed31-NoComm | sink1 | +7.895 [+0.000, +15.789] | +9.942 [+1.974, +18.767] | 0.1094 |
| E5_projection:P_InStar256-NoComm | original | +7.895 [+1.316, +15.789] | +8.631 [+1.504, +16.400] | 0.07031 |
| E5_projection:P_InStar256-NoComm | abstention_aware | +2.632 [-2.632, +7.895] | +2.256 [-2.632, +7.519] | 0.625 |
| E5_projection:P_InStar256-NoComm | sink1 | +7.895 [+0.000, +15.789] | +10.642 [+2.368, +19.448] | 0.1094 |
| E5:P_Learned_seed31-Broadcast | original | -11.842 [-21.053, -2.632] | -14.955 [-24.504, -5.478] | 0.02246 |
| E5:P_Learned_seed31-Broadcast | abstention_aware | +0.000 [-5.263, +5.263] | -0.601 [-5.405, +3.919] | 1 |
| E5:P_Learned_seed31-Broadcast | sink1 | +0.000 [+0.000, +0.000] | -0.700 [-2.256, +0.157] | 1 |
| E5_projection:P_InStar256-Broadcast | original | -11.842 [-21.053, -2.632] | -14.078 [-23.601, -4.642] | 0.02246 |
| E5_projection:P_InStar256-Broadcast | abstention_aware | +0.000 [-5.263, +5.263] | -0.601 [-5.405, +3.919] | 1 |
| E5_projection:P_InStar256-Broadcast | sink1 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1 |
| E1:P_CW-NoComm | original | +9.211 [+2.632, +17.105] | +11.701 [+3.368, +20.437] | 0.03906 |
| E1:P_CW-NoComm | abstention_aware | +2.632 [-2.632, +7.895] | +0.475 [-3.172, +3.694] | 0.625 |
| E1:P_CW-NoComm | sink1 | +3.947 [-1.316, +10.526] | +7.550 [+0.591, +15.106] | 0.375 |
| E1:P_CCW-NoComm | original | +7.895 [+1.316, +14.474] | +9.838 [+1.822, +17.942] | 0.07031 |
| E1:P_CCW-NoComm | abstention_aware | +1.316 [+0.000, +3.947] | +1.262 [+0.004, +3.247] | 1 |
| E1:P_CCW-NoComm | sink1 | +2.632 [-2.632, +7.895] | +2.481 [-3.708, +8.671] | 0.625 |
| E1:P_SelfUpdate-NoComm | original | +0.000 [-3.947, +3.947] | +0.877 [-3.070, +5.263] | 1 |
| E1:P_SelfUpdate-NoComm | abstention_aware | +1.316 [-2.632, +6.579] | +0.125 [-3.977, +4.218] | 1 |
| E1:P_SelfUpdate-NoComm | sink1 | +0.000 [-3.947, +3.947] | -0.475 [-3.852, +1.981] | 1 |

## E1: its actual saved partitioned arms

E1 contains NoComm, clockwise and counterclockwise rings, and SelfUpdate. Broadcast, learned seed31, and InStar256 were not saved in E1 and are not imputed.

| Stratum | Arm | Original vote | Abstention-aware vote | Fixed Hub 1 |
|---|---|---:|---:|---:|
| Disjoint-Gold | P_NoComm | 6/62; 13.9249 | 28/62; 64.5374 | 14/62; 30.9901 |
| Disjoint-Gold | P_CW | 11/62; 24.2358 | 30/62; 65.3036 | 16/62; 36.5255 |
| Disjoint-Gold | P_CCW | 11/62; 23.3756 | 29/62; 66.0786 | 15/62; 31.1367 |
| Disjoint-Gold | P_SelfUpdate | 6/62; 15.0002 | 29/62; 64.6808 | 14/62; 30.3987 |
| Co-located-Gold | P_NoComm | 1/14; 7.1429 | 9/14; 69.3233 | 3/14; 22.1805 |
| Co-located-Gold | P_CW | 3/14; 25.0000 | 9/14; 68.5065 | 4/14; 38.6508 |
| Co-located-Gold | P_CCW | 2/14; 18.6975 | 9/14; 69.3468 | 4/14; 35.0000 |
| Co-located-Gold | P_SelfUpdate | 1/14; 7.1429 | 9/14; 69.3651 | 3/14; 22.2222 |
| Retrieval-Missing-Gold | P_NoComm | 4/224; 2.7530 | 25/224; 17.3596 | 17/224; 10.7450 |
| Retrieval-Missing-Gold | P_CW | 6/224; 3.9610 | 25/224; 17.1769 | 14/224; 9.2915 |
| Retrieval-Missing-Gold | P_CCW | 6/224; 4.4685 | 24/224; 16.3642 | 15/224; 10.0058 |
| Retrieval-Missing-Gold | P_SelfUpdate | 4/224; 3.1101 | 22/224; 15.8143 | 14/224; 8.9534 |
| Gold-Complete | P_NoComm | 7/76; 12.6756 | 37/76; 65.4190 | 17/76; 29.3673 |
| Gold-Complete | P_CW | 14/76; 24.3766 | 39/76; 65.8936 | 20/76; 36.9170 |
| Gold-Complete | P_CCW | 13/76; 22.5138 | 38/76; 66.6807 | 19/76; 31.8484 |
| Gold-Complete | P_SelfUpdate | 7/76; 13.5528 | 38/76; 65.5437 | 17/76; 28.8925 |

## Scope and reproducibility

- The disjoint slice is larger than the 2Wiki pilot, but that alone does not establish synthesis or a significant routing benefit.
- Keep abstention-aware readout matched across arms; original-vote communication gains and readout gains are different contrasts.
- E1 and E5 are paired analyses of the same identities, not independent replications.
- Gold-Complete reuses the first two strata. Its intervals are not independent confirmations.
- Fixed-Hub projection equality remains structural; a zero-width interval for reused outputs is not a statistical equivalence test.
- Missing-gold classification is based on the retrieval pool, not proof that the omitted gold source was uniquely necessary.
- P4/P5 provenance and proxy-token caveats in the existing synthesis remain unresolved. This stratification does not validate the proposed abstract's topology-invariance or lossless-halting claims.

[All 360 summary cells, including shared R](stratified_summary.csv) · [All paired intervals and readout contrasts](stratified_results.json) · [300 question memberships](question_strata.jsonl) · [Input hashes and verification](provenance.json)

Reproduce from repository root: `python -m WEBExtension.pilots.hotpot_readout_20261016.stratify`. No API backend is imported or called.
