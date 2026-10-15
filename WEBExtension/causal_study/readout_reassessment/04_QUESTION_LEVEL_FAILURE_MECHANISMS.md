# Question-level failure mechanisms

**Observable transitions are separated from causal explanations.** Every changed-correctness instance is included in [results/failure_case_index.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/results/failure_case_index.json). All 16,200 paired communication rows, all 2,378 normalized-answer readout changes and the complete sparse joint-cell memberships are saved; no favorable subset is substituted for the 300-question analyses.

A readout intervention holds the ordered response list fixed. A communication comparison can change answer identities, counts, abstentions and selected citations simultaneously. A newly correct proposal is temporal appearance, not proof it came from a delivered source. A correct proposal lost after update is observable damage, not proof of conformity. Exact gold agreement does not certify semantic source support.

## Broadcast transitions under each identical readout

### P_Broadcast: original


| Any-correct initial→final | Vote 00 | Vote 01 | Vote 10 | Vote 11 |
|---|---|---|---|---|
| 00 | 228 | 0 | 0 | 0 |
| 01 | 3 | 2 | 0 | 0 |
| 10 | 5 | 0 | 1 | 0 |
| 11 | 29 | 23 | 0 | 9 |



### P_Broadcast: abstention_aware


| Any-correct initial→final | Vote 00 | Vote 01 | Vote 10 | Vote 11 |
|---|---|---|---|---|
| 00 | 228 | 0 | 0 | 0 |
| 01 | 0 | 5 | 0 | 0 |
| 10 | 1 | 0 | 5 | 0 |
| 11 | 1 | 3 | 0 | 57 |



### P_Broadcast: abstention_lexical


| Any-correct initial→final | Vote 00 | Vote 01 | Vote 10 | Vote 11 |
|---|---|---|---|---|
| 00 | 228 | 0 | 0 | 0 |
| 01 | 0 | 5 | 0 | 0 |
| 10 | 0 | 0 | 6 | 0 |
| 11 | 2 | 1 | 0 | 58 |



### R_Broadcast: original


| Any-correct initial→final | Vote 00 | Vote 01 | Vote 10 | Vote 11 |
|---|---|---|---|---|
| 00 | 218 | 0 | 0 | 0 |
| 01 | 1 | 2 | 0 | 0 |
| 10 | 4 | 0 | 2 | 0 |
| 11 | 3 | 3 | 2 | 65 |



### R_Broadcast: abstention_aware


| Any-correct initial→final | Vote 00 | Vote 01 | Vote 10 | Vote 11 |
|---|---|---|---|---|
| 00 | 218 | 0 | 0 | 0 |
| 01 | 0 | 3 | 0 | 0 |
| 10 | 2 | 0 | 4 | 0 |
| 11 | 2 | 2 | 1 | 68 |



### R_Broadcast: abstention_lexical


| Any-correct initial→final | Vote 00 | Vote 01 | Vote 10 | Vote 11 |
|---|---|---|---|---|
| 00 | 218 | 0 | 0 | 0 |
| 01 | 0 | 3 | 0 | 0 |
| 10 | 2 | 0 | 4 | 0 |
| 11 | 2 | 2 | 1 | 68 |


Here 01 means false→true and 10 true→false. In partitioned original voting,23 gains retain correct coverage,2 gain coverage and 1 loss loses coverage. Under aware voting,5 gains gain coverage,3 gains retain it, and all 5 losses lose coverage. Under lexical-aware voting,5 newly covered gains and 1 retained-coverage gain balance 6 lost-coverage losses. Marginal coverage 67→66 hides 5 gained and 6 lost memberships.

Thus the matched residual does not establish a pure readout or generation mechanism. All 13 aware EM-changing Broadcast questions change the candidate set; none is a fixed-set selection-only counterfactual. The offline readout intervention itself supplies the true fixed-proposal contrast. Copying an answer, independently correcting it and verifying a citation remain indistinguishable as causes in many saved updates.

## Exact partitioned aware Broadcast gains and losses

Gains:
- [5a72bb045542994cef4bc3e4](/home/saurab/Documents/projectRLMAIN/WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl:14): coverage 01; 'The Conjuring film series'→'The Conjuring'; correct nodes[]→[0].
- [5a7b76225542997c3ec9719c](/home/saurab/Documents/projectRLMAIN/WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl:36): coverage 01; 'noanswer'→'The Devil Came on Horseback'; correct nodes[]→[1].
- [5a72b9155542992359bc3187](/home/saurab/Documents/projectRLMAIN/WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl:42): coverage 11; 'The Crescent'→'Acela Express'; correct nodes[1]→[1, 2].
- [5abbc7a055429931dba1451b](/home/saurab/Documents/projectRLMAIN/WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl:67): coverage 11; 'It is the first film in the trilogy.'→'first'; correct nodes[2]→[0, 1].
- [5ae5736e5542990ba0bbb2b3](/home/saurab/Documents/projectRLMAIN/WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl:87): coverage 11; 'c. 1964'→'April 1, 1949'; correct nodes[1]→[1, 2].
- [5a746abf55429929fddd8417](/home/saurab/Documents/projectRLMAIN/WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl:137): coverage 01; 'noanswer'→"Macy's"; correct nodes[]→[0].
- [5ae54e005542993aec5ec19a](/home/saurab/Documents/projectRLMAIN/WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl:230): coverage 01; 'noanswer'→'no'; correct nodes[]→[0].
- [5a87bc3f5542996432c57275](/home/saurab/Documents/projectRLMAIN/WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl:233): coverage 01; 'Cascade Locks'→'Cascade Locks and Canal'; correct nodes[]→[1, 2].

Losses:
- [5ae765185542995703ce8c0f](/home/saurab/Documents/projectRLMAIN/WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl:127): coverage 10; 'XVideos'→'noanswer'; correct nodes[1]→[].
- [5a7a1cd85542990783324e65](/home/saurab/Documents/projectRLMAIN/WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl:184): coverage 10; 'United States presidential election of 2016'→'the 2016 United States presidential election'; correct nodes[0]→[].
- [5ac15c1c5542994ab5c67cf7](/home/saurab/Documents/projectRLMAIN/WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl:201): coverage 10; 'Kingscliff, New South Wales'→'noanswer'; correct nodes[1]→[].
- [5a8f7e785542992414482ae6](/home/saurab/Documents/projectRLMAIN/WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl:260): coverage 10; 'No'→'noanswer'; correct nodes[0]→[].
- [5a88981c554299206df2b307](/home/saurab/Documents/projectRLMAIN/WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl:295): coverage 10; 'war'→'American Civil War'; correct nodes[0, 1]→[].

## All-arm joint transition inventory

Each following row summarizes all 300 questions for one arm/readout; coverage/vote labels identify exact joint cells. Complete question IDs for every cell are in [results/joint_transitions.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/results/joint_transitions.json). Repeated IDs across arms/readouts are paired observations, not independent examples.


| Arm | Readout | Coverage/vote cells (n) |
|---|---|---|
| P_SelfUpdate | original | 00/00=228; 01/00=5; 10/00=4; 11/00=51; 11/01=2; 11/11=10 |
| P_Static | original | 00/00=229; 01/00=4; 10/00=5; 11/00=46; 11/01=6; 11/10=3; 11/11=7 |
| P_Broadcast | original | 00/00=228; 01/00=3; 01/01=2; 10/00=5; 10/10=1; 11/00=29; 11/01=23; 11/11=9 |
| P_Learned_seed7 | original | 00/00=228; 01/00=3; 01/01=2; 10/00=5; 10/10=1; 11/00=29; 11/01=23; 11/11=9 |
| P_Learned_seed19 | original | 00/00=231; 01/00=1; 01/01=1; 10/00=1; 11/00=45; 11/01=11; 11/10=2; 11/11=8 |
| P_Learned_seed31 | original | 00/00=231; 01/00=2; 10/00=3; 11/00=45; 11/01=9; 11/10=2; 11/11=8 |
| P_MatchedRandom_seed7 | original | 00/00=228; 01/00=3; 01/01=2; 10/00=5; 10/10=1; 11/00=29; 11/01=23; 11/11=9 |
| P_MatchedRandom_seed19 | original | 00/00=231; 01/00=2; 10/00=3; 11/00=48; 11/01=6; 11/10=1; 11/11=9 |
| P_MatchedRandom_seed31 | original | 00/00=228; 01/00=5; 10/00=3; 11/00=48; 11/01=6; 11/10=3; 11/11=7 |
| P_SelfUpdate | abstention_aware | 00/00=228; 01/00=1; 01/01=4; 10/10=4; 11/00=4; 11/01=1; 11/10=2; 11/11=56 |
| P_Static | abstention_aware | 00/00=229; 01/01=4; 10/00=2; 10/10=3; 11/00=3; 11/10=3; 11/11=56 |
| P_Broadcast | abstention_aware | 00/00=228; 01/01=5; 10/00=1; 10/10=5; 11/00=1; 11/01=3; 11/11=57 |
| P_Learned_seed7 | abstention_aware | 00/00=228; 01/01=5; 10/00=1; 10/10=5; 11/00=1; 11/01=3; 11/11=57 |
| P_Learned_seed19 | abstention_aware | 00/00=231; 01/01=2; 10/10=1; 11/00=2; 11/01=3; 11/10=1; 11/11=60 |
| P_Learned_seed31 | abstention_aware | 00/00=231; 01/01=2; 10/00=1; 10/10=2; 11/00=2; 11/01=2; 11/10=1; 11/11=59 |
| P_MatchedRandom_seed7 | abstention_aware | 00/00=228; 01/01=5; 10/00=1; 10/10=5; 11/00=1; 11/01=3; 11/11=57 |
| P_MatchedRandom_seed19 | abstention_aware | 00/00=231; 01/01=2; 10/00=1; 10/10=2; 11/00=3; 11/01=1; 11/11=60 |
| P_MatchedRandom_seed31 | abstention_aware | 00/00=228; 01/01=5; 10/00=1; 10/10=2; 11/00=2; 11/01=2; 11/10=1; 11/11=59 |
| P_SelfUpdate | abstention_lexical | 00/00=228; 01/00=1; 01/01=4; 10/00=1; 10/10=3; 11/00=2; 11/10=4; 11/11=57 |
| P_Static | abstention_lexical | 00/00=229; 01/01=4; 10/00=1; 10/10=4; 11/00=2; 11/10=3; 11/11=57 |
| P_Broadcast | abstention_lexical | 00/00=228; 01/01=5; 10/10=6; 11/00=2; 11/01=1; 11/11=58 |
| P_Learned_seed7 | abstention_lexical | 00/00=228; 01/01=5; 10/10=6; 11/00=2; 11/01=1; 11/11=58 |
| P_Learned_seed19 | abstention_lexical | 00/00=231; 01/01=2; 10/10=1; 11/00=2; 11/01=1; 11/10=1; 11/11=62 |
| P_Learned_seed31 | abstention_lexical | 00/00=231; 01/01=2; 10/10=3; 11/00=2; 11/01=1; 11/10=1; 11/11=60 |
| P_MatchedRandom_seed7 | abstention_lexical | 00/00=228; 01/01=5; 10/10=6; 11/00=2; 11/01=1; 11/11=58 |
| P_MatchedRandom_seed19 | abstention_lexical | 00/00=231; 01/01=2; 10/10=3; 11/00=2; 11/01=1; 11/11=61 |
| P_MatchedRandom_seed31 | abstention_lexical | 00/00=228; 01/01=5; 10/00=1; 10/10=2; 11/00=2; 11/10=1; 11/11=61 |
| R_SelfUpdate | original | 00/00=218; 01/00=1; 01/01=2; 10/00=3; 10/10=2; 11/00=6; 11/01=1; 11/10=2; 11/11=65 |
| R_Static | original | 00/00=219; 01/00=2; 10/00=2; 11/00=6; 11/01=2; 11/10=3; 11/11=66 |
| R_Broadcast | original | 00/00=218; 01/00=1; 01/01=2; 10/00=4; 10/10=2; 11/00=3; 11/01=3; 11/10=2; 11/11=65 |
| R_Learned_seed7 | original | 00/00=217; 01/00=2; 01/01=2; 10/00=3; 11/00=2; 11/01=5; 11/10=3; 11/11=66 |
| R_Learned_seed19 | original | 00/00=219; 01/00=1; 01/01=1; 10/00=1; 11/00=5; 11/01=4; 11/10=3; 11/11=66 |
| R_Learned_seed31 | original | 00/00=220; 01/00=1; 10/00=2; 11/00=8; 11/10=1; 11/11=68 |
| R_MatchedRandom_seed7 | original | 00/00=219; 01/00=1; 01/01=1; 10/00=4; 11/00=5; 11/01=1; 11/10=2; 11/11=67 |
| R_MatchedRandom_seed19 | original | 00/00=219; 01/00=1; 01/01=1; 10/00=1; 11/00=5; 11/01=4; 11/10=3; 11/11=66 |
| R_MatchedRandom_seed31 | original | 00/00=220; 01/00=1; 10/00=3; 10/10=1; 11/00=6; 11/01=1; 11/10=2; 11/11=66 |
| R_SelfUpdate | abstention_aware | 00/00=218; 01/01=3; 10/00=1; 10/10=4; 11/00=5; 11/10=1; 11/11=68 |
| R_Static | abstention_aware | 00/00=219; 01/00=1; 01/01=1; 10/00=2; 11/00=3; 11/01=1; 11/11=73 |
| R_Broadcast | abstention_aware | 00/00=218; 01/01=3; 10/00=2; 10/10=4; 11/00=2; 11/01=2; 11/10=1; 11/11=68 |
| R_Learned_seed7 | abstention_aware | 00/00=217; 01/01=4; 10/00=1; 10/10=2; 11/00=2; 11/01=3; 11/11=71 |
| R_Learned_seed19 | abstention_aware | 00/00=219; 01/01=2; 10/00=1; 11/00=2; 11/01=3; 11/10=1; 11/11=72 |
| R_Learned_seed31 | abstention_aware | 00/00=220; 01/00=1; 10/00=2; 11/00=4; 11/10=1; 11/11=72 |
| R_MatchedRandom_seed7 | abstention_aware | 00/00=219; 01/00=1; 01/01=1; 10/00=2; 10/10=2; 11/00=2; 11/01=2; 11/10=1; 11/11=70 |
| R_MatchedRandom_seed19 | abstention_aware | 00/00=219; 01/00=1; 01/01=1; 10/00=1; 11/00=3; 11/01=2; 11/11=73 |
| R_MatchedRandom_seed31 | abstention_aware | 00/00=220; 01/00=1; 10/00=1; 10/10=3; 11/00=5; 11/10=1; 11/11=69 |
| R_SelfUpdate | abstention_lexical | 00/00=218; 01/01=3; 10/00=1; 10/10=4; 11/00=5; 11/10=1; 11/11=68 |
| R_Static | abstention_lexical | 00/00=219; 01/01=2; 10/00=2; 11/00=3; 11/01=1; 11/11=73 |
| R_Broadcast | abstention_lexical | 00/00=218; 01/01=3; 10/00=2; 10/10=4; 11/00=2; 11/01=2; 11/10=1; 11/11=68 |
| R_Learned_seed7 | abstention_lexical | 00/00=217; 01/01=4; 10/00=1; 10/10=2; 11/00=2; 11/01=3; 11/11=71 |
| R_Learned_seed19 | abstention_lexical | 00/00=219; 01/01=2; 10/00=1; 11/00=2; 11/01=3; 11/10=1; 11/11=72 |
| R_Learned_seed31 | abstention_lexical | 00/00=220; 01/01=1; 10/00=2; 11/00=4; 11/10=1; 11/11=72 |
| R_MatchedRandom_seed7 | abstention_lexical | 00/00=219; 01/01=2; 10/00=2; 10/10=2; 11/00=2; 11/01=2; 11/10=1; 11/11=70 |
| R_MatchedRandom_seed19 | abstention_lexical | 00/00=219; 01/01=2; 10/00=1; 11/00=3; 11/01=2; 11/11=73 |
| R_MatchedRandom_seed31 | abstention_lexical | 00/00=220; 01/01=1; 10/00=1; 10/10=3; 11/00=5; 11/10=1; 11/11=69 |


## Evidence availability and abstention

The following initial-state audit uses the union of actual E5 initial request sentences, checked against annotated support identities. Both regimes share the same source unions. Counts are descriptive; missing annotations do not prove the union cannot answer the question.


| Regime | Annotated support titles present | n | Initial agent abstentions/3n | All agents abstain | Aware correct | Any correct | Original overrides |
|---|---|---|---|---|---|---|---|
| P | 0 | 60 | 169/180 | 49 | 1 | 1 | 11 |
| P | 1 | 164 | 402/492 | 96 | 23 | 26 | 52 |
| P | 2 | 76 | 148/228 | 13 | 38 | 40 | 51 |
| R | 0 | 60 | 152/180 | 48 | 1 | 2 | 3 |
| R | 1 | 164 | 297/492 | 93 | 27 | 29 | 7 |
| R | 2 | 76 | 7/228 | 1 | 45 | 48 | 1 |


For complete annotated support, shared initial responses abstain on 7/228 agent slots, compared with 148/228 partitioned slots. These are the same 76 question IDs with different local evidence exposure, context lengths and proposal realizations. It is a striking observational diagnostic, not a randomized semantic-scarcity mediation estimate. Without complete annotated support, alternative answer-bearing sources and model knowledge remain possible. Independent sufficiency and citation-entailment labels are pending, not silently coded as negative.

The case index records abstention overrides, node ties, rejected correct minorities, new/lost correct nodes, proposal multiplicity, set/multiset/order invariance and missing annotated support. Multiple flags can describe one case; their totals are not a disjoint causal decomposition. Insufficient source support is explicitly `NOT_INDEPENDENTLY_ASSESSED`. No paid model or judge was used to fill it in.
