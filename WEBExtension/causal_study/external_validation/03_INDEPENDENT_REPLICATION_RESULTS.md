# Independent reconstruction and external checks

**The ProjectRL abstention attenuation is computationally reproduced; its generalization to independent Web QA is not established.** External numeric-QA traces show ordinary readout sensitivity, but cannot identify the proposed source-sufficiency mechanism.

## Reproduced historical finding

A new standalone implementation applies all four rules to 20 E5 arms on 300 identities, producing 24000 question/readout records. It imports neither the earlier aggregator nor its scorer. All 60 old original/aware/lexical counts match the preserved reassessment, and the original raw results remain unchanged.

- `P_NoComm`: 10 / 62 / 64 / 9 correct out of 300 (original / aware / lexical / strict majority).
- `P_SelfUpdate`: 12 / 61 / 61 / 9 correct out of 300 (original / aware / lexical / strict majority).
- `P_Broadcast`: 34 / 65 / 64 / 32 correct out of 300 (original / aware / lexical / strict majority).
- `R_NoComm`: 69 / 73 / 73 / 69 correct out of 300 (original / aware / lexical / strict majority).
- `R_SelfUpdate`: 68 / 71 / 71 / 68 correct out of 300 (original / aware / lexical / strict majority).
- `R_Broadcast`: 70 / 73 / 73 / 70 correct out of 300 (original / aware / lexical / strict majority).

Partitioned Broadcast–NoComm gains are +8.000 pp original, +1.000 pp aware,0 lexical, and +7.667 pp strict majority. The aware interaction is −7.000 pp, paired question-bootstrap 95% CI[−10.333,−4.000]. The observed 87.5% attenuation describes these two estimated gains; it is not a mediation fraction or a population estimate independently confirmed here. Against the equal-update-count SelfUpdate control, the original gain is +7.333 pp and aware gain +1.333 pp; the corresponding interaction is −6.000 pp CI[−9.667,−2.667]. No new confirmatory p-value claim is made from these post-hoc questions.

The conservative rule yields 9 NoComm and 32 Broadcast correct. It preserves a large communication difference while reducing answerable accuracy relative to the aware readout. This prevents the claim that every competent readout eliminates communication benefit. This all-answerable Hotpot reference set cannot determine whether that cost buys appropriate abstention on unsupported contexts. Existing learned-policy descriptive trade-offs remain historical, not newly generalized routing superiority.

## Independent public GSM8K states

All 36000 agent states across eight configurations and three rounds were inspected. Original majority, saved final numeric grade and saved tie flags reconstruct exactly for 4000 model-question records. Re-extracting all raw responses with the pinned source parser gives zero disagreements with saved parsed values. There is one numeric parse failure and no empty raw response. Provider-refusal labels are unavailable. Thirteen raw states mention possible uncertainty phrases; this diagnostic search is not an independent semantic annotation and does not justify reclassifying those numeric decisions as abstentions.

- `05b`: original 243→244; aware identical; lexical 241→234; strict majority 211→224 (out of 500).
- `15b`: original 362→376; aware identical; lexical 351→375; strict majority 337→372 (out of 500).
- `3b`: original 442→444; aware identical; lexical 446→444; strict majority 439→444 (out of 500).
- `7b`: original 470→468; aware identical; lexical 470→469; strict majority 467→468 (out of 500).
- `gemma1b`: original 232→272; aware identical; lexical 235→272; strict majority 219→264 (out of 500).
- `gemma4b`: original 441→437; aware identical; lexical 440→438; strict majority 436→437 (out of 500).
- `llama1b`: original 257→221; aware identical; lexical 253→211; strict majority 220→203 (out of 500).
- `llama3b`: original 413→411; aware identical; lexical 410→407; strict majority 403→406 (out of 500).

All aware-minus-original readout-by-round interactions are exactly 0 under this source contract, because it already drops parsing failures and exposes no distinct abstention class. A degenerate paired bootstrap[0,0] describes this algebraic identity; it is not a powered null interval for semantic abstention behavior.

Other rules do affect the observed communication comparison. For Qwen 1.5 B, strict-majority minus original changes the initial-to-final gain by +4.2 pp, exploratory 95% CI[+2.2,+6.2]; for Llama 1 B by +3.8 pp[+1.2,+6.4]. These are source-specific numeric-QA contrasts, without a matched three-seat private-revision control or unsupported-question labels. They demonstrate generic aggregation sensitivity, which prior work already motivates. The eight configurations share 500 identities; model-wise estimates and bootstrap samples preserve those identities, and these exploratory intervals are not multiplicity-adjusted confirmations.

[results/external_gsm_results.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/results/external_gsm_results.json) includes coverage/selective risk and every model estimate. [results/external_gsm_question_readouts.jsonl](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/results/external_gsm_question_readouts.jsonl) contains 48000 decisions. [results/historical_four_readouts.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/results/historical_four_readouts.json) and [results/historical_question_readouts.jsonl](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/results/historical_question_readouts.jsonl) preserve the complete internal reanalysis. Numeric accuracy is not silently renamed exact string EM.

[Figure: historical and independent sensitivity](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/figures/readout_sensitivity.png)

## Evidence not obtained

MuSiQue structure and licensed source availability were inspected, but there are no newly generated MuSiQue agent outcomes. Its answerable/unsupported false-answer comparison remains **NOT RUN**. No independent Web-QA readout×communication×sufficiency interaction, human sufficiency validation, model superiority, or noninferiority has been established. Same-pipeline replication and an independent benchmark data inspection do not substitute for those outcomes.
