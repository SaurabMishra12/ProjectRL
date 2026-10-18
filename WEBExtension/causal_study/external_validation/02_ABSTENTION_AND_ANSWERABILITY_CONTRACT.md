# Abstention, answerability and aggregation contracts

**Ignoring abstentions is not a generally safe repair.** On frozen proposals `[abstain, abstain, Paris]`, ordinary plurality abstains, while abstention-aware plurality answers Paris. If the evidence is insufficient and abstention is required, the latter has false-answer rate 100% for that constructed case. This is a tested logical counterexample, not an observed benchmark failure rate. On an answerable Paris question, the same change can improve EM. Gold is never used to choose a candidate.

## Separate the objects

- **Question answerability in the world:** a fact may have a real answer independently of this packet.
- **Source sufficiency:** the presented documents support deriving an answer; missing annotated support alone cannot prove insufficiency.
- **Expressed abstention:** a source-defined deliberate action, such as ProjectRL `noanswer` or MuSiQue `predicted_answerable=false`.
- **Provider refusal:** a distinct provider signal. Preserve it even when a readout cannot use it as a vote; do not retry it as though it were a transport failure.
- **Empty/malformed/missing output:** a formatting, extraction or availability problem, not evidence that a model assessed answerability.
- **Substantive wrong answer:** an answer attempt that fails the source evaluator; it is not abstention.

[contracts.py](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/contracts.py) implements typed proposals and independent gold-free readouts. Raw categories remain available at proposal level; system decisions also retain all-refusal and all-invalid provenance. A system abstention produced by a conservative vote is a readout action, not a newly observed model statement. The old E5 contains only substantive answers, explicit abstentions and flagged refusals; invalid-output handling is not silently retrofitted to those historical results.

## Four frozen readouts

1. Original ProjectRL normalized plurality: count explicit abstentions and refusal-empty category; select the first seat among tied normalized answer strings. This is plurality, not a strict majority requirement.
2. Abstention-aware plurality: ignore explicit abstentions and provider refusals as candidate votes; select a substantive plurality, retaining first-seat ties. With explicit abstentions but no substantive proposal, abstain. All refusals remain refusals; all invalid outputs remain invalid.
3. Abstention-aware lexical ties: the same substantive pool, with lexically smallest normalized tied string. No semantic ranking or label fitting.
4. Conservative strict majority: a substantive answer needs at least two of the original three seats agreeing; otherwise abstain. The denominator includes unavailable seats. This sacrifices some answerable recall; it is not a proven safety guarantee, since two agents may repeat an unsupported answer.

The numeric GSM8K adapter follows its original contract: numeric parsing failures are excluded from votes and grading uses a numeric tolerance of 1 e-6. Thus its original vote already behaves like an availability-aware vote; no semantic abstention policy is inferred from `None`. A normalization-empty legitimate answer such as `A` remains an answer. Lexical and strict-majority rules are additional offline policies, not purported reconstructions of source originals.

## MuSiQue-Full

The [original author evaluator](https://github.com/stonybrooknlp/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/evaluate_v1.0.py) has distinct `predicted_answer`, `predicted_answerable` and `predicted_support_idxs`. Answer EM/F1 are scored on answerable records; grouped sufficiency gates the answer score using both same-question answerability decisions. Insufficient records retain a world-answer reference: matching it does not excuse predicting answerable. Our ProjectRL-style comparator is not an official MuSiQue aggregator.

The inspected mirror has 4834 records,2417 paired IDs, exactly one true/false label per ID and identical question strings within pairs. Source counts range 15–20, not uniformly 20; all paragraph indices are unique within a record and all texts nonempty. This structural inspection does not independently certify all sufficiency labels. Original definitions and CC BY 4.0 licensing are stated by the [authors](https://github.com/stonybrooknlp/musique); seed-dataset overlap cautions also apply.

Report answerable EM/F1, unsupported false-answer rate(answer attempts divided by insufficient records), valid abstention rate, coverage(answer attempts divided by all records), selective risk(errors/answered), and separate provider/invalid rates. Report `null` when a denominator is absent. Refusal or malformed output is not credited as a semantically valid abstention in custom utility. Preserve the official metrics alongside this custom decision utility.

Two independently ordered blinded packets cover 48 contexts from 24 disjoint question identities. Both annotators' labels are **PENDING**. [annotation/README.md](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/annotation/README.md) specifies first-pass source sufficiency without gold, locked labels, reference validation, and adjudication. No completed human annotation or empirical unsupported false-answer estimate is claimed. The prospective selection deduplicates seven repeated question texts across different IDs, retaining 2410 question-text units and 2366 after the disjoint pilot/annotation reservations. Both original contexts of each retained ID remain together.
