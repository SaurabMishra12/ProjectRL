# Source sufficiency and citation annotation protocol

Prepared on 15 October 2026, before any new-cohort qualification or inference.
Historical source exploration is retrospective. The agent author has seen aggregate
outcomes and cannot claim to be an independent, blinded human annotator.

## Files and separation of duties

Give each of two independent human reviewers only `blinded_source_review.jsonl`
and a separate copy of `human_source_annotation_template.csv`. The package contains
the question, actual supplied article titles, original complete source sentences,
canonical content hashes, and an opaque review ID. It contains no reference answer,
annotated support title, condition, candidate answer, correctness, or treatment gain.
Do not give reviewers the private source observations or agent review files.

Give citation reviewers `blinded_citation_review.jsonl` only after source judgments
are locked. It contains proposed answers and the actual cited sentences, but no
reference, score, arm, agent identity, revision stage, or majority decision. Use the
full source packet to resolve pronouns only when this is explicitly recorded.
Judge the cited set first, and the full union second; do not silently substitute
uncited text when labeling citation support.

## Stage 1: independent question and source review

1. Read the question and write an independently resolved answer or `UNRESOLVED`.
   Record ambiguity, temporal interpretation and any external knowledge required.
   Assess whether the question itself has a unique defensible answer.
2. Read each source as a whole, including its title. Record the specific relation
   and sentence IDs it supplies. A name mention, overlapping answer word, or
   answer-bearing title does not by itself establish a relation.
3. List every identified minimal sufficient source set in the supplied union.
   Separately record whether one source yields a plausible answer cue and whether
   it verifies the full requested relation or conjunction. For a question about
   both entities, one entity's attribute cannot verify the other entity.
4. For a proposed two-source chain, remove each article in turn while retaining
   every alternative supplied source. Record the precise missing premise. If a
   non-annotated source supplies it, the annotated pair is not uniquely necessary.
   Do not use a model ablation, correctness, confidence, or expected difficulty.
5. Label union sufficiency `SUFFICIENT`, `PARTIAL`, `INSUFFICIENT`, or `UNRESOLVED`.
   A claim of insufficiency must identify the missing relation; absent gold
   annotation or absent literal answer text is not sufficient evidence.
6. Lock and hash both independent judgments. Only then may a curator reveal the
   dataset reference and support annotations for a separate validity review.
   A flawed or incomplete reference is flagged; historical scores are preserved.

## Stage 2: citation entailment, without correctness labels

For each proposed answer and cited set, label:

- `ENTAILS`: cited text supports the answer as an answer to the question under the
  frozen snapshot and ordinary language interpretation, with all required joins.
- `PARTIAL`: a required premise is supported but another premise or entity is
  missing. State what is missing.
- `NO_SUPPORT`: the text merely mentions the answer or describes a different
  relation, entity, time or attribute.
- `CONTRADICTS`: the cited text explicitly rules out the proposed answer.
- `UNRESOLVED`: ambiguity or specialist knowledge prevents reliable judgment.

Record source identity validity separately. A correct title/sentence address is
provenance, not entailment. No citation is `NO_CITATION`; abstention and empty
parsed response are not answer assertions. Entailment must not be inferred from
matching the private reference or from whether the evaluated model succeeded.

## Adjudication and qualification

Report the confusion matrix and raw agreement for each judgment dimension before
adjudication; report prevalence-sensitive agreement statistics only alongside the
raw counts. A third human adjudicates disagreements without seeing outcomes.
Preserve both initial judgments, reasons and sentence IDs. Unresolved cases remain
unresolved. Agent reviews are qualitative research notes and never count as either
independent reviewer, adjudicator, or gold annotation.

For a future strict complementarity cohort, require a unique valid question and
reference, adequate union, two independently reviewed necessary sources, no
single-article sufficient path, no alternative sufficient path in any supplied
article, and outcome blinding. Hash/freeze all decisions before inference. Record
every screened identity and reason, including negative and unresolved cases.

If a future study instead concerns citation-binding metadata, complementarity is
not necessary. Define that population separately and require independent labels
for original binding support and swapped binding support before calling any swap
an evidence corruption. A syntactically valid original is not necessarily sound,
and a swapped target may accidentally be more relevant.

## Sampling and generalization

The historical audit includes machine-observable fields for all 308 unique
questions. Qualitative review covers all eight E6 questions plus two old questions
selected by SHA256(`source-audit-v1|id`) in each of four annotated-support-presence
by initial-disagreement strata. This is a fixed bounded diagnostic sample, not a
random estimate of semantic prevalence and not a confirmatory holdout. No final
accuracy or treatment effect enters the ranking.

For new confirmation, start with a prespecified fresh question pool and fixed
hash order, review the complete supplied union, and keep all qualifying questions
up to a previously fixed cap. Report the screening flow. Do not search until a
desired effect, difficulty, answer-in-candidate, or accuracy profile is reached.
Changing to a qualified target population changes the estimand and requires a new
power calculation. An oracle source/annotation screen is a diagnostic population
construction procedure, not an implemented retrieval or routing method.
