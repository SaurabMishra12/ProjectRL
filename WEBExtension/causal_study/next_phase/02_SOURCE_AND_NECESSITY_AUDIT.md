# Source provenance and evidence necessity audit

**Decision: no independently complementarity-qualified cohort is currently
available.** Article separation was implemented, but neither lexical answer
occurrence nor article-level support annotations establish necessary information
separation. All historical observations are retained. This audit made zero
inference requests and changed no historical experiment or budget artifact.

## Scope, provenance and what can be established

The audit covers the 300 unique E1–E5/E8 questions and all eight unique paid E6
questions. [source_audit.py](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/source_audit.py) reconstructs the actual saved six-document
source unions from `E1.jsonl` states, checks that shared and partitioned unions are
identical, and checks each retrieved article against `sample_gold.json` candidate
text. E6 uses the frozen manifest's six-document union and private support labels.
The historical E1–E5, corrected E3 and E6 answer/citation records supply assertion
provenance. All material inputs, source content, generated records and bounded
review inputs have recorded SHA-256 hashes in
[source_audit_manifest.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/source_artifacts/source_audit_manifest.json).

This establishes provenance to saved benchmark Wikipedia snippets. The files do
not preserve article revision timestamps or full original Web retrieval responses;
they cannot establish current Web truth or historical page-revision authenticity.
Citation addresses are zero-based title/sentence pairs. A valid address means that
text exists in the saved question union, not that it entails the proposed answer
or was visible in a particular request. Visibility is a separate request-level
audit; this audit does not convert union validity into a visibility claim.

Graph preflight supplied by the parent found project `projectRLMAIN`, generation
`2026-10-13T17:28:45Z`, with the new causal-study files not tracked and relevant
scopes empty. Direct source/JSON inspection was therefore necessary. No negative
or exhaustive codebase claim relies on empty graph results.

The complete per-question classification rows are in
[source_observations.jsonl](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/source_artifacts/source_observations.jsonl). Every row
records question identity, exact articles, content hashes, support availability,
literal reference occurrences, an unresolved independent semantic classification,
and any separate bounded provisional agent judgment. All 308 independent semantic
labels remain `NOT_ANNOTATED` and all 308 qualification fields remain
`NOT_QUALIFIED`. This is an explicit classification of what is observed versus
what still needs annotation, not a fabricated complete semantic annotation set.

## Mechanical results on the complete populations

Of the old 300 retrieved unions, 76 contain both annotated support articles and
all annotated source sentences; 164 contain one annotated article and 60 contain
neither. Thus 224/300 (74.67%) lack some annotated gold evidence. This is **not** an
estimate that 224 questions are unanswerable: alternative articles can supply the
same relation, and questions may expose redundant identifying clues.

A normalized whole-token reference sequence occurs in a source sentence for
149/300 questions, including 59/300 with an occurrence in an article that is not
annotated support. The reference occurs in an article title for 43/300. These are
lexical flags, not sufficiency labels. Common `yes`/`no` answers are flagged as
especially uninformative lexical events. A missing full reference sequence also
cannot establish absence of paraphrased evidence.

All eight E6 unions contain both annotated support articles and all annotated
sentences. All eight contain a reference sequence in a sentence, two contain one
in a non-support article, and three have reference-bearing titles. These full-pool
counts are deterministic and available in
[source_summary.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/source_artifacts/source_summary.json).

## Bounded source-only semantic assessment

The audit re-read the actual E6 source sentences, rather than adopting the old
reviewer's conclusions. The new agent author knows aggregate outcomes and cannot
claim independence or human blinding. Source assessment therefore remains
**provisional qualitative evidence**. Source facts and sentence identifiers are
preserved in [agent_source_reviews.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/source_artifacts/agent_source_reviews.json).
No question is dropped, replaced, reweighted or retrospectively qualified.

The six E6 bridge cases contain strong single-article answer-identification cues:

- `5abace6b5542996cc5e49e99`: Arthur Quinlan [0–1] names the journalist and
  interviewee Che Guevara. Che Guevara [1] verifies the Cuban-Revolution descriptor;
  separating it does not remove the named interviewee from Quinlan's article.
- `5a8080365542992097ad2fed`: Flexible-fuel vehicle [2] associates hydrogen with
  bi-fuel examples. This is also a validity concern: Hydrogen vehicle [2] includes
  combustion **or fuel-cell** conversion, and the example does not entail that
  every hydrogen vehicle has the required two-tank bi-fuel design. The question's
  binary classification requires human adjudication, not automatic entailment.
- `5abfca465542993fe9a41e65`: 2017–18 Swiss Super League [0] and its title directly
  reproduce the distinctive 121th-season descriptor. Ulrich Forte [1] supplies the
  additional coach-club-league relation.
- `5a7c651a55429935c91b5188`: Indigenous Australians [4] explicitly states the
  genetic-research habitation date, 80,000 years BP. Mulga apple [0] supplies the
  food-to-people bridge.
- `5ac540365542994611c8b470`: Loughborough University [0–1] directly matches the
  public-research-university, 1966 and 1909 clues. The Union article [0] separately
  verifies service to those students.
- `5abe4a365542993f32c2a0dc`: Chicco [0] directly matches Artsana and over-50-years
  descriptors and supplies more than 120 countries. Rosi Golan [2] supplies the
  commercials-to-brand bridge.

The two E6 comparison cases show why answer generation and full verification must
be separated. `5ac3044f5542990b17b154dc` has vocalist cues in Danny Jones [0] and
Aleksi Sihvonen [0]; either can cue the shared word, but both entities must be
checked to verify the conjunction. `5a735e9155429901807db00d` independently states
Spain in each park's article [0], with the same distinction. These observations do
not prove that every model answers from one article or that either article is
universally unnecessary. They show that annotated separation does not establish
removal of the answer-identifying information measured by answer F1/EM.

The complete unions also contain semantically relevant non-support evidence:
McFly [2] and Lies (McFly song) [2] document Danny Jones's vocal role; The Missing
Lynx [5] explicitly locates Doñana National Park in Spain; Marisma de Hinojos
[0,2] gives a connected geographical path. These are partial question-relevant
facts even when they cannot verify both entities. Conversely, Loughborough
University F.C. [0] mentions the answer in a football-club relationship; that is
not evidence of the requested Union or historical-date relation. A lexical match
can therefore represent useful evidence, partial evidence, or an irrelevant name.

## Prespecified bounded sample of the old population

Before reviewing old case semantics or citation assertions, the audit selected
two identities by SHA256(`source-audit-v1|id`) from each of four strata defined by
complete annotated support and disagreement among E5 partitioned initial answers.
The strata contain 146 incomplete-support/unanimous, 78 incomplete/conflicting,
13 complete/unanimous and 63 complete/conflicting questions. Normalization matches
the historical answer normalization. No reference correctness, final accuracy or
treatment gain enters the ranking. The selection is preserved in
[bounded_review_selection.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/source_artifacts/bounded_review_selection.json).
It is retrospective diagnostic sampling and yields no representative semantic
prevalence estimate.

- `5adde99b5542992200553ba1`: neither annotated Thunderbirds/division article was
  retrieved. The 189th Field Artillery Regiment article [1] supplies the gold
  division's relationship to the Oklahoma Guard, but no film relationship.
- `5a8b86645542997f31a41d62`: neither annotated article was retrieved. List of The
  Flash characters [5] names Matt Letscher, but the inspected union does not
  establish his target birth date or Gods and Generals relation.
- `5ae0c29c5542993d6555ec46`: Tha Trademarc [1] identifies John Cena as wrestler and
  album collaborator; the song source is absent and the full profession reference
  is not established. Partial profession sufficiency remains an annotation issue.
- `5a8f21c755429918e830d195`: selected Delany autobiographical topics and Foster
  novel descriptions do not establish comparable author-wide topic breadth.
- `5ab56e32554299637185c594`: both building articles are present, but “used for real
  estate” is ambiguous; publisher/apartments and a realty-company headquarters do
  not straightforwardly justify the reference `no`. Historical scoring is retained.
- `5ab3cecd5542992ade7c6eae`: Forbidden Creatures [2] and No Animals Were Harmed
  [2] supply distinct book subjects and plausibly require cross-article comparison.
  However, reference “his own opinions changed” comes from only the second book's
  [3] and is not a complete comparative answer. Independent reference validation is
  required before treating this as a clean complementarity case.
- `5ae655845542991bbc9760c3`: Amory Lovins [0] directly supplies institution and
  role; Soft energy path [0] supplies the coiner. Non-support Negawatt power [2]
  also gives Lovins's Rocky Mountain Institute chief-scientist relation.
- `5ae765185542995703ce8c0f`: XVideos [1] and title directly supply WGCZ registration;
  YouPorn [3] supplies the competitor relation. XHamster [3] merely supplies a
  related ranking mention, not the requested registration relation.

## Citation support: valid addresses are not verified entailment

Across E1–E5, corrected E3 and E6, the saved records contain 1,232 distinct
question/normalized-answer/citation-set assertions. Of these, 848 have nonempty
answers and citations whose addresses exist in the supplied union; 369 are
abstentions; nine normalize to an empty/malformed answer; six nonempty answers
have no citations. Repeated calls, conditions and seeds are provenance events,
not independent evidence units.

[citation_assertions_private.jsonl](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/source_artifacts/citation_assertions_private.jsonl)
retains answer text, exact cited source text, sentence hashes and raw-record
locations. All 848 semantic entailment judgments remain human-needed. A separate
[agent_citation_reviews.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/source_artifacts/agent_citation_reviews.json) preserves
40 bounded provisional assessments, including supported, partial, unsupported and
ambiguous examples. These are not independent ground truth and do not qualify any
experimental unit.

Examples establish the importance of separate labels. The park answer with both
park [0] citations supports the conjunction; Doñana [0] alone supports only one
park. The Danny/Aleksi vocalists answer is supported by both biographies, and also
by McFly [2] plus Aleksi [0], regardless of a singular/plural metric mismatch.
Loughborough University with only the football-club [0] citation lacks the required
relation. The Bi-fuel response citing Flexible-fuel vehicles in the United States
[0] has valid provenance but cites fleet size, which does not support its hydrogen
classification. A Samuel Delany breadth answer citing only Glory Lane [0] likewise
lacks the comparative evidence. The detailed two-book comparative answer citing
both book-focus [2] sentences is source-supported despite the incomplete reference.
These examples were assessed from question, answer and source content, not from
their model scores.

## Qualification protocol and feasible next populations

The ready-to-distribute [blinded source package](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/source_artifacts/blinded_source_review.jsonl),
[blinded citation package](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/source_artifacts/blinded_citation_review.jsonl), blank
[human worksheet](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/source_artifacts/human_source_annotation_template.csv), and
[annotation protocol](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/source_artifacts/BLINDED_ANNOTATION_PROTOCOL.md) cover every
question and every nonempty cited assertion. They omit outcome, arm, score and
reference information. Two humans independently resolve the question, minimal
sufficient sets, alternative paths, and question validity. Their source judgments
are locked before reference/support labels are revealed for a separate reference
validity check. A third human adjudicates disagreements. Unresolved cases remain
excluded from strict qualification and are reported.

For strict complementarity, the old source pool contains **at most 76** candidates
under the annotated-pair requirement; even those need full-union semantic review.
The other 224 can still support a different alternative-evidence population, but
cannot be silently promoted into the annotated-pair cohort. No pass fraction is
estimated from the bounded sample. In the frozen fresh-source selection, 1,935
questions passed the previous broad mechanical screen, including the eight E6
questions, leaving 1,927 before additional exposure and semantic screening. This
is a mechanical upper bound for that controlled pool, not a count of clean
complementary questions. Screening may reduce it substantially.

A defensible fresh cohort uses a fixed initial pool, hash order, complete-source
review and a fixed cap, keeping the full screening log. Do not select on evaluated
model success, existence of a correct candidate, treatment gain, or predicted
difficulty. A qualified cohort defines a new target population and needs its own
power assumptions. Gold-support construction is an oracle diagnostic, not a
deployable retrieval method.

For the proposed citation-binding diagnostic, source complementarity is not
required. The same question, candidates and source bank can support a narrow
pointer-association intervention. Yet original addresses must be independently
classified before describing a swap as degrading evidence: an original may already
be partial/unsupported, and a swapped target may accidentally improve relevance.
Without that annotation, the estimand is sensitivity to pointer associations, not
the benefit of valid source-grounded verification or general evidence access.

## Consequence for the previous placement hypothesis

The existing E6 protocol still measures relocating annotated articles under its
fixed broadcast/private-revision system. It does not identify necessary
complementarity: six of eight cases have direct answer-identification cues within
one article, while the other two have single-entity cues for the shared answer and
require a distinct full-conjunction endpoint. The hydrogen and selected old
question/reference issues add measurement concerns. These are source-construct
limitations, not an explanation proved by the answer outcomes, and they do not
make the observed interaction a population null result.

For old E8, 224 missing-support unions and 59 non-support lexical occurrences
make blanket interpretations of “annotated support absent” as “no relevant
evidence” untenable. The causal contribution of sources remains unobserved until
a valid source intervention is implemented and semantically characterized. The
appropriate scientific status remains **REVISE**.

## Local validation

Run `python3 WEBExtension/causal_study/next_phase/source_audit.py` and
`python3 -m unittest discover -s WEBExtension/causal_study/next_phase/tests -p test_source_audit.py -v`.
Ten tests verify lexical/semantic separation, source-only packet blinding,
outcome-independent case ranking, malformed citation handling, refusal to qualify
agent labels, complete 308-row alignment, source/generated hashes, no network or
provider imports, and refusal to write outputs into historical study directories.
No API client is imported and zero paid requests are made.
