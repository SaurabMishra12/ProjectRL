# Independent adversarial review of E9

Review date: 2026-10-16. Scope: current design, runtime, analysis, tests, candidate and annotation status, causal protocol and novelty report. This is a read-only review apart from this new record. No provider inference was performed. Parent implementation was changing during review; findings below refer to the inspected versions and must be rechecked after fixes.

Graph Verify was attempted first on projectRLMAIN, generation 2026-10-13T17:28:45Z. Searches returned no study symbols and coverage marked the study paths not_tracked. Conclusions therefore use direct current-source reads, not graph completeness claims.

## Gate assessment

The currently executable pathway is correctly limited to offline work. execute_paid raises unconditionally; the CLI does not instantiate a live provider transport. Candidate status records 40 candidates, zero qualified questions, zero completed human reviewers and paid_gate STOP. The 330-slot candidate mock must remain an operational demonstration, not a source-qualified pilot or evidence about F1.

The narrow estimand remains scientifically coherent: net usefulness of one fixed genuine packet versus one private revision, moderated by a receiver source replacement. Reuse of the same packet and within-placement receiver state removes important alternatives. It does not distinguish source evidence, suggested answers, confidence/status cues, distraction and anchoring. Source replacement also adds d1 while removing B; it is not the isolated causal effect of B absence.

## Prioritized findings and minimal corrections

### P2 — Scientific analysis entry accepts untyped records

At inspection, analysis/statistics.py:49–51 rejects only the exact string mode=MOCK. Two hand-built records with no mode are accepted and return EXPLORATORY_PILOT, cell scores and an interaction. This is not evidence that a provider was called, but it makes the stated mock/science boundary weaker than the report implies.

Minimal correction: scientific exports should require explicit REAL provenance and a qualified, frozen cohort/assignment binding. Preserve pure arithmetic unit tests separately; reject missing, unknown or synthetic record modes before scoring. A future live release also needs actual response/manifest validation rather than treating a user-supplied mode field as authentication.

### P2 — Concatenating received quotes invents a permitted span

At inspection, runtime.py:214–221 concatenates multiple peer quotes from an unseen document using newlines, and design.py:88–93 accepts any substring of that constructed text. A receiver quotation spanning two separated transmitted snippets can therefore pass validation even though it occurs in neither snippet and need not be contiguous in the original source. The reproducer with received spans alpha and beta accepted alpha-newline-beta as VALID.

Minimal correction: retain each received span separately and require a quotation to lie inside one actual transmitted span, unless the receiver locally has the full original document. Add an adversarial regression test. This affects citation-validity diagnostics and malformed-cell handling; it does not establish or refute reliance on evidence.

### P2 — Empty-packet mock does not validate substantive consultation flow

The candidate mock transport always emits an explicit abstention with no quotes. Its 330-slot checks verify counting, request identity and restart behavior, but equality of the empty packet does not exercise nonempty answer/quote transmission. The original runtime tests use the same empty-output fixture.

Minimal correction: add a separate one-question integration fixture through run_question with a predetermined nonempty sender packet, multiple source spans, an answer suggestion and assertions on both consultation requests and citation visibility. Also test malformed/refused sender propagation. Keep this fixture explicitly synthetic and outside scientific scoring.

### P2 — Counterbalance wording exceeds the implemented assignment

For the actual first 30 candidates, receiver nodes and receiver roles each occur ten times. Sender–receiver roles cover three cyclic pairs, ten each; the other three ordered distinct-role pairs are absent. Source slots are hash-shuffled, yielding 12 retained-first and 18 missing-first positions at inspection. Retained-support orientation, placement-first and update-first orders are perfectly coupled by index parity.

This does not invalidate the paired four-cell contrast. It limits the averaging population and the strength of claims about role or order invariance. Minimal correction: describe deterministic cyclic roles and hash-shuffled positions accurately; do not claim independent/full-factorial role assignment or exact alternating source positions. A newly balanced schedule would require a new explicit freeze and review of changed source bundles.

## Annotation issue reassessed after concurrent correction

The first inspected preparation had only five subset cases, which could not qualify A+d1: a distractor can supply a missing link when combined with A even if A alone and distractors alone are insufficient. The preparation was then expanded to full-union-minus-A/B and subsequently to both exact missing-receiver bundles. The later inspected private key has 720 cases: nine cases × 40 candidates × two reviewers. Every key entry carries candidate_union_sha256 and planned_assignment_sha256.

Under strict fixed-answer entailment with consistent facts, insufficiency of each five-source superset would imply insufficiency of its two-source subsets. General sufficiency under conflict/ambiguity need not be monotone, making the added exact bundles a useful safeguard. Neither their presence nor expected labels constitute completed human judgments.

The later coordinator instructions correctly require independent humans, exclusion of unresolved ambiguity/conflict, source-bound rationales, hash binding, controlled presentation and renewed review after changed allocation. Repeated question text can still reveal related cases; perfect blinding must not be claimed. Reindexing the surviving qualified cohort changes assign(candidate,index), so the current candidate mock allocation cannot silently become the paid allocation. Packet-augmented sufficiency should be reviewed as a post-treatment diagnostic, never used to select the primary analysis cohort after model responses are observed.

## Verification observed

The initial run encountered an in-flight case-count mismatch while the annotation worker was changing seven cases to nine. A subsequent run returned 25 passed and two failures: the newly added provenance and cross-span regression tests. These two failures reproduce the P2 findings above; final verification must follow implementation changes. Pure calculations confirmed the 30-question role/order counts and demonstrated that untyped records were scored.

No unrun E9 effect is supported. Conditional novelty GO, software/mock readiness and authorization for a source-qualified paid study remain three separate judgments. The current paid STOP should remain in force until genuine human qualification and the separate release checks are complete.

