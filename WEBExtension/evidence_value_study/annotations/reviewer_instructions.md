# Source-bound question review

Review only the assigned blinded JSONL package and this instruction sheet. The coordinator must not provide construction keys, benchmark labels, other reviewers' responses, model outputs, or the study hypothesis. You must be a qualified human reviewer. An automated system or language model cannot complete this review.

Work independently. Evaluate each displayed case on its own, in its assigned order. Do not compare related cases or use external search, benchmark lookups, or remembered facts to fill missing evidence. Titles are part of the displayed evidence. Source identifiers are neutral citation handles.

For each case, complete the matching empty label entry:

- `sufficient`: true only if the displayed sources alone justify a specific answer to the question; false if a required relationship or fact is absent; use null only while unfinished.
- `answer_from_sources`: your concise answer if sufficient; otherwise describe the missing evidence rather than guessing.
- `cited_source_ids`: the displayed neutral source identifiers supporting your judgment.
- `rationale`: explain the necessary links and the evidence supporting or missing each link. A word match alone does not establish sufficiency.
- `title_metadata_leak`: record whether titles or displayed identifiers reveal an answer or an otherwise unsupported relationship. Explain the mechanism in the rationale. Do not assume that the presence of an answer string alone proves a leak.
- `uncertainty`: describe ambiguity, conflicting statements, incomplete context, alternate valid interpretations, or limits on your assessment.
- `reviewed_at_utc`: record the actual review time.

Do not change `case_id` or `reviewer_slot`. Return only your completed labels and your own reviewer identity and qualification attestation. Do not confer with the other reviewer before both independent submissions are locked. The coordinator will arrange human adjudication afterward.
