# E9 causal protocol

Status: frozen candidate/mock protocol; **NOT a source-qualified pilot release**. The scientific question is provisionally distinct enough to investigate, but no independent human qualification is complete. No provider inference is permitted at the current gate. `experiments/runtime.py::execute_paid` raises before any provider access; no CLI flag bypasses it.

## Question, intervention and estimand

Does receiving the same genuine peer information rather than privately revising have greater incremental value when the receiver lacks a contextually necessary supporting document? The primary outcome is receiver answer F1 on its final revision. The primary contrast is the question-level mean of `(D-C)-(B-A)`, where A=complete/private, B=complete/peer, C=missing/private and D=missing/peer. The directional scientific hypothesis is positive interaction; the prespecified statistical test is two-sided. Report both simple effects: a positive interaction with both simple effects negative is not beneficial consultation.

The treatment is the whole peer packet: source quotes, claims, answer suggestion, uncertainty and possible distraction. It does not isolate evidence transfer from answer influence. Equal receiver revision calls do not equalize input tokens, realized reasoning, or cost. Contextual necessity does not exclude parametric knowledge or lucky answers.

## Sources and allocation

Each question has the same six original benchmark paragraphs in both placements: designated supports s1,s2 and four distractors d1–d4. Original titles and full benchmark paragraph text are retained; no answer-dependent edits, crops, oracle summaries or replacement passages are made. These are static Wikipedia-derived sources, not live Web retrieval.

Counterbalance the designated receiver over nodes0,1,2; sender is the next node. Rotate analyst/verifier/reasoner roles deterministically in blocks. Alternate retained support orientation, placement order and private/peer execution order; use three-question blocks to counterbalance the receiver source-slot order. Hash-shuffle distractor allocation and other agent source slots. In the 30 candidate blocks every receiver occurs10 times, each support orientation15 times, and each update order15 times. Receiver source slots are15/15. Role rotation covers three cyclic sender-receiver role pairs, not every pair; support orientation, placement order and update order share index parity. This is marginal balance with restricted joint coverage, not full-factorial role/order randomization. It limits heterogeneity claims without changing the within-question receiver contrast. Exact per-question assignments are in `frozen/e9_assignments.json`. They are research metadata and never passed as model-visible objects.

For retained support A, missing support B and fixed distractors:

- Complete: receiver A+B; designated sender B+d2; third agent d1+d3+d4.
- Missing: receiver A+d1; designated sender B+d2; third agent d1+d3+d4.

Counts are2/2/3 per agent in both placements. Overlap is intentional: it holds sender and third-agent contexts fixed while only the receiver source changes. The union is the same six documents. Removing B also substitutes d1, so this estimates that source-access substitution in the retained source environment, not an abstract context-length-invariant removal effect. Support-specific length and content differences remain part of the accessibility intervention and must be reported.

Neither isolated annotated support, distractors, their combinations nor metadata can be assumed insufficient. Two independent qualified human reviewers and adjudication must validate the actual text contexts and title cues before a source-qualified sample exists. Require A+B/full union sufficient, each support-only and distractor-only insufficient, full union minus either support insufficient, and exact missing-receiver contexts insufficient. Ambiguity or conflict warrants exclusion before outcomes, not model-performance-based replacement. Qualification is tied to source and assignment hashes. Any changed context needs renewed review.

## Eleven call slots and frozen packet

Generate three initial responses per placement (six calls). Roles and visible document assignments are logged. For each placement, fork the exact same cached receiver initial response into one private update and one peer update (four calls). Do not regenerate the initial response separately for either branch.

Generate the packet once from the predesignated sender's complete-placement initial response. The sender has B+d2 in both placements. Reuse that **identical packet in B and D**. No gold, correctness, disagreement, evidence-delivery outcome or answer quality enters selection. The other initial responses are diagnostics and research overhead; their generation remains charged. They cannot be cherry-picked into the packet.

Each initial response must be JSON with `answer`, up to three `{document_id,quote}` evidence entries, and `uncertain`. Quotes must occur verbatim in a sender-visible source. The frozen packet contains the sender's parsed answer, evidence and operational status; it is never replaced by an ideal message. Invalid/refused senders yield an explicit unavailable/empty information packet and remain in the assigned analysis. This is not a fabricated sender answer. Preserve the provider body.

Both update prompts have identical instructions, source schemas, own-state schema and one information-packet slot. Private receives no peer answer or evidence. The peer branch receives the frozen packet. No arbitrary filler is added to force token equality; repetitive controls could introduce another semantic intervention. The remaining slot/content length difference is measured with the pinned tokenizer and disclosed. Gold strings naturally present in original documents are intended evidence; private gold/reference fields, support flags, allocation labels, treatment names, question IDs, hashes and evaluation metadata are excluded from prompts by allowlists.

The eleventh call is single-agent Full-Evidence RAG over the exact full six-document union, with the same model/decoding/answer schema and evaluator. It is separate from the factorial estimand. It may dominate distributed inference. One full-context call is not a compute-matched single-agent upper bound.

## Inference, parsing and usage

Planned deployment reuses the existing Azure v1 `gpt-oss-120b` alias, temperature0 and pinned local tokenizer; immutable serving weights and deployment tariff remain NOT_AVAILABLE. No live identity/model-list probe was made. The offline preparation uses max_tokens384. This fits the suggested output envelope for330 successes (126720 tokens), but its adequacy for this reasoning model is unvalidated; truncation is a material engineering risk, not a demonstrated safe setting. Existing E6 mean usage is a planning reference only. No provider text is invented for unsuccessful or uncertain requests.

Parsing is identical across endpoints. Successful malformed or truncated generations consume their slot and are not automatically rerun. Explicit refusal/abstention is retained separately; nonempty uncertain answers are scored as returned. Invalid JSON or invalid source quotations is a malformed observation, with its raw answer/body preserved. Source-faithfulness flags are diagnostics; parser strictness must remain frozen rather than excluding scientifically inconvenient successful responses after scoring.

The prepared backend is mock-only. It reuses canonical hashing and directory-fsynced atomic writing from the earlier provider but not its authorization, paid directory or remaining slots. It records request/response hashes, physical attempt events, usage missingness, response IDs and timings, locks the run namespace and recovers cache writes from raw attempt artifacts. A started attempt without recoverable execution status halts across restart. Only explicit ungenerated429 fixtures retry, at most twice per slot; retry counts persist. No SDK/HTTP layer is instantiated. A live transport/reconciliation review remains required after scientific gates pass.

## Analysis and missingness

The independent unit is the question, not agent, arm or seed. All30 qualified assigned questions are analyzed; successful delivery is not an inclusion condition. Use the preserved MuSiQue answer normalization/F1 functions and maximize over publisher answer aliases. Record all four F1/EM cells, simple effects, interaction, wins/losses/ties and full-union baseline. Bootstrap10,000 whole question blocks with seed20261016. The two-sided paired sign-flip test uses10,000 Monte Carlo draws or exact enumeration when at most20 contrasts are nonzero. It assumes independent symmetric question contrasts under the null; it is **not** a design-based randomization test merely because execution order was randomized. The primary test is exploratory at N30. Secondary and mechanism comparisons are descriptive; a later confirmatory family needs multiplicity control.

Missing/failed/malformed cells are not filled by generated answers or silently complete-case filtered. Preserve all assigned questions and provide full-cohort identification bounds with each missing F1 constrained to[0,1]. With unresolved primary cells, report primary effect/p-value INSUFFICIENT_EVIDENCE, plus missingness patterns and bounds. Explicit refusal/abstention on this answerable cohort is an observed zero-answer outcome; uncertainty remains distinct. A scientific analysis function rejects mock records.

Mechanism review must distinguish source ownership, quote actually included, quote visible to receiver, source support reflected in the response, and correctness. Verbatim quote validation proves copying/provenance only. Human semantic review of the actual packet and response is required for necessary-fact delivery, unsupported claims and supported derivations. Answer agreement does not establish reliance.

## Pilot sequence and ceilings

After novelty, human qualification/adjudication, source freeze, prompt exposure and offline validity gates pass: first five prespecified questions×11=55 successes; inspect engineering correctness without using effect direction to change the hypothesis or sample. Continue the remaining25 only after this checkpoint. No extra finalizer, judge, assignment replicate, backbone or policy training. Total330 successes and390 inference attempts across all restarts; suggested initial input300000/output140000 ceilings are retained in preparation. Token reservations can stop earlier than these call ceilings. Never reuse E6's six unused slots or create a fresh paid namespace to reset spending.

No E9 paid request is executable in this deliverable because qualification is absent. The frozen30-block mock cohort is candidate-only. After human review, select the first30 qualifying candidates under the frozen priority/reserve rule before outcomes; if fewer than30 qualify, stop. Freeze exact new assignments and revalidate any changed annotation contexts. Do not describe the current40 candidates as30 valid questions. E10/E11 and200/300/600-question confirmation remain NOT_AUTHORIZED/NOT_RUN.
