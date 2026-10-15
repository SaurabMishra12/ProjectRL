# E6/E7 causal design and pilot preregistration

Protocol version: `causal-pilot-e6-matched-revision-v1`. Selection seed: `20261015`.

This protocol is fixed before the new paid pilot. The implementation in `../design.py` constructs assignments and prompts without network access. The runtime freezes the selected IDs, source/protocol hashes, inference configuration, and full request payloads before inference. Existing experiments and all their outputs remain intact. Neither the hypothesis nor sample may change in response to pilot F1.

## 1. Identification target and justified change to the requested pilot

The primary estimand is the change in the communication effect caused by redistributing the same six source articles:

\[
\Delta_q=(F1_{q,D,C}-F1_{q,D,N})-(F1_{q,L,C}-F1_{q,L,N}),\qquad
\widehat\Delta=\frac1n\sum_q\Delta_q.
\]

Here D denotes distributed support, L co-located support, C a fixed broadcast revision, and N an independent own-state revision. F1 is in percentage points, with the existing official-compatible HotpotQA normalization. The unit is one unique question.

**Critical correction:** comparing three initial NoComm responses against three communication revisions would conflate access to peer information with an extra inference update. Both primary arms therefore receive one update per agent, forked from identical initial responses *within the same placement*. NoComm revises its own state without peers; communication revises with the two other agents' frozen initial messages. The initial-majority answer remains an additional descriptive anchor at no inference cost.

This costs 18 completions per question: two placements × (three initial responses + three NoComm revisions + three broadcast revisions). The eight-question pilot requires 144 successful completions, within the authorized 150-completion and 180-attempt ceilings. The requested approximate 10-question/120-completion plan assumed an unmatched baseline. The reduction to eight is a validity correction, not an outcome-driven sample change. The six unused completion slots do not justify an underpowered E7 pilot. **Paid E7 is deferred.**

Equal update counts address an inference-compute confound. Input length and evidence semantics necessarily differ when information is delivered; this experiment does not separate informational content from every consequence of reading a longer prompt. Provider input/output tokens and latency must be reported by arm.

## 2. Sampling and population

Use the locally verified official 7,405-item HotpotQA development source, whose provenance is validated by the existing `WEBExtension/protocol.py:load_official`. The runtime records the actual source checksum. No new retrieval or provider call is required for construction.

Eligibility requires exactly two distinct annotated supporting articles present in the supplied candidate documents, at least six distinct candidate titles, valid supporting-sentence indices, and intact nonempty source-sentence strings. A missing supporting article is an exclusion, never an invitation to reconstruct or invent evidence. The exclusion reason describes missing evidence in the supplied candidates, not BM25 failure. Documents and complete sentences are retained verbatim.

Exclude known previously exposed question IDs and normalized question strings supplied from saved E1–E5 sample artifacts, controller train/validation/held-out/pilot split files, and any additional known legacy sample. Retain hashes and paths of exposure sources. Unknown earlier exposure and backbone pretraining contamination cannot be excluded. Candidate-title overlap with exposed material is reported; this is not a document-disjoint held-out test or a benchmark claim.

Within the eligible population, allocate proportional quotas to question types with deterministic largest-remainder rounding. Rank by SHA256 of `[seed, "sample", question_id]`; execution order uses a separate `[seed, "order", question_id]` hash. Deduplicate normalized questions before sampling. Source-row ordering cannot change selection. Neither answers, agent predictions, historical F1, nor expected treatment response enters the rank. Supporting annotations determine intervention eligibility only.

This population is intentionally conditional on both annotated supporting articles being available. It differs from natural retrieval, from the existing 300-question sample, and from the full HotpotQA population. Do not infer that the result applies to missing-evidence cases or the open Web.

## 3. E6 intervention

For each question, retain its two supporting articles plus four deterministically hash-ranked non-support articles. The same six-document union appears in both placements; no article is added, removed, summarized, or clipped. Each of three agents receives two articles.

- Co-located: one agent owns both supporting articles; the other two each own two distractors.
- Distributed: two different agents each own one supporting article and one distractor; the third owns two distractors.

Supporting-article order and distractor choice are independently hashed per question. The co-located owner cycles over node identities; distributed ownership uses both cyclic directions. Specialist roles use a separate block schedule, preventing the co-located support owner from always taking the same specialist role. Document order, placement execution order, and revision-arm execution order use separate deterministic random streams. Roles remain identical across paired placements.

Exact balance across all roles, ownerships, and source positions is impossible with eight questions. The local validator reports realized owner/role/document-position counts. Randomization is counterbalancing, not proof of balance. Four distractors are held fixed within each pair but reassigned as required by the intervention. No post-pilot reallocation is allowed.

Two articles per agent does not imply equal evidence length. Preserve factual content rather than crop it to force equality. Record each agent's serialized local-evidence characters, words, and exact tokenizer count when available; report within-question and placement length differences. Word counts are explicitly a proxy when an exact tokenizer is unavailable. Report actual physical provider token counts separately. Context-length allocation remains part of the implemented placement intervention and limits a narrower claim about fragmentation alone.

## 4. E6 prompts, revisions, and aggregation

`initial_messages` supplies only the question, the assigned local source sentences, the specialist instruction, a null previous response, and an empty peer-message list. No source from another agent is included. Each evidence item preserves its original document title, zero-based sentence index, and complete text.

For each placement, generate the three initial responses once. Cache/reuse is permitted only for the identical complete request, model deployment, decoding configuration, and initial evidence assignment. Never share a response across the co-located and distributed placements. The provider-level request hash binds the actual payload and configuration.

Both revisions use the same specialist instruction, local evidence, and sanitized frozen own initial answer/citations. NoComm receives an empty peer list. Broadcast receives one message from each other initial agent, containing its unverified answer proposal and the complete source sentences corresponding to its valid citations. A cited fact is eligible for transmission only if it was actually visible to that sender. Invalid or hallucinated citations do not become source evidence. There is no oracle-written explanation, gold answer proposal, adaptive routing, second communication round, or information from another revision arm. The three revisions are conceptually simultaneous: none may consume a peer's revised output.

Delivery of a peer proposal alone is not evidence delivery. Report directed messages, valid complete source facts, newly visible annotated facts, receiver citation use, and answer changes separately. An empty-citation message can transmit a proposal without transmitting an auditable supporting fact. Whether the model had memorized an answer is unobservable from this protocol.

All four primary cells use the same deterministic aggregator: unweighted normalized majority among the three revised answers; ties follow fixed node order. The initial-majority anchors use the identical rule on initial responses. The aggregator has no gold access and makes no LLM invocation. Report correct candidates rejected by majority and harmful revisions; a collective failure can arise even when one agent has both supports. Do not substitute oracle any-agent selection for practical aggregation.

Gold construction/scoring metadata lives in the manifest's separate `private.labels` mapping. Payload builders use an explicit allowlist and ignore answer/support metadata attached to question or response dictionaries. Natural source sentences can contain the target answer; that is legitimate evidence, not a reason to remove the sentence. Tests check metadata separation rather than banning an answer string from real source text.

## 5. E7 local intervention construction

E7 is implemented for local validation only. Its primary future contrast is receiver F1(relevant) minus receiver F1(redundant); relevant minus distractor is secondary. A future paid study needs a separate adequate sample and authorization; no spare-quota sample is run here.

Construct an E7 receiver from the distributed placement with exactly one annotated supporting article locally available. Its existing initial state is frozen and hashed. Select packets without consulting initial answer correctness. All arms show the same receiving agent/role, local evidence, previous answer/citations, one evidence-source slot, source format, decoding configuration, one update, and receiver-answer endpoint.

Each packet contains one complete unedited source sentence:

- Relevant: an annotated supporting sentence from the missing supporting article, absent from the receiver's visible evidence.
- Redundant: an annotated supporting sentence already visible in the receiver's own supporting article.
- Distractor: a sentence from a non-annotated-support article outside the receiver's local evidence.

The common source slot is an evidence-delivery intervention, not a learned sender agent or an oracle explanation. Treatment labels and annotations never appear in the model request. Retain title/index/text provenance. Require all three texts to be distinct, reject normalized duplicates and high lexical overlap, and reject relevant facts duplicated anywhere in the receiver's local documents. Choose a deterministic whole-sentence triple minimizing length ratio and difference, with maximum ratio 1.5; use exact serialized-packet tokens when available and otherwise an explicitly labelled sentence-word proxy. Never clip a sentence or add invented padding. Report actual residual packet and full-prompt token differences.

**Remaining validity limitation:** lack of an annotated support label is not proof that a distractor lacks helpful information. Lexical checks cannot exclude every paraphrase or shared supporting proposition. Every constructed plan therefore has `semantic_nonsupport_verified=False` and `paid_ready=False`. A blind content review and stronger matching are required before any future paid E7. Local eligibility is not an established semantic intervention or a measured knowledge-transfer result.

Mechanism analysis must separate source delivery, citation uptake, supported answer use, improved F1, exact-match recovery, harmful revision, contradiction, and unsupported claims. Citation overlap can show attributed source use; it cannot alone prove entailment or correct reasoning.

## 6. Outcomes and analysis frozen before paid inference

Primary E6 endpoint: mean question-level interaction in answer F1. Report the four revised-answer cell means, each within-placement communication effect, the interaction, and every question's paired difference. Also report helped/harmed/unchanged counts, EM separately, initial-majority anchors, source-delivery compliance, valid citation behavior, aggregation losses, physical requests, tokens, and latency. These are secondary/descriptive; no separate confirmatory discoveries are claimed.

Use paired question-level bootstrap intervals with a fixed analysis seed. Use a prespecified two-sided sign-flip reference test on question interactions (exact enumeration for eight complete questions), explicitly stating the sign-exchangeability assumption. All four potential cells are evaluated rather than randomized to different questions: this is not an exact design-based randomization p-value. Repeated agents, updates, placements, and arms are not independent observations. A significant comparison in only one placement does not establish an interaction. No noninferiority or equivalence follows from nonsignificance.

With eight questions, bootstrap tails and variance/power estimates are unstable. Treat all intervals and p-values as exploratory. No outcome-based replacement, new sample search, optional expansion, or early favorable-result stopping is permitted. If a question is incomplete because the provider or budget stops, preserve it and report completeness; do not score a missing call as an incorrect answer. Primary paired analysis includes only fully observed factorial questions and discloses all incomplete questions and reasons. Do not silently replace them.

Estimate power for a future study over a prespecified range of small/moderate interaction effects and variance inflation factors. Report 80% and 90% targets with sensitivity ranges. Do not use the pilot's observed effect as an exact prediction. A future confirmatory cohort needs frozen fresh selection and multiple assignment replicates clustered within question; seeds/replicates cannot inflate the sample size.

## 7. Local correctness gates and resource protection

Before any paid call, `validate_manifest` checks the real frozen union, exact source content, two-documents-per-agent assignment, actual support ownership, role consistency, question uniqueness, and the 18-per-question projection. Automated tests cover source corruption, candidate/support absence, exposure exclusion, input-order invariance, gold metadata separation, local-only NoComm context, common frozen initial states, sender-visible citation filtering, E7 whole-sentence identity, distinctness/novelty failures, frozen receiver states, and deterministic aggregation/interaction arithmetic.

The runtime additionally must demonstrate a complete mocked run, exact projected call count, success-preserving resume, request-hash duplicate detection, global budget enforcement, question-level statistics, and zero provider calls during dry-run validation. Full request bodies and model configuration must be frozen and auditable. A design-unit test is not a substitute for these integrated runtime checks.

Use the existing model deployment (`gpt-oss-120b`) and pinned tokenizer/configuration; do not change deployment for a favorable result. The frozen runtime records the exact output cap and all generation parameters. Freeze a single configuration across arms. Use conservative concurrency and bounded transient retries, count every HTTP inference attempt, and stop at 150 successful completions or 180 attempts. A completion that is malformed or unusable still consumes the successful-provider-completion budget; do not treat it as free. Unknown execution outcomes may have been billed: preserve them and stop rather than transparently replay. No PPO training or full evaluation is authorized. Monetary estimates must not invent a tariff absent reliable deployment-specific pricing.

## 8. Reviewer challenge and permitted conclusion

The study can establish a conditional, finite-protocol effect of controlled evidence allocation under a fixed model, message construction, revision rule, and majority aggregator. It cannot establish novel Web retrieval, superiority of learned routing, an optimal topology, natural-retrieval gains, guaranteed semantic knowledge transfer, or model-general behavior. Conditioning on known support availability and moving whole articles is artificial but intentional; it controls a mechanism that the observational retrieval regimes could not isolate.

A valid pilot may show a positive, negative, or near-zero interaction. All are retained. The implementation may merit a larger confirmatory study even with an unfavorable point estimate, whereas a favorable but invalid contrast does not merit GO. The final recommendation must depend on manipulation fidelity, resource accounting, uncertainty, and remaining identification weaknesses—not a desirable p-value.
