# Pre-inference dry-run validation

Status: **PASS_MOCK_ONLY**. Real provider calls in this phase: **0**. The integrated local study used the frozen eight-question manifest, real exact tokenizer, real persistent budget/response-cache implementation, real prompt construction, real parsing, and mocked HTTP transport only. It made144 mock requests, all18 per question; rerunning every question generated zero additional mock requests. Mock responses are marked MOCK_ONLY and are never analyzed as experimental outcomes.

Manifest checksum: `f7181aec0eeb20426d1aef9f47c97a6bb1189b9c034b20cfb7b3c8f18e8e971c`. The dry-run gate additionally binds the frozen provider configuration SHA256 and current execution-source hashes. Manifest checksums, source content, private labels, public inputs, tokenizer identity and assignment constraints are revalidated before paid inference. Corruption tests fail closed.

## Invariant evidence

1. All four E6 cells are constructed per question. Their six-document unions and exact source sentences agree; each agent has two documents. Annotated support placement compliance is100% over8 questions.
2. Co-located owner counts are3/3/2 across nodes. Support article positions are8/8 co-located and6/10 distributed. Role/support counts are recorded in `assignment_validation.json`; exact balance is impossible at n8. Local evidence varies150–694 exact tokens, so equal article counts must not be called equal length.
3. Prompt allowlist tests inject distinctive gold metadata and verify that it never enters inference. Own prior answer is generated text; natural source content may contain an answer legitimately.
4. Each placement generates its own initial states; internal cache scopes prevent cross-placement reuse without changing model-visible payloads. Both revision arms use the same three frozen initial states for that placement.
5. NoComm receives only local evidence, own state, and no peer messages. Broadcast receives two frozen peer packets per node, six directed deliveries per placement. Valid cited whole source sentences are transmitted; invalid citations never grant source access. Realized source delivery remains an outcome diagnostic because real agents may cite nothing.
6. Identical deterministic majority aggregation follows all four primary cells. There is no aggregation API request. Both NoComm and Comm receive exactly one revision per agent.
7. E7 packets have fixed receiver/state/format/one source slot and distinct whole source sentences. Six of eight sampled questions yield provisional triples using exact token matching; this is local construction, not semantic non-support validation. All paid_ready flags remain false, and E7 has no scheduled paid calls.
8. Expected physical count is144, not192 (which would double-count shared initials). Actual mock count agrees; resume adds0.
9. Question-level statistics reject duplicate question IDs and malformed/missing scores. Interaction arithmetic, paired bootstrap and exact small-sample sign flips are tested. Agent calls are not independent question replicates.
10. Budget tests cover completion and attempt exhaustion, explicit429 retry counting/headers, persistent resume, timeout/HTTP500 uncertainty halt, malformed successful outputs consuming quota, HTTP400 completion-shaped refusals consuming quota, auth rejection without retry, cache tampering, changed config, and crash recovery after an atomic saved response. Additional boundary tests exercise the actual150-success and180-attempt caps; no real HTTP transport is used.

Validation commands/results are saved in `../dry_run/tests.log` and `budget_boundary_tests.log`. The pre-inference full suite includes the existing64 WEBExtension tests plus47 study tests, all passed; added boundary tests are separately recorded. The original red states for missing provider/runtime modules and the manifest-integrity regression were observed before implementation/correction. No paid inference was used to debug these conditions.

## Limits carried into analysis

The provider alias is not an immutable snapshot. Actual billed usage and actual source delivery cannot be certified by a mock. Supporting annotations do not prove complementary necessity; article-count matching does not equalize tokens. Broadcast source content depends on frozen initial citations. E7 semantic non-support remains unresolved and its paid study is deferred. All pilot intervals will be exploratory; n8 cannot settle the paper hypothesis.

## Final verification after analysis

The combined suite completed with **114 passed and 12 subtests passed** (`../final_tests.log`). A final study-only run completed with **50 passed and 12 subtests passed** (`../final_study_tests.log`). `../COMPLETION_AUDIT.json` additionally verifies every144 raw request/response hash and intended peer-context boundary, unchanged frozen execution sources, all8 reports,144 provider request-header receipts, and exact actual usage. A literal credential scan of the new study found0 secret matches. The paid run completed with144 successes/144 attempts and no retries, rejections or uncertain requests. No further inference was used for statistics, figures, source review or report regeneration.
