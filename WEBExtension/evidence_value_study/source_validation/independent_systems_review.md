# Independent E9 systems and statistics review

Reviewed 16 October 2026, read-only, with no inference. This review covers `experiments/design.py`, `runtime.py`, `run.py`, `analysis/statistics.py`, their three test files, and the inherited pure atomic-write/statistics helpers. It assesses offline preparation, not a live-ready provider. The unconditional paid gate is deliberate and correct.

Graph Verify discovery returned no E9 functions. Exact coverage for the nine code/test/helper paths reported `not_tracked`, no recorded gaps; complete direct-source fallback was used. No repository completeness claim is made.

Current targeted suite: **19 passed** in 3.36 seconds. Additional independent probes used temporary directories and injected local transports only. Source code was not edited.

## Findings requiring small corrections

### P2 — Bind cached responses to the expected slot/request, not only their own checksum

Location: [runtime.py:148](/home/saurab/Documents/projectRLMAIN/WEBExtension/evidence_value_study/experiments/runtime.py:148), `_validate` at line 99, recovery at lines 74–97.

The cache-read branch verifies that a saved JSON body has a self-consistent checksum, then returns it without comparing its `slot`, `slot_hash`, `request`, or `prompt_sha256` with the current request. A valid saved response copied into the wrong response filename is therefore accepted silently. Independent reproduction: create slots A and B with different messages, replace A's cached body with the intact body from B, then call `complete(A, messages_A)`. The result reports `slot-B` and B's hash while consuming no new request. Self-checksumming the wrong object does not detect mistaken placement.

Minimal correction: validate the saved slot and hash against the requested slot/key, require the exact saved request and prompt hash to agree with the already frozen request record, and require `attempt_id` to correspond to the intended successful ledger event. In recovery also check `saved.attempt_id == row.attempt_id`; a successful-body path and ledger row must reference the same attempt. Reject inconsistencies, preserving all files. Add cache-swap and attempt-ID mismatch regression cases. This is an identity/integrity guard; the reported test is not a claim that historical data were corrupted.

### P2 — Keep completion-shaped malformed message bodies inside the parser failure policy

Location: [runtime.py:186](/home/saurab/Documents/projectRLMAIN/WEBExtension/evidence_value_study/experiments/runtime.py:186).

`_parsed` assumes `choice['message']` is a mapping. With `{'choices':[{'message':'invalid-schema','finish_reason':'stop'}]}` it raises `AttributeError` before `parse_response` can return MALFORMED. The backend has already recorded a successful completion. Restart reuses that same cached body and hits the same exception, preventing the question from producing its planned per-cell missingness/bounds record.

Minimal correction: schema-check message/content/refusal types before dereferencing. Preserve the original raw body in the attempt/cache and return a parsed MALFORMED state for invalid choice/message/content schemas. Successful malformed bodies still consume one completion and must not be transparently retried. Add fixtures for non-dict message, non-string content and invalid choices shapes, asserting continuation and one physical success.

### P2 — Return the prespecified EM and full-union-baseline diagnostics

Location: [statistics.py:49](/home/saurab/Documents/projectRLMAIN/WEBExtension/evidence_value_study/analysis/statistics.py:49), successful return at line 82.

`answer_score` computes both EM and F1, but `analyze_records` keeps only F1. It ignores every record's `baseline` and returns only the four-cell F1 analysis. A valid input containing all four cells plus a valid full-union baseline produces no EM or baseline fields. Protocol report 03 requires all four F1/EM cells, their contrasts and the full-union baseline.

Minimal correction: retain per-question EM and F1, compute a clearly secondary paired EM analysis using the same question assignments, and summarize the single-agent full-union baseline separately with explicit missingness. Preserve the existing primary F1 interaction and its missing-cell bounds. A missing baseline must not silently remove an otherwise complete primary four-cell question. Add tests distinguishing F1 from EM and retaining a missing baseline separately.

### P2 — Fsync the ledger's containing directory when creating it

Location: [runtime.py:63](/home/saurab/Documents/projectRLMAIN/WEBExtension/evidence_value_study/experiments/runtime.py:63).

`_append` flushes and fsyncs the ledger file, but not the directory entry when `ledger.jsonl` is first created. File data durability alone does not guarantee that a newly created filename survives a filesystem/power crash. The durable-attempt invariant needs the started event discoverable after restart before any transport executes. Later request/attempt atomic writes do not repair the crash window immediately after the initial transport dispatch.

Minimal correction: fsync the runtime directory after first ledger creation and before transport dispatch, or durably precreate the ledger during initialization. The existing `atomic_json` helper demonstrates directory fsync. Test ordering with a mocked directory-fsync hook; no destructive power-loss test was run here. This concerns power-loss durability; ordinary process-crash recovery probes below pass.

## Confirmed behavior and additional probes

- **Eleven slots per question:** six initials, four receiver updates and one full-union baseline. The predetermined sender packet is derived from the canonical complete-placement sender response, with no gold or outcome-dependent sender selection, and reused unchanged in both peer arms.
- **Matched receiver state:** within each accessibility setting, private/peer updates share identical own initial state and local documents. Accessibility changes only the receiver's source substitution; the other agents' contexts remain equal. Two local receiver documents are retained in both settings. The source union remains six.
- **Common crash windows:** the existing success-ledger-before-cache regression passes. An independent interruption after raw attempt save but before success-ledger append recovered to one attempt/one success. A simulated interruption after four initial slots resumed to exactly eleven unique slots and eleven attempts.
- **Unknown execution:** the existing timeout fixture halts across restart, retaining the started liability. Explicit 429 fixtures retry within the shared and per-slot caps; successful malformed responses are cached and charged rather than transparently retried.
- **Caps:** independent tests hit `BudgetExceeded` at one-success completion ceiling, two-attempt ceiling with zero successes, and a 767-token output reservation cap after one 384-token reservation. The current input-reservation regression also passes. Reservations are conservative across attempts and survive restart.
- **Question pairing and F1:** duplicate identities, missing numeric cell scores and MOCK records are rejected by the intended entry points. The primary formula is `A − B − C + D = (missing_peer−missing_private)−(complete_peer−complete_private)`. A complete A=1, B=0 and missing C/D example gives identification bounds [0,2]; all four missing gives [−2,2]. No complete-case substitution or missing-as-wrong scoring is used.
- **Statistical scope:** inherited finite-range score validation and question-level bootstrap/sign-flip assumptions are appropriate for the exploratory primary F1 analysis. The sign-flip test is explicitly assumption-based; it does not acquire design-based validity from alternating execution order. Mock outcomes are not presented as scientific results.
- **Paid gate:** `execute_paid` unconditionally raises. No live transport, credential access or provider invocation was used in this review. Qualification remains zero.

## Important risk already disclosed, not a present offline defect

`max_tokens=384` may be inadequate for this reasoning model's JSON/evidence output. The current deterministic abstention mock cannot measure truncation or citation-format reliability. The protocol correctly labels this unvalidated and the paid path is closed. Before any future release, freeze an engineering failure/stop rule and verify output-cap adequacy under the separately approved smoke protocol; do not adjust after looking at scientific effect direction or silently rerun consumed slots.

The 140,000-token output reservation ceiling permits at most 364 attempts at a 384-token reserve, less than the 390-attempt call ceiling; 390×384=149,760. This is conservative early stopping, already disclosed in the projection, not a cap overrun. Similarly, the input envelope uses an unverified framing/serialization allowance. Neither figure guarantees completion of all 330 scientific slots in a future live run. Partial-run bounds and durable liabilities must remain the fallback.

The frozen source manifest will need regeneration/amendment after any parent corrections, followed by the targeted suite, dry-run/resume validation, and historical-preservation check. Passing these offline checks does not satisfy the independent human annotation gate or authorize inference.
