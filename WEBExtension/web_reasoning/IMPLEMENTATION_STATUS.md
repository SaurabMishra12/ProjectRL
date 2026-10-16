# Local implementation status

Date: 2026-10-16. Branch: `codex/web-reasoning`. Scope: the new Web experiment package and its design/plan documents. No new benchmark accuracy or empirical scaling result has been produced.

## Verification

- Final local suite: **125 passed**: 119 package tests and six existing protocol tests. Includes all-N W1/W2 execution/resume, fixed supported-state controls, actual adaptive W4 dispatch and same-state stopping pairs, cohort isolation, interrupted-run coverage, FinQA endpoint visibility, and learned-policy overhead reporting.
- Python compilation and CLI help succeeded. Existing `WEBExtension/tests/test_protocol.py`: 6 passed.
- W1 and adaptive unit/integration tests exercise N=2,3,4,5. CLI tests block network access, reject scientific reports for mocks, reject unpermitted live dispatch, verify source/manifest integrity, and check durable resume.
- A complete 20-question synthetic N=3 W1 pilot produced 240 logical references and 190 physical mock completions. Frozen-file verification passed; repeating the workflow produced no additional transport invocations. Usage counts are fixture bytes, not provider tokens.
- All 16 historical source/result SHA-256 values in the design source manifest matched current originals.
- Independently recomputed E5 original/abstention-aware correct counts: partitioned NoComm 10/62, partitioned Broadcast 34/65; shared NoComm 69/73, shared Broadcast 70/73. These are historical diagnostics, not results of this new runner.
- Graph project `projectRLMAIN`, Tier 2, generation `2026-10-13T17:28:45Z`: targeted new-package symbol search returned no nodes; coverage marked new paths `not_tracked`. Direct reads and runnable tests provide implementation evidence; graph completeness is not claimed.

## Boundaries

Default execution is MOCK. Publisher datasets, compatible alias files, a pinned local serving tokenizer, actual deployment identity, a real pilot manifest, and explicit spending authorization are still required for live validation. No actual benchmark download or paid inference occurred. Synthetic utility-regression training in tests establishes engineering behavior only. The optional learned scorer has no bundled benchmark-trained weights.

Native allocation/scoring bounds and theoretical edge counts are distinct from measured empirical cost scaling. Python-traced routing memory excludes native tensor/GPU allocations. A valid citation establishes visibility, not semantic entailment. FinQA report-note preservation can be verbose; real-tokenizer packet omission and truncation rates must be checked in the pilot.

## Implementation rulings

1. The native managed worktree was read-only under the filesystem sandbox, so implementation used the worktree skill's feature-branch fallback in the writable checkout. Risk: unrelated checkout edits require precise staging; those edits were excluded.
2. Delivered source text appears once in `visible_sources`, with packet claims and IDs retained separately. Risk: source/prompt serializers could diverge; integration tests check source identity.
3. Upstream evaluators remain byte-exact; unavailable `ujson` is substituted with standard-library `json` in memory. Risk: unusual nonstandard JSON could deserialize differently; input files must be standard JSON.
4. The always-run adaptive diagnostic sends a fixed original-question verification request when no explicit need exists. Risk: this contrast includes the declared request policy as well as stopping.
5. The legacy prompt condition is a legacy-style answer/citation ablation, with a strict source adapter. Risk: it does not reproduce an exact historical role-specific prompt effect.
6. CLI preparation, live transport, serialization, and workflow are separate helper modules. This keeps credential access lazy and private targets outside execution. Risk: additional interfaces can drift; CLI/resume tests exercise them together.

## Final review

The fresh reviewer found five Important issues and no Critical or Minor issues. All five were reproduced by failing tests and corrected in one pass; the full 125-test suite passed afterward. The original review verdict was not ready to merge. No second review approval is claimed: the fixes are verified by regression tests and the full suite. See [findings and response](REVIEW_AND_RESPONSE.md). No merge or push has been performed.

The review's untested areas remain explicit rulings: real-data adapter/split feasibility (risk: corrections or sampling amendments), pinned-tokenizer packet fit (risk: omitted atoms), live provider/revision/usage compatibility (risk: deployment mismatch), empirical performance/scaling/generalization (risk: no improvement), and unrelated research changes (outside this review's assurance). No minor findings were deferred.
