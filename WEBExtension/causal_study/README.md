# Controlled evidence study

The eight required scientific reports are in [reports](reports/). Start with [07_FINAL_RECOMMENDATION.md](reports/07_FINAL_RECOMMENDATION.md). Historical WEBExtension source and results were read only.

Run commands from `/home/saurab/Documents/projectRLMAIN` using the installed system `python3`. Local model tokenizer files are already cached; set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to avoid tokenizer network access. The saved environment versions are in `frozen/environment.json`.

## Reproduce without inference

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m pytest WEBExtension/tests WEBExtension/causal_study/tests -q
python3 WEBExtension/causal_study/offline_e8.py
python3 -m WEBExtension.causal_study.analyze
python3 -m WEBExtension.causal_study.run status
```

E8 verifies previous artifact hashes and exact initial-state links. Analysis reads existing paid outputs only; it does not request completions. Tables, figures, question-clustered statistics, power scenarios and reports04/05/07 are rebuilt under this study directory. Mock F1 is never used as an experimental outcome. Power scenarios are approximate and authorize no requests.

## Frozen design and local checks

```sh
python3 -m WEBExtension.causal_study.run freeze
python3 -m WEBExtension.causal_study.run dry-run
```

The sample is eight deterministically selected new eligible questions, six bridge and two comparison, excluding known exposed IDs/questions. Each has a fixed six-article union, two support placements, three independent initials per placement and three private/broadcast revisions from that initial state. There are 18 physical generations/question and four final deterministic-vote cells; summing four logical cell costs would double-count initials. Paid E7 is deferred; local E7 source-packet plans are in `dry_run/e7_distinctness.json`.

Existing frozen files are checked rather than overwritten. Execution-source changes fail closed on resume. `frozen/source_snapshot/` contains the exact implementation used for paid inference. `frozen/private_labels.json` is construction/scoring data, never a model payload. Gold answers appearing naturally in source sentences are legitimate evidence. Labels and supporting annotations are excluded from inference builders.

## Paid execution and recovery

The user authorized this initial pilot only, capped at150 successful completions and180 total HTTP inference attempts. The paid action is:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m WEBExtension.causal_study.run pilot
```

Do not use this command to start a new cohort. This runner always uses one `paid_pilot/` directory and one persistent budget ledger. A complete run resumes from checkpoints without another call; an uncertain request blocks any new paid call, including across restart. Never rename/delete/reset the ledger, change deployment/config, regenerate a successful output, or infer a timeout was free. A paid E7 study or a full confirmation needs separate user authorization and its own reviewed protocol.

Only explicit429 ungenerated rejections may retry, at most twice per request, honoring retry headers. Auth/client errors stop. Timeout/connection uncertainty/HTTP5xx stops. Raw successful but malformed or truncated output consumes quota and is preserved; it is not replaced with a synthetic prediction or regenerated. Refusal completions are explicit abstentions and consume quota. Provider-specific request IDs and usage are preserved when exposed. Credentials are read only into memory using the existing project helper, never written into requests or reports.

## Artifact map

- `frozen/`: exact manifests, labels, inference config, pinned tokenizer identity, source hashes/snapshots and environment versions.
- `dry_run/`:144 mocked calls, resume proof, boundary tests, assignment validation, mock outputs and E7 local distinctness plans. No provider usage here is real.
- `paid_pilot/`: global ledger, full requests, every HTTP attempt/raw response, successful response cache, atomic per-question outputs and status.
- `offline/`: E8 raw question/condition transitions, summary tables, unobservable labels, old source hashes and official-score verification.
- `analysis/`: real provider usage table, four-cell/question effects, mechanism diagnostics, bootstrap/sign-flip results, power sensitivities and scientific figures.
- `reports/`: the eight requested reports plus directly verified primary literature notes.

No authoritative Azure deployment tariff or billing export is available. Monetary cost is unknown; measured provider tokens are not a dollar invoice. The frozen deployment is an alias, not an immutable provider snapshot. Final scientific conclusions remain limited to the exact controlled pipeline and pilot population.

Final verification:114 tests and12 subtests passed. The completed pilot uses144 requests and no retry/uncertain attempts. See `COMPLETION_AUDIT.json`, `final_tests.log`, and `artifact_manifest.json` for verifiable outputs.
