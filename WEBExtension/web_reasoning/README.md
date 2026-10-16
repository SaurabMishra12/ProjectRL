# SparseComm Web reasoning experiments

Implements the [approved specification](../web_reasoning_design_20261016/EXPERIMENT_DESIGN.md) and [implementation plan](../../docs/superpowers/plans/2026-10-16-web-reasoning.md). **2WikiMultihopQA is primary; FinQA is secondary. No new benchmark accuracy or empirical scaling result is supplied by this implementation.**

The default is local mock execution. Mock responses and mock tokenizer counts are engineering fixtures and cannot enter scientific reports. This package does not change historical E1–E9 data, gates, checkpoints, or manuscript claims.

## What changed

- A local abstention retains cited partial facts and precise missing relations.
- Evidence packets keep complete source sentences or table rows, provenance, headers, and units. The 192-token default includes the complete transport envelope; oversized atoms are omitted and logged.
- Fixed, random, and selected sparse routing use identical task prompts, packet caps, and a single sink update. Complete In-Star, dense, private revision, NoComm, and full-union single-agent controls are included.
- Bounded gather/request/reply consultation has an explicit `3(N-1)` data/request ceiling plus charged control frames. Matching answer strings alone cannot halt a run.
- An optional shared sender scorer has fixed-width features and no N×N state. Scoring is O(N); implemented sender sorting is O(N log N). Its training is offline utility regression from complete matched N=3 training pairs, with validation-only selection. No trained benchmark model is bundled.
- Official evaluator files and licenses are pinned by commit and SHA-256. The 2Wiki loader substitutes the standard-library JSON import for unavailable ujson in memory; upstream scoring functions and checked-in source bytes remain unchanged. FinQA program execution is checked against its official evaluator. Operand provenance is a separate structural diagnostic.

## Run a complete engineering pilot

From the repository root, using the existing Python environment:

```bash
python -m WEBExtension.web_reasoning prepare --config WEBExtension/web_reasoning/configs/mock.json --sources WEBExtension/web_reasoning/configs/mock_sources.json --output WEBExtension/web_reasoning/artifacts/mock_pilot
python -m WEBExtension.web_reasoning forecast --manifest WEBExtension/web_reasoning/artifacts/mock_pilot/manifest.json
python -m WEBExtension.web_reasoning run --manifest WEBExtension/web_reasoning/artifacts/mock_pilot/manifest.json --mode mock
python -m WEBExtension.web_reasoning verify --run WEBExtension/web_reasoning/artifacts/mock_pilot
python -m WEBExtension.web_reasoning report --run WEBExtension/web_reasoning/artifacts/mock_pilot --engineering
```

The supplied sources contain **20 explicitly synthetic, publisher-shaped test questions**, not actual 2Wiki benchmark examples. The N=3 W1 menu has 240 logical completions; physical mock invocations can be fewer because identical requests share the cache. Run the same command again to verify resume without extra executions. Reports are under `artifacts/mock_pilot/reports/` and prominently withhold scientific scores.

The standalone tests exercise N=2,3,4,5, adaptive follow-ups, exact call accounting, cache recovery, official scoring functions, and the optional scorer. No credentials, tokenizer downloads, or model requests are needed:

```bash
python -m pytest WEBExtension/web_reasoning/tests -q
python -m WEBExtension.web_reasoning --help
```

Dependencies are listed locally in [requirements.txt](requirements.txt). Existing environment packages are reused. Training/scorer code imports PyTorch only when requested; the default CLI does not construct an SDK client.

## Prepare actual benchmark inputs

Download publisher data separately from the [2Wiki repository](https://github.com/Alab-NII/2wikimultihop) or [FinQA repository](https://github.com/czyssrs/FinQA), retaining source URLs, release IDs, and original files. Preparation reads local files and checks their SHA-256 values. It makes no model calls.

Provide a source descriptor with:

- `dataset`: `2wiki` or `finqa`.
- `provenance`: `publisher` for a real run, `synthetic` for fixtures.
- `train` and `evaluation`: each an object with local `path` and actual `sha256`; `evaluation` refers to the publisher development file used for the held-out study.
- `aliases`: for real 2Wiki, the compatible local alias file's `path` and actual `sha256`; object JSON and publisher JSON-lines alias records are supported.
- `sample_sizes`: pilot 20, development 40, evaluation 200, scaling 50 by default. Pilot/development use training data; scaling is a prespecified type-proportional subset of evaluation.
- `tokenizer`: real local `path`, pinned `revision`, and tokenizer-backend `sha256` for REAL mode. Network loading and unpinned `main` revisions are rejected.
- `provider`: compatible HTTPS `base_url` and an `api_key_env` variable name. Never put a key value in the descriptor.
- Optional `placement`: `natural`, `split_support`, or `co_located`. The last two are explicitly oracle diagnostic assignments and must not be presented as natural retrieval conditions.

Use a config JSON containing the `RunConfig` fields you need to override. REAL mode requires an explicit model/deployment revision and pinned tokenizer revision. An alias without authenticated weights does not establish immutable serving weights; retain that limitation in the study report. No automatic provider identity probe is made.

Preparation writes separate public inputs, private targets, frozen assignments, balanced sinks, source/code hashes, and the call forecast. Cross-partition duplicate questions and shared sources are rejected. Infeasible grouped exact sample sizes raise a descriptive error; do not silently relax grouping or replace records after reading outcomes. Grouped partitions may require a recorded design amendment on a real corpus.

Primary `pilot`, `development`, and `evaluation` execution uses N=3. `--cohort scaling` uses the configured `team_sizes` and the frozen nested evaluation keys. Its N=3 results reuse evaluation records only when the entire contract matches; other N values use the same full source union. W2 is selected with `experiment: "W2"`; its forecast is a maximum-call schedule, with actual stopping measured separately. W4 variants are produced by the `variants` command and must be frozen individually; the factorial never runs automatically.

FinQA consumes text and rendered table rows, not raw HTML. A program must use the provided operation/constant/reference syntax. Arithmetic execution is deterministic postprocessing/scoring shared by all arms; no hidden extra model/tool loop is used. A bare numeric answer without a program cannot receive execution accuracy. Bounded numerical programs outside the safe interpreter envelope remain explicitly incomplete rather than receiving an invented official score. A row citation establishes visibility, not correct year selection or semantic support.

## Live boundary and accounting

**No paid run is authorized by building or testing this package.** Review the exact prepared manifest and forecast first. Live execution requires an independently supplied permit matching the manifest file hash and its exact finite completion, attempt, input-token, and output-token ceilings. The CLI does not generate that permit. Credentials and client initialization occur only after this check and frozen-file verification.

The client uses `max_retries=0`, and the harness alone handles bounded explicit ungenerated 429 rejections. Unknown transport status blocks the namespace across restart; successful malformed/refused responses consume their slots. [Official Python SDK documentation](https://developers.openai.com/api/reference/python) supports the raw-response and retry configuration used by the optional adapter.

Each raw response and physical attempt remains in `inference/`. Logical arm references identify the original ledger/attempt and distinguish cache hits. Repeated references are not additive physical spending. Reports retain missing usage as unknown and currency cost as unknown. The backend has persistent cumulative ceilings across all cohorts in one prepared run; a restart cannot reset them.

Engineering reports omit accuracy. Scientific reporting requires REAL records, private targets, verified sources/code, and official scoring. Sink and abstention-aware vote are both reported for every arm. Paired intervals resample whole questions; N values are presented separately. No equivalence/noninferiority claim is generated automatically.

## Local design choices

The legacy prompt condition is a **legacy-style answer/citation ablation**, not an exact replay of historical role-specific runs. Prefix cropping is an explicit content intervention; only complete parseable atoms become new citation-visible evidence. The always-run diagnostic uses a fixed original-question verification request when no sink request is available; its contrast includes that declared request policy.

Source-support annotations are diagnostic proxies. Valid locators, copied quotes, and a correct answer do not establish semantic necessity, entailment, or faithful internal source use. A positive Web showcase or empirical scaling claim requires real, cleanly rerun results.
