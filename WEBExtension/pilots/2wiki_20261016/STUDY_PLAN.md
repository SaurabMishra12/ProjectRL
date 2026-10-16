# Authorized 2Wiki pilot, 16 October 2026

The user directly requested running the pilot to determine whether performance improves. This authorizes the previously proposed small live pilot, not a full benchmark, policy-training run, or automatic FinQA/scaling expansion.

## Frozen execution

- 20 publisher training questions, N=3, partitioned whole-article evidence.
- Existing Azure `gpt-oss-120b` deployment and pinned local tokenizer revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
- W1 common initial states; NoComm, private revision, static sparse, random sparse, deterministic selected sparse, complete In-Star, dense broadcast, and full-union single-agent controls.
- Same improved task prompts and scoring contract across routing arms. This study does not isolate the effect of prompt changes or evaluate trained router weights.
- At most 240 completed generations, 260 provider attempts, 2,000,000 input tokens, and 500,000 output tokens; 2,048 output tokens per call. Currency cost remains unknown rather than assuming a tariff.
- 192-token complete-atom packets and 96-token control frames. Physical cache reuse is tracked separately from logical arm references.
- A three-initial-response preflight is reused by the final run. It found no truncation, two valid outputs, and one unresolved-status/non-null-answer contract failure. Proceed unchanged; preserve errors. No prompt tuning or output repair after viewing the pilot.
- The primary planned contrast is selected-minus-random under common sink readout. Other topology/readout contrasts are exploratory. Use paired whole-question intervals; do not infer equivalence from a small or nonsignificant difference.

## Source checks and amendments before inference

The publisher's corrected `data_ids_april7.zip` was downloaded through the official repository's release link. Original archive SHA-256: `95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef`; original training member SHA-256: `b318dbafbfed51a8029718fa59be8b616600cbff675a3b587694b28c5eedfc13`.

Of 167,454 training questions, 9,583 have duplicate source titles and fail the existing unambiguous-locator contract. Exclude these before sampling, retaining the rejected initial candidate and counts. The pilot is conditional on source-valid questions.

The original full-corpus component rule was actually attempted: 156,704 of 157,871 eligible questions fall in one source-connected component. An exact 20-question component sample would therefore use only the tiny disconnected tail. Before all inference, replace that rule for this exploratory pilot with seeded hash ranking within type, proportional quotas with at least one question per type, and no shared article titles or normalized questions among selected examples. Counts: 9 compositional, 6 comparison, 4 bridge-comparison, 1 inference. No answers or model outcomes determine selection. Preserve both the unused strict-group sample and the final amended sample. Future training/development/evaluation splits need a separate overlap audit and a recorded grouping decision.

All selected questions include their annotated support in the full source pool. Eighteen have support articles split across agents. The fixed sink starts with all gold articles for one question, some for twelve, and none for seven. These are annotation-based diagnostics, not semantic sufficiency guarantees.

## Analysis conventions

Keep the official adapter's incomplete scores and identification bounds for malformed/unresolved outputs. Additionally report operational success among all 20 assigned questions, counting pipeline failures as zero, clearly distinguished from official complete scores. Never drop failed endpoints or choose an easier denominator. Retain all 20 predictions, raw provider responses, immutable source/input/code hashes, observed usage, and packet omissions. Per-arm token references overlap; do not sum them as billed usage.

Code tests passed before dispatch: 119 package tests. All source sampling and generation settings were frozen before examining benchmark accuracy. The three preflight response outcomes were inspected only for output-contract validity and truncation.

## Current execution state

The subsequent user instruction was conditional: check trends, then launch the full run if they are good. The pilot does not satisfy that condition. Nineteen of twenty assigned question records completed; one transport timeout remains unresolved. The original inference namespace must not be replayed or reset. The commands below document the original execution, not permission to repeat it. Use the offline analysis command for the preserved partial run:

```bash
python -m WEBExtension.pilots.2wiki_20261016.analyse_partial
```

See [FINDINGS.md](FINDINGS.md) for measured trends, uncertainty, and the decision against expansion.

## Original execution commands

From the repository root, the preparation scripts stream the downloaded archive without requiring the entire source-atom corpus in memory. They never invoke a model. The live runner requires the frozen operational permit reflecting the user's authorization and reads credentials into process memory only after permit and frozen-file verification; credentials are not saved or printed.

```bash
python -m WEBExtension.pilots.2wiki_20261016.prepare_pilot
python -m WEBExtension.pilots.2wiki_20261016.prepare_representative
python -u -m WEBExtension.pilots.2wiki_20261016.run_live --manifest WEBExtension/web_reasoning/artifacts/2wiki_pilot_20261016/representative_run/manifest.json --permit WEBExtension/web_reasoning/artifacts/2wiki_pilot_20261016/permit_representative.json --initial-only
python -u -m WEBExtension.pilots.2wiki_20261016.run_live --manifest WEBExtension/web_reasoning/artifacts/2wiki_pilot_20261016/representative_run/manifest.json --permit WEBExtension/web_reasoning/artifacts/2wiki_pilot_20261016/permit_representative.json
python -m WEBExtension.pilots.2wiki_20261016.analyse_pilot
```

Operational scope/limits are frozen in the run manifest and permit, with the direct user instruction recorded in `pre_inference_review.json`. This file does not authorize a different study or increased limits.
