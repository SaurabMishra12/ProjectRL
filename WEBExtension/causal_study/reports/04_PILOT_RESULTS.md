# Pilot results

Status: **COMPLETE**. Complete unique factorial questions: **8/8**. Incomplete question IDs: none. E7: **NOT_RUN_PAID_INFERENCE_DEFERRED**. No missing or uncertain call is scored as a wrong prediction.

## Executed resources

Successful provider completions: 144; HTTP attempts: 144; retries: 0; explicit rejections: 0; uncertain requests: 0. Four E6 cells per complete question; both revision arms receive three updates per placement. The 18 physical calls/question share initial states only within placement.

Provider input tokens: 92594; output tokens: 34564; missing input/output usage rows: 0/0. Known lower bounds: 92594/34564. Actual monetary cost: **UNVERIFIED_DEPLOYMENT_PRICE**, not zero. Remaining authorized budget: 6 completions / 36 attempts. E7 is not run to exhaust quota.

Runtime for the recorded pilot session: 289.4476947784424 seconds. Successful HTTP latency sum: 141.283 seconds; mean: 0.9811345915347576. Full request usage, hashes and provider IDs: `../analysis/tables/request_usage.csv`. Provider reasoning is included only as exposed by provider usage. Returned model aliases: ['gpt-oss-120b'].

## Manipulation and paired effects

Frozen assignment compliance: 100% for equal six-document unions, source identity and intended support ownership. Two documents per agent, actual local evidence [150, 694] exact tokens. Full realized role/position counts: `../dry_run/assignment_validation.json`. Receiver-update counts and within-placement local context are held fixed; equal input-token cost is not claimed.

- Co-located private revision: F1 **35.714** points; EM 25.0%.
- Co-located broadcast: F1 **79.762** points; EM 50.0%.
- Distributed private revision: F1 **45.714** points; EM 25.0%.
- Distributed broadcast: F1 **81.548** points; EM 62.5%.

- Co-located communication effect: **44.048** F1 points; exploratory question-bootstrap95% CI [12.500, 75.595]; helped/harmed/unchanged {'helped': 4, 'harmed': 0, 'unchanged': 4}. Two-sided sign-flip reference p=0.125.
- Distributed communication effect: **35.833** F1 points; exploratory question-bootstrap95% CI [8.333, 66.667]; helped/harmed/unchanged {'helped': 4, 'harmed': 0, 'unchanged': 4}. Two-sided sign-flip reference p=0.125.
- Primary difference-in-differences: **-8.214** F1 points; exploratory question-bootstrap95% CI [-30.000, 5.357]; helped/harmed/unchanged {'helped': 1, 'harmed': 1, 'unchanged': 6}. Two-sided sign-flip reference p=1.0.

Observed direction is **opposite to the working hypothesis**. With n=8, it remains too unstable to establish a population interaction. These p-values are exploratory symmetry-based references, not physical treatment-randomization p-values, and secondary tests are not multiplicity-adjusted. A zero/insignificant result is not equivalence or noninferiority. All question effects, including harms and ties, are retained in `../analysis/tables/e6_question_effects.csv`.

## Power and uncertainty

Observed paired-interaction SD: 0.29433425397195023; baseline planning SD: 0.29433425397195023 in unit F1. The planning floor is0.25 (25 F1 points); sensitivity multiplies SD by1,1.5,2. Assumed effects are2,5,10 F1 points, never the observed pilot effect. Each grid cell uses2000 simulated studies. Recommendations use the lower Monte Carlo95% bound, a deliberately conservative coarse grid.

- Effect2 F1 points, 80% power: simulated grid n=2000, normal-approximation n=1700; SIMULATED_GRID_REACHED.
- Effect2 F1 points, 90% power: simulated grid n=None, normal-approximation n=2276; NOT_REACHED_IN_GRID.
- Effect5 F1 points, 80% power: simulated grid n=300, normal-approximation n=272; SIMULATED_GRID_REACHED.
- Effect5 F1 points, 90% power: simulated grid n=600, normal-approximation n=365; SIMULATED_GRID_REACHED.
- Effect10 F1 points, 80% power: simulated grid n=100, normal-approximation n=68; SIMULATED_GRID_REACHED.
- Effect10 F1 points, 90% power: simulated grid n=100, normal-approximation n=92; SIMULATED_GRID_REACHED.

All sensitivity scenarios and API-call projections are in `../analysis/tables/power_grid.csv` and `power_recommendations.csv`. The simulation uses a conditional-normal approximation to the paired sign-flip statistic. Empirical residual shape and variance are uncertain at n8; scaled residuals can leave the feasible interaction range. Inspect the reported violation fractions. Before confirmatory spending, validate the selected sample size with the exact planned analysis and a scientifically justified minimum important effect. No full study is launched.

The current verified source has 1927 remaining eligible unique questions after excluding this pilot and known exposures. A2,000-question scenario exceeds this pool and is not executable from the current source. A new complementarity screen can shrink the pool further; never duplicate questions to meet a target. Scenarios beyond the pool require a separately justified verified source/population and fresh power analysis.

## Data provenance

Manifest SHA256: `f7181aec0eeb20426d1aef9f47c97a6bb1189b9c034b20cfb7b3c8f18e8e971c`. Raw requests/attempts/responses, atomic question checkpoints and immutable inference settings are in `../paid_pilot/` and `../frozen/`. The mock ledger is separate and never included in these metrics. Reports are reproducible without inference using `python3 -m WEBExtension.causal_study.analyze`.
