# Exact interaction power and API budget

**Spend now: 0 inference calls.** The conditional experiment is not conservatively powered merely because 1000 questions were selected, and no dollar ceiling or deployed version has been approved. The six unused E6 completions remain unauthorized.

## Power for the actual question-cluster contrast

[power_budget.py](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/power_budget.py) uses the actual average of the two context-specific four-cell interactions, with two model-specific alpha 0.025 tests. It does not recycle a single paired EM discordance rate or assume the old 7 pp effect will reproduce. The frozen minimum meaningful effect is 5 pp in custom utility. With normal sensitivity, `N=ceil(((z(1-alpha/2)+z(power))*SD/delta)^2)`. The grid also varies the threshold 2/5/10pp and 80/90% target power.

- Assumed SD upper value 0.25: normal sensitivity N=238; 8,568 successful completions over both backbones; pool-fit=True; bounded Monte Carlo power=0.8724.
- Assumed SD upper value 0.5: normal sensitivity N=951; 34,236 successful completions over both backbones; pool-fit=True; bounded Monte Carlo power=0.8054.
- Assumed SD upper value 1: normal sensitivity N=3803; 136,908 successful completions over both backbones; pool-fit=False; bounded Monte Carlo power=0.80255.
- Assumed SD upper value 2: normal sensitivity N=15209; 547,524 successful completions over both backbones; pool-fit=False; bounded Monte Carlo power=not applicable.

Available disjoint confirmation pool=2366 unique questions. At SD 1 or the universal bounded-contrast SD upper value 2, the 5 pp 80% design exceeds that pool. More contexts, agents, models or seeds are not additional independent questions. A 1000-question study therefore has conditional, not distribution-free, power. The 1000 choice rounds the 951-question SD 0.5 sensitivity upward, but is not justified as robustly powered without better variance information.

For each feasible sensitivity distribution,20000 Monte Carlo replicates sample the exact question-level contrast from a declared bounded distribution and apply the actual studentized mean test. This validates the contrast/test arithmetic under those distributions, not the distribution's future truth. The full support/probabilities, Monte Carlo standard errors and variance are saved in [results/power_sensitivity.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/results/power_sensitivity.json). A fixedN forecast cannot be rescued by choosing the favorable pilot effect. No noninferiority/equivalence margin is claimed.

## Architecture accounting

Each context/backbone needs 3 initial +3 private +3 peer calls=9. Each unique question has two contexts: 18/model. Two backbones require 36 N successful completions. Initial NoComm is available without an extra call, and all four readouts are offline. Additional seed replication would multiply inference demand; no such replicate is hidden in the budget. The primary already includes the additional backbone. Pilot identities are never reused as confirmation.

- operational_pilot: 20 unique questions, 720 successes; 10% failed-attempt reserve 72; hard request quota 792.
- illustrative_minimum_confirmation_SD_0.5_80 pct: 951 unique questions, 34,236 successes; 10% failed-attempt reserve 3,424; hard request quota 37,660.
- conditional_frozen_confirmation: 1000 unique questions, 36,000 successes; 10% failed-attempt reserve 3,600; hard request quota 39,600.
- full_available_confirmation_pool: 2366 unique questions, 85,176 successes; 10% failed-attempt reserve 8,518; hard request quota 93,694.

The 20-question pilot is an operational diagnostic, not an effect-size or variance guarantee. One question pair across both models would take 36 successes and exercise plumbing only; that is not a scientific pilot. The full declared operational pilot requires 720. If both pilot and frozen confirmation were authorized later, they require 36720 successes and 40392 total attempts under separate 10% reserves. A reserve is a hard spending quota, not an assumption that 10% overhead guarantees completion. Failed/rejected/uncertain attempts consume the quota and any reported billed usage; undocumented execution halts, and refusals are not retried to purchase a better answer.

## Historical tokens, limits and latency

The prior physical accounting contains real referenced request usage. Deduplicating shared initials for the new 9-call architecture gives, per question/context: P input 5391.63 and output at least 2058.723; R input 12568.20 and output at least 2484.29. These derive from `(SelfUpdate+Broadcast-NoComm)/300`, not summing overlapping arm totals as exclusive spend. Respectively 15 and 12 success-output observations are missing; lower bounds are not imputed complete means. [results/api_budget.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/results/api_budget.json) saves source hashes and calculations.

For the 720-call pilot, P-reference input is 431331 and output at least 164698; R-reference input 1005456 and output at least 198744. These are historical arithmetic references, not MuSiQue forecasts. MuSiQue has longer source banks, a different source-access regime and a second backbone. Input sensitivities multiply these references by 1,2,4 while output remains a separate lower-bound reference. Every attempted call has a 2048 output-token cap and 32768 input-token cap; those hard resource ceilings, including the failed-attempt reserve, are saved separately. Unavailable output/reasoning usage remains unknown.

E6 observed 289.4477 summed seconds over 144 successes, approximately 2.01005 s/call;720 calls would correspond arithmetically to 1447.24 summed seconds. Neither this nor concurrent division predicts MuSiQue wall time. Measure actual per-call latency, summed latency, end-to-end wall time, prompt/output/reasoning/cache tokens, model identity, dispatches, receipts and all failures.

Current deployment, provider rate tier and immutable snapshot are unresolved. No fabricated dollar estimate is offered: cost is billed input/output tokens times the actual approved per-million rates, plus any other provider charges and required serving costs. Price/latency/token feasibility and conservative power are unresolved, so the budget does not support GO FOR SMALL PILOT today.
