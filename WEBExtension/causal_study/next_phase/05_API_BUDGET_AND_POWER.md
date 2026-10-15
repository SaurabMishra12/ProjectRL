# API budget and power analysis

**Status: planning only; new paid completions=0, new paid HTTP attempts=0.** The six unused E6 completion slots are not used or reassigned.

## Exact architecture and pilot proposal

The cached diagnostic freezes12 questions and reuses36 initial responses. It needs exactly24 finalizer calls, no new initial calls, no peer explanations, no extra candidate-order repetitions, no paid validation calls, and no judges. A separately approved pilot could cap successful completions at24 and HTTP attempts at30;6 attempts are reserved for explicit429 retries, not extra outcomes. Current code forbids paid execution.

Pinned-tokenizer content counts for the24 actual prepared prompts range275–573, mean375.83. The two arms are exactly matched per question. Measured E6 provider-minus-content overhead is[74, 74.0, 74]; applying its mean gives a pilot input projection of10796 tokens. It is still an estimate of server framing, not actual future usage.

E6 actually used92,594 input/34,564 output tokens in144 calls and289.45s. Mean output was240.03; p10/median/p90 were[129.6, 217.0, 371.80000000000007]. Reusing only those observed output references gives approximately5761 pilot output tokens, with an empirical-reference p10–p90 scenario of[3111, 8924]. These are not prediction intervals for the new selector. The hard output cap is49152 tokens, with2048/request. The finalizer can have different reasoning length or truncate; no new observations exist.

E6 serial timing implies a48.2s pilot reference at2.01s/call; a2s dispatch floor would be retained for any future live run. Changed prompt lengths, provider queueing and429s may increase latency. No verified deployment tariff or invoice is available: expected and actual monetary cost remain **UNKNOWN**, not zero. No model switch is planned.

## Prospective power model

Primary outcome is paired binary selector EM, not the old E6 F1 interaction. Assume a true paired EM difference of3,5 or10 percentage points and total paired discordance of0.10,0.25 or0.50. For each question, gain/loss probabilities are(q+delta)/2 and(q−delta)/2. Simulate5,000 studies per scenario and apply the exact two-sided conditional McNemar test at0.05. Recommendations require the lower Monte Carlo95% bound to reach80% or90%. Simulation uncertainty is not uncertainty about the assumed population rates. All input rates are explicitly hypothetical; no new effect or variance has been measured.

The pilot has only12 questions and the saved mechanical pool only20. Neither is a confirmatory power sample. Cached eligible yield is20/300=0.0667, Wilson95% CI[0.043567515545158274, 0.10072305951611496]. Fresh source-capable questions available under gold-free document validation: **6809**. This differs from the old1,927 complementarity-screened upper bound because the association experiment does not require both annotated supports. Neither count guarantees substantive conflicts or semantic validity.

## Cost of finding eligible fresh states

Cold confirmation must generate three initial responses for every screened source question, including those that fail eligibility. Under the point yield, each accepted question costs about45 proposal completions plus2 finalizers, not5 total calls. The number of screened questions is stochastic; point-yield projections are expectations, not spending guarantees. Predefine an absolute screening budget and stop without replacing cases when it is exhausted. Fresh response yield and semantic qualification may be worse.

- Five-point effect, discordance0.25, 80% power: n=1600 accepted questions; expected screening=24000 source questions, proposal calls=72000, finalizer calls=3200, total=75200; input≈44318467, output-reference≈18283589, serial runtime≈41.99hours. Fits current source pool at point yield: **False**. Money unknown.
- Five-point effect, discordance0.25, 90% power: n=1600 accepted questions; expected screening=24000 source questions, proposal calls=72000, finalizer calls=3200, total=75200; input≈44318467, output-reference≈18283589, serial runtime≈41.99hours. Fits current source pool at point yield: **False**. Money unknown.

Every3/5/10-point and discordance scenario,80%/90% target, yield range, actual call decomposition and source-pool flag is saved in `planning_artifacts/recommendations.csv`. Token projections use measured E6 initial-call means for proposal screening and actual prepared prompt sizes for selectors; source/context distribution shift remains unmeasured. No optional judge or explanation costs are hidden. Larger target samples can exceed the fresh source pool long before the nominal accepted-question count does.

## Decision

The minimum conditional mechanism-validation pilot is24 new completions/30 attempted requests, only after separate authorization and resolution of material source/novelty/review objections. The minimum actual expenditure justified **now is0**. No precise larger budget is justified from this phase: independent semantic annotation, a clear scientific distinction, a chosen minimum important effect, a credible fresh conflict yield and feasibility must precede commitment. Planning scenarios are not approval to run300/600 questions, train PPO, spend old quota, or change deployments.

## Exact-power refinement after the coarse grid

The frozen planning module intentionally evaluated a doubling grid. Its 1,600-question entries above are coarse simulated targets, not minimum sample sizes. The separate `power_refinement.py` integrates the binomial distribution of discordant questions and the conditional binomial rejection probability exactly. It searches nearby multiples of25 without claiming a globally minimal integer sample. These refined scenarios supersede the coarse targets for budget comparison; both original files and source snapshots are preserved.

- Five-point effect, discordance0.25, target80%: n=825, exact power=0.8043; screen12,375 source questions; 37,125 proposal+1,650 finalizer=38,775 completions, suggested attempt ceiling48,469; input-reference22,851,710, output-reference9,427,476, serial reference21.65h. Within fresh pool: False.
- Five-point effect, discordance0.25, target90%: n=1100, exact power=0.9058; screen16,500 source questions; 49,500 proposal+2,200 finalizer=51,700 completions, suggested attempt ceiling64,625; input-reference30,468,946, output-reference12,569,968, serial reference28.87h. Within fresh pool: False.

All18 refined scenarios are in `planning_artifacts/refined_power_targets.csv`. The 6,809 source questions would yield about454 mechanically eligible states at the historical point rate, before semantic qualification. This is an expectation, not a guaranteed eligible pool. Smaller effects and more discordance need more questions; larger effects can be feasible under some assumptions. No sensitivity scenario establishes what the actual effect is or authorizes spending.
