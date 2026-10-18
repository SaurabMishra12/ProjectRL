# External validation audit

Decision: **REVISE; no paid pilot now; current contribution category C.** Read [08_FINAL_RESEARCH_DECISION.md](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/08_FINAL_RESEARCH_DECISION.md) first.

This directory contains the eight requested reports, an independent four-readout implementation, frozen-proposal analyses, a bounded public-trace audit, original benchmark contracts, blinded annotation packages, three independent automated reviews, and a conditional prospective specification. Nothing in it enables paid inference.

## Reports

1. [01_EXTERNAL_TRACE_AVAILABILITY.md](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/01_EXTERNAL_TRACE_AVAILABILITY.md)
2. [02_ABSTENTION_AND_ANSWERABILITY_CONTRACT.md](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/02_ABSTENTION_AND_ANSWERABILITY_CONTRACT.md)
3. [03_INDEPENDENT_REPLICATION_RESULTS.md](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/03_INDEPENDENT_REPLICATION_RESULTS.md)
4. [04_RELATED_WORK_AND_NOVELTY.md](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/04_RELATED_WORK_AND_NOVELTY.md)
5. [05_FROZEN_CONFIRMATORY_DESIGN.md](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/05_FROZEN_CONFIRMATORY_DESIGN.md)
6. [06_API_AND_POWER_BUDGET.md](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/06_API_AND_POWER_BUDGET.md)
7. [07_ADVERSARIAL_REVIEW.md](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/07_ADVERSARIAL_REVIEW.md)
8. [08_FINAL_RESEARCH_DECISION.md](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/08_FINAL_RESEARCH_DECISION.md)

## Offline reproduction

From `/home/saurab/Documents/projectRLMAIN`, run sequentially:

```sh
python3 -m WEBExtension.causal_study.external_validation.reanalyse
python3 -m WEBExtension.causal_study.external_validation.power_budget
python3 -m WEBExtension.causal_study.external_validation.design
python3 -m WEBExtension.causal_study.external_validation.write_reports
python3 -m pytest WEBExtension/causal_study/external_validation/tests -q
```

Public source snapshots are already local. Re-fetching is unnecessary and would require network access; `fetch_public.py` performs public GETs only, never inference. Its original bounded selection and fetch receipts remain evidence. It is not a one-command complete environment installer. Full upstream repositories and prediction archives were not cloned wholesale.

After analyses/tests, `python3 -m WEBExtension.causal_study.external_validation.verify` checks preservation, source and design hashes, labels remaining pending, count alignment, report links, and emits the completion/manifest records. It never rewrites historical outputs. New tests cover meaningful failure boundaries, refusal semantics, benchmark parsing, question clustering and power assumptions.

Do not send sealed annotation/evaluation keys to annotators. Source mirror/data and externally published raw traces are local audit inputs; do not redistribute external traces without resolving their license. The source repo's lack of a visible license is not a permission grant. Generated figures are standard scientific artifacts, not interactive product UI.
