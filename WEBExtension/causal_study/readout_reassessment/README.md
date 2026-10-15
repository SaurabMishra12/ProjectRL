# Matched readout reassessment

Start with the [scientific decision](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/07_FINAL_SCIENTIFIC_DECISION.md). **REVISE/pivot toward independently validated findings work;0 new paid calls.** Matched abstention-aware Broadcast gives65/300 correct versus62/300 NoComm; the earlier62-versus34 cross-readout comparison cannot establish harm. Some frozen policies retain favorable descriptive quality–cost trade-offs, with important seed and resource limitations.

## Seven requested reports

1. [Independent baseline verification](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/01_INDEPENDENT_BASELINE_VERIFICATION.md).
2. [All matched aggregation results and physical usage](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/02_MATCHED_AGGREGATION_RESULTS.md).
3. [Paired communication effect reassessment](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/03_COMMUNICATION_EFFECT_REASSESSMENT.md).
4. [Question-level failure mechanisms](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/04_QUESTION_LEVEL_FAILURE_MECHANISMS.md).
5. [Primary literature and WWW significance](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/05_WWW_NOVELTY_AND_SIGNIFICANCE.md).
6. [Feasibility and minimum additional evidence](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/06_FEASIBILITY_AND_MINIMUM_EXPERIMENT.md).
7. [Final decision and six explicit answers](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/readout_reassessment/07_FINAL_SCIENTIFIC_DECISION.md).

## Reproduce offline

Run from `/home/saurab/Documents/projectRLMAIN` with the existing Python environment. Required libraries: NumPy, SciPy, pytest and Matplotlib. No script initializes a provider, reads credentials, downloads a model, trains a policy, or writes to historical directories.

```bash
python3 -m WEBExtension.causal_study.readout_reassessment.analysis
python3 -m WEBExtension.causal_study.readout_reassessment.accounting
python3 -m WEBExtension.causal_study.readout_reassessment.diagnostics
python3 -m WEBExtension.causal_study.readout_reassessment.reporting
python3 -m pytest WEBExtension/tests WEBExtension/causal_study/tests WEBExtension/causal_study/next_phase/tests WEBExtension/causal_study/readout_reassessment/tests -q > WEBExtension/causal_study/readout_reassessment/full_tests.log
python3 -m WEBExtension.causal_study.readout_reassessment.verify
```

`verify` independently rechecks the initial counts using a separate standard-library voting implementation, recomputes the fresh pool, checks historical artifacts and physical-source hashes, validates report links, then refreshes this phase's completion audit and artifact manifest. Its reference to187 passing tests is this frozen revision's expected result; deliberate test additions need an explicit updated verification expectation. Reports05–07 and literature records are editorial scientific judgments, not automatically revalidated by rerunning the code. The last command does not authorize a later model run.

The combined suite passes **187 tests and12 subtests**, including16 new tests. An initial combined collection failed because the new test shared the old module basename `test_analysis`; renaming only the new file fixed it. No old test/cache needed deletion. A source-independent vote implementation and official evaluator cross-check provide evidence beyond those unit tests.

## Data and provenance

- `results/question_readouts.jsonl`:18,000 records across20 arms, three readouts and300 paired question identities. Includes exact original/final strings, refusal markers, coverage, readout decisions, source flags and raw-line links. Do not treat rows as18,000 independent questions.
- `results/matched_results_with_costs.csv`: complete60-row flat outcome/resource table; the same arm's physical references recur across readouts and must not be summed.
- `results/communication_effects.json`, `readout_effects.json`, `interaction_effects.json`:108 matched communication comparisons,40 readout comparisons and18 readout-by-communication contrasts, with paired intervals/exact EM tests where appropriate.
- `results/communication_transitions.jsonl`, `joint_transitions.json`, `failure_case_index.json`, `readout_answer_changes.jsonl`: exhaustive paired cells and changed-question memberships, including changes in correct-node coverage and candidate multiplicity.
- `results/physical_accounting.json`, `attempt_inventory.json`, `referenced_request_keys.json`: deduplicated ledger/dispatch/receipt/cache evidence.7,648 attempts reference7,634 successful requests;37 output usages and5 uncertain executions prevent an exact all-in billing total.
- `results/source_strata.jsonl`, `evidence_abstention_strata.json`: all600 regime/question source unions and full predefined strata. Annotated overlap is not independent semantic sufficiency.
- `results/original_code_comparison.json`, `verification.json`, `response_and_reference_audit.json`, `fresh_pool_verification.json`: frozen/current function comparison, official scoring checks, reference edge cases and unexposed source-capable IDs.
- `literature/`:10 primary-source identities and intervention comparisons, plus bounded search provenance. Original linked papers are not claimed reproduced.
- `figures/`: observed matched comparison PNG/SVG and its exact plotted data. No synthetic model outcome is plotted.
- `PRESERVATION_BASELINE.json`: protected prior artifacts/manifests and controller checkpoints. `COMPLETION_AUDIT.json` and `artifact_manifest.json` document the final local revision; caches, locks and the manifest itself are excluded from its hash list.

The initial snapshot includes every file in both earlier study manifests and the manifests themselves; controller checkpoint hashes were added before any checkpoint operation. The original E5 raw hash and all referenced physical sources are independently checked. Previous E6 remains at144 successes; its six unused slots are not authorization. No paid inference entry point exists in this reassessment.
