# Fresh review and implementation response

Date: 2026-10-16. Reviewer: a fresh `gpt-6-astra` agent, read-only. Reviewed range: `4aacf58647cd41fd06c01cf5974edb711781e9b5..7d002322f`. The original verdict was **not ready to merge**, with five Important findings, no Critical findings, and no Minor findings. Review verification passed 98 new-package tests and six existing protocol tests. The findings were accepted and corrected in one test-driven fix pass; no second reviewer approval is claimed.

The reviewer credited the public/private separation, immutable W1 forks and matched logical calls, evidence locators, distinct failure outcomes, durable physical accounting, and exclusion of mock scientific scores. The six original implementation rulings were judged understandable and adequately qualified.

## Accepted findings and regressions

1. **Hidden FinQA evidence and null-answer programs could receive correctness.** The parser now requires supported programs to map every nonconstant operand to actual visible, cited evidence, and rejects programs attached to unresolved answers. Predicted execution receives only the endpoint's visible table rows. The full registry is used for trusted reference checks. Direct scoring without endpoint visibility remains incomplete. Regressions: `test_hidden_table_or_abstaining_program_cannot_score`, `test_scoring_requires_endpoint_visibility_and_supported_status`, and positive mapped-table parity. The first two failed before the fix.
2. **Reports pooled pilot, development, and held-out results.** The CLI now requires an explicit cohort, saves source-cohort identity, and filters the expected population before loading results. Scaling applies the same frozen nested keys at every N, including reused N=3. Regression: `test_reports_isolate_staged_cohorts_and_nested_scaling`, failed before the fix.
3. **Missing assigned cells vanished from primary analyses.** Reporting left-joins sealed results onto the frozen expected question/team-size population. It includes missing cells, coverage, unresolved bounds, and no complete primary estimate until required outcomes resolve. Raw partial-question attempts remain in durable logs. Regressions: `test_missing_assigned_cell_keeps_primary_effect_incomplete` (failed before fix), empty-run report, and a two-question run interrupted by explicit provider rejection after its first sealed question.
4. **Initially supported answers halted fixed private and dense controls.** Private updates now execute the frozen phase prefix; dense is explicitly fixed-depth. With all three slots, private uses `k(N)+2` updates and dense `3N`. Regression: `test_fixed_controls_ignore_initial_supported_answer`, failed before fix at every tested N and phase count.
5. **W4 stopping variants dispatched fixed W1 execution.** Configuration now separates the experiment label and execution protocol. Both stopping conditions explicitly use adaptive gather/request/reply machinery; forecasts follow the same protocol. Each adaptive W4 run also forks both stopping arms from the exact same initial states, avoiding dependence on identical provider outputs across independent manifests. Regression: `test_w4_stopping_variants_use_adaptive_protocol`, failed before fix. With a supported initial shared-evidence fixture, observable stopping uses zero selected updates and always-run uses three at N=3. The additional counterfactual is charged in the forecast and reports.

## Behaviors the reviewer declined to judge and executor rulings

1. Publisher-wide adapter coverage and grouped split feasibility remain for the real-data pilot because publisher files were unavailable. Risk: a release may require an adapter correction or a recorded sampling amendment.
2. Real-tokenizer packet fit remains for the pinned-tokenizer pilot. Risk: complete FinQA rows and notes may exceed the packet cap often.
3. Live provider compatibility, authenticated revision, and usage remain for an explicitly authorized run. Risk: deployment or usage reporting may differ from the adapter assumptions.
4. Routing quality, cost scaling, and learned-policy generalization remain empirical hypotheses. Risk: real outcomes may show no improvement.
5. Unrelated external-validation changes and research directories remain outside this review. Risk: this review establishes no correctness for those independent changes.

No minor findings were deferred. None of these local checks establishes benchmark accuracy, empirical linear scaling, or semantic citation faithfulness. No paid calls were made.
