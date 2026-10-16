# SDD ledger — plan: docs/superpowers/plans/2026-10-16-web-reasoning.md
Execution: Native, approved by Continue on 2026-10-16. Branch codex/web-reasoning.
Baseline: WEBExtension/tests/test_protocol.py — 6 passed.
Ruling: Managed worktree is read-only under sandbox; use worktree-skill fallback on a new local feature branch — writable isolation preserves existing edits — cost if wrong: unrelated checkout edits require careful staging.
Pre-flight 1→2→3: frozen public records consumed by adapters and prompt allowlists; context/strata stored as immutable tuple pairs.
Pre-flight 3→4→5: parsed source-owned Fact IDs feed ledger/metadata; packets carry complete source text and original locators.
Pre-flight 5→6→7: schedules emit unique recipients; physical request identity excludes logical arm labels.
Pre-flight 6→7→8: successful malformed/refused outputs keep usage and raw hashes; unresolved cells remain incomplete.
Pre-flight 7→9→10: immutable initial states fork by arm; three-phase limits distinct from observed delivery counts.
Pre-flight 2→8→12: alias/private program targets never reach runtime; official parity required for official scores.
Pre-flight 5→11: sender metadata is bounded before feature extraction; learned state must remain O(N).
Pre-flight 6→13: live transport lazy and disabled unless a separate exact reviewed run permit exists.
Task 1: complete (commits 4aacf58..8dc9e0b, tests: python -m pytest WEBExtension/web_reasoning/tests/test_contracts.py -q → 3 passed in 0.03s)
Task 2: complete (commits 8dc9e0b..f2b08bc, tests: python -m pytest WEBExtension/web_reasoning/tests/test_data.py WEBExtension/web_reasoning/tests/test_splits.py -q → 11 passed in 0.04s)
Task 3: complete (commits f2b08bc..419c721, tests: python -m pytest WEBExtension/web_reasoning/tests/test_prompts.py WEBExtension/web_reasoning/tests/test_parsing.py WEBExtension/web_reasoning/tests/test_tokenization.py -q → 12 passed in 0.04s)
Task 4: complete (commits 419c721..bcb610b, tests: python -m pytest WEBExtension/web_reasoning/tests/test_packets.py WEBExtension/web_reasoning/tests/test_evidence.py WEBExtension/web_reasoning/tests/test_parsing.py -q → 15 passed in 0.04s)
Task 5: complete (commits bcb610b..7c47fd9, tests: python -m pytest WEBExtension/web_reasoning/tests/test_policies.py -q → 14 passed in 0.07s)
Task 6: complete (commits 7c47fd9..ba27a58, tests: python -m pytest WEBExtension/web_reasoning/tests/test_storage.py WEBExtension/web_reasoning/tests/test_backend.py WEBExtension/web_reasoning/tests/test_accounting.py -q → 10 passed in 0.07s)
Ruling: Render delivered source text once in visible_sources and retain packet claims/IDs separately — prevents duplicate context without hidden source access — cost if wrong: prompt and transport serializers could diverge; integration tests pin source identity.
Task 7: complete (commits ba27a58..4028547, tests: python -m pytest WEBExtension/web_reasoning/tests/test_runner.py WEBExtension/web_reasoning/tests/test_policies.py WEBExtension/web_reasoning/tests/test_backend.py -q → 27 passed in 0.40s)
Ruling: Keep upstream evaluator files byte-exact; use a documented runtime import substitution from ujson to stdlib json — numerical metric functions remain upstream and hash-verified — cost if wrong: unusual JSON syntax could deserialize differently; source inputs must be standard JSON.
Task 8: complete (commits 4028547..0b1de11, tests: python -m pytest WEBExtension/web_reasoning/tests/test_readout.py WEBExtension/web_reasoning/tests/test_statistics.py WEBExtension/web_reasoning/tests/test_evaluators.py -q → 5 passed in 0.14s)
Ruling: Always-run diagnostic uses a fixed original-question verification request if the sink emits no need — this permits the planned fixed consultation depth without a hidden planner — cost if wrong: its stopping contrast includes this declared request policy, not only halting.
Task 9: complete (commits 0b1de11..087252c, tests: python -m pytest WEBExtension/web_reasoning/tests/test_adaptive.py WEBExtension/web_reasoning/tests/test_runner.py -q → 12 passed in 0.79s)
Ruling: Legacy prompt ablation is explicitly legacy-style with a strict citation-to-source adapter, not a replay of old role-specific runs — same actual task template goes to every topology — cost if wrong: it cannot identify an exact historical prompt effect.
Task 10: complete (commits 087252c..0b90376, tests: python -m pytest WEBExtension/web_reasoning/tests/test_scaling.py WEBExtension/web_reasoning/tests/test_ablations.py -q → 5 passed in 0.23s)
Task 11: complete (commits 0b90376..b69d70e, tests: python -m pytest WEBExtension/web_reasoning/tests/test_learned.py WEBExtension/web_reasoning/tests/test_training.py -q → 4 passed in 2.05s)
Task 12: complete (commits b69d70e..6ab28ee, tests: python -m pytest WEBExtension/web_reasoning/tests/test_finqa.py WEBExtension/web_reasoning/tests/test_data.py WEBExtension/web_reasoning/tests/test_evaluators.py -q → 15 passed in 0.39s)
Ruling: CLI preparation, live transport, serialization, and workflow use separate helper modules — keeps credential access lazy and targets outside execution — cost if wrong: extra interfaces can drift; CLI/resume tests exercise them together.
Task 13: complete (commits 6ab28ee..7d00232, tests: python -m pytest WEBExtension/web_reasoning/tests -q → 98 passed in 4.02s)
Final review: fresh gpt-6-astra reviewer; five Important findings accepted: FinQA hidden-source/program coherence; report cohort isolation; missing assigned cells; fixed private/dense schedules; W4 protocol dispatch. No Critical or Minor findings. All enter one TDD fix pass.
Final: Ruling: Publisher-wide adapters and grouped split feasibility were set aside by review — verify on publisher inputs in the separately authorized pilot — cost if wrong: adapter correction or sampling amendment may be needed.
Final: Ruling: Real-tokenizer packet fit was set aside by review — verify with the pinned serving tokenizer in the pilot — cost if wrong: many complete FinQA atoms may be omitted.
Final: Ruling: Live provider compatibility, authenticated revision and usage were set aside by review — verify only in an authorized run — cost if wrong: deployment or usage may differ from assumptions.
Final: Ruling: Empirical quality, scaling and learned generalization were set aside by review — retain as untested hypotheses — cost if wrong: real experiments may show no benefit.
Final: Ruling: Unrelated external-validation and research changes were set aside by review — exclude their files from this branch work — cost if wrong: no review assurance applies to those independent changes.
Final: fixed FinQA hidden source execution and program/status coherence — test_hidden_table_or_abstaining_program_cannot_score and test_scoring_requires_endpoint_visibility_and_supported_status RED→GREEN, suite 125/125.
Final: fixed cohort pooling and mismatched scaling population — test_reports_isolate_staged_cohorts_and_nested_scaling RED→GREEN, suite 125/125.
Final: fixed omission of assigned missing cells — test_missing_assigned_cell_keeps_primary_effect_incomplete RED→GREEN, suite 125/125; interrupted two-question CLI case also covered.
Final: fixed outcome-dependent fixed controls — test_fixed_controls_ignore_initial_supported_answer RED→GREEN for N=2..5 and phases=1..3, suite 125/125.
Final: fixed W4 stopping protocol dispatch and exact initial-state pairing — test_w4_stopping_variants_use_adaptive_protocol RED→GREEN, suite 125/125.
Final: deferred minors: none. No Critical findings. No second review dispatched; fixes verified by regression tests and the full package plus baseline suite.
