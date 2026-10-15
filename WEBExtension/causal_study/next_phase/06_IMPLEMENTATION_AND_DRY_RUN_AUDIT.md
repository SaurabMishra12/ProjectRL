# Implementation and dry-run audit

**Local validation passes; scientific decision: REVISE DESIGN. No paid inference ran.** The implementation prepares the two-arm source-association diagnostic specified in report 04. It cannot authorize or execute a live study. Passing software checks does not resolve semantic validity, novelty, power, or independent review.

## Implemented components

- `binding_design.py`: gold-free eligibility, frozen three-candidate sets, verbatim source registry, two exchanged citation targets, preserved candidate/source degrees, deterministic hash selection, crossed candidate-order and execution-order balancing, checksums, payload allowlists, and exact prompt-token equality.
- `study.py`: original initial-response/cache provenance, frozen manifest/configuration/private labels, source snapshots, strictly mock-only execution, durable attempt/completion accounting through the existing tested backend, atomic question checkpoints, strict candidate-ID parsing, and safe resume. `execute_paid()` always raises before provider access. No flag enables it.
- `offline_analysis.py`: exact question-set and paired-cell checks, checkpoint hashes, candidate-ID-to-answer mapping, arm-specific prompt hashes, official answer EM/F1, paired statistical analysis, and explicit rejection of mock records as scientific outcomes. Synthetic unit-test scoring is labeled as such. Default analysis reports `NOT_RUN_REAL_INFERENCE` with null effects and p-value.
- `planning.py`: actual prompt measurements, measured E6 usage reference, fresh-pool exclusion, question-level paired McNemar simulations and complete screening-cost accounting. `power_refinement.py` integrates exact rejection probabilities to refine the coarse grid without editing the frozen execution sources.
- `deep_transitions.py` and `source_audit.py`: raw historical reconstruction, question-level bootstrap summaries, fixed abstention-aware readout, mechanical evidence audit, blinded annotation material, generated report 01 and data supporting the editorial report 02. The literature matrix and scientific judgments are editorial primary-source reviews, not automated novelty certificates.
- `audit_pipeline.py`: independent local checks of all prepared packets, initial cache/trace identity, official source text and historical artifact preservation. All writes stay in this new study directory.

The prepared configuration retains the original provider/model alias and temperature, and proposes a finalizer-only output cap of 2,048. It sets24 mock completions and 30 mock attempts. The mock dispatch interval is 0 because it never contacts a provider; a future live protocol would need its own authorized configuration and dispatch floor. The present code is intentionally a pilot preparation and mock harness. Its75-question construction ceiling and inherited provider caps are not a runnable implementation of a thousand-question confirmation. Scaling requires a separately reviewed configuration and gate, rather than silently loosening caps.

## Exact prepared cohort and payload review

The frozen manifest contains 12 hash-selected cases from 20 mechanically eligible E5 partitioned states. Thirty-six existing initial responses are reused. All six candidate permutations occur twice; each permutation occurs once with each arm order. Recorded/exchanged and exchanged/recorded each occur six times. There is one realization per arm per question, not12 questions multiplied by six independent repetitions.

The full24 model-visible message pairs are saved in [prepared/model_payloads.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/prepared/model_payloads.json). Their content lengths range 275–573 pinned-tokenizer tokens, mean375.833, with exact equality within every pair. Server framing is excluded from these content counts; E6 measured74 additional prompt tokens per request, used only as a projection.

For every pair, the question, system instruction, candidate answers, candidate multiplicities/order, source titles, sentence IDs, text, source ordering, and number of candidate/source citations are identical. Exactly two candidate `source_ids` values differ. All IDs resolve in the same registry. The only user-payload keys are `question`, `candidates`, and `sources`; the candidate/source subfields are separately allowlisted. Arm names, gold references, correctness labels, confidence, provider keys and internal hashes are absent. Ordinary source text may contain the answer; that is an intended source property, not secret reference metadata. Identical opaque IDs do not establish semantic support or model reliance.

All 900 cached initial responses were checked against their saved provider text. All 7,711 initial request evidence-sentence occurrences were matched, by title and index, to the preserved official source text. The selected candidate answers match the original trace byte-for-byte. These checks strengthen mechanical provenance; they do not adjudicate entailment or necessity. Evidence selected by the original models is outcome-independent with respect to the future two-arm contrast, but remains a model-selected source bank.

## Mock execution, scoring and resume

The dry run completed12 paired question blocks using24 mock HTTP attempts and 24 mock completions. A second execution reused the checkpoints and added **0** requests. Mock output always selects positional C0 and is labeled `MOCK_SELECTOR_ONLY`; its synthetic usage is never substituted for actual future token usage or accuracy.

The saved validation says `scored_scientific_effects=false`, `real_provider_calls=0`, and `paid_authorized=false`. Default offline analysis validates all 24 cells, preserves the paired question set and reports no empirical selector effect. Its callable scorer is tested with explicitly synthetic fixtures. A future real-output analysis additionally needs authenticated raw provider artifacts and a separately authorized execution process; an `execution_mode` field alone is not proof of provenance or authorization.

The reused budget backend has durable request identity, response caches, global attempt/completion ceilings, bounded explicit429 retries and an unknown-execution stop. Unit tests in the combined suite exercise these failure paths. The normal new-study dry run had no injected failures. A successful HTTP response with invalid JSON, a new answer instead of an allowed ID, extra fields, or truncation spends quota and stops. Missing cells are neither zero-valued predictions nor automatically discarded observations. Checkpoint hashes detect accidental changes; they are integrity checks, not signatures against malicious rewriting.

## Verification evidence

From the repository root:

```bash
python3 -m pytest WEBExtension/tests WEBExtension/causal_study/tests WEBExtension/causal_study/next_phase/tests -q
```

Result: **171 passed,12 subtests passed**,52.94 seconds. The new-phase portion comprises57 tests; historical suites remain included. Tests cover treatment invariants, sender-invisible citations, source/candidate mutation, gold-field exclusion, exact token equality, order balance, duplicate question IDs, normalization edge cases, parsing/truncation, paid prohibition, resume, budget failures, source auditing, reconstruction, paired scoring and power calculations. Exact power is also checked against the all-gains binomial case and null test size.

The initial red checks for the two new analysis modules failed because the modules did not exist; their implemented suites now pass. Logs and machine-readable evidence are [full_tests.log](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/full_tests.log), [mock/validation.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/mock/validation.json), [mock/offline_validation.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/mock/offline_validation.json), and [PAYLOAD_AND_PRESERVATION_AUDIT.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/PAYLOAD_AND_PRESERVATION_AUDIT.json).

The previous artifact manifest's947 files are byte-identical. The original paid ledger retains144 successes/144 attempts and SHA256 `ad7ebd4e56e10e4ec9bc7672f3315a4a48c6f2973e36bdbb9c65f9e2e7401d52`. Its six unused completion slots were not spent, reset or reassigned. No Azure request, model-list probe, paid judge, PPO training, or deployment switch occurred.

## Remaining validity and replication limits

Mechanical checks all pass for the prepared artifacts. Independent human necessity/entailment annotation remains absent. The cached cohort is already inspected, substantive string differences can be synonyms, original citations need not be supporting, and the swapped mapping can improve support. Consequently the runnable mock identifies only an association manipulation. The actual new finalizer's accuracy, reasoning length, invalid-output rate and latency are unmeasured. The deployment is an alias, not a verified immutable model snapshot. No claim of guaranteed determinism follows from temperature0.

Graph Verify preflight was repeated after continuation: project `projectRLMAIN`, generation `2026-10-13T17:28:45Z`, fast/ready. All new source/test paths are untracked by that generation; relevant source was read and checked directly. No graph absence or completeness claim is used. Postimplementation reviewer perspectives and the independence limitation are stated in report 07. These are reasons to retain REVISE DESIGN even though local software validation succeeds.
