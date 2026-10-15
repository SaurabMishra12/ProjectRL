# Matched aggregation reassessment implementation plan

> Implement inline using the executing-plans workflow. The user's detailed offline brief authorizes execution through the scientific decision; no additional approval pause is needed.

**Goal:** independently reproduce the baseline counts, apply identical readouts to all20 saved E5 arms, and reassess communication quality/cost without inference.

**Architecture:** a new independent analysis module reads raw E5 JSONL and provider artifacts directly. It does not import the old vote, E8 reconstruction or any provider. Separate accounting, reporting and figure modules consume its question-level results. All writes remain in this directory.

**Tech stack:** Python standard library, NumPy/SciPy, pytest and Matplotlib. Spec: `/home/saurab/.codex/attachments/ca8a9d4d-430c-4deb-be43-b055199ffe6f/Pasted text.txt`.

## Constraints and review focus

- Zero paid calls, judges, training, historical writes or checkpoint changes. Preserve both previous artifact manifests and every listed artifact.
- Aggregators accept only ordered response dictionaries, never reference labels or scores. Preserve literal A, yes and no as substantive answers.
- Missing/duplicate questions, arms, provider outputs or unmarked empty answers fail validation. Do not fill missing predictions.
- Three learned and MatchedRandom seeds remain separate. Repeated seeds/readouts share questions and provider calls.
- Count unique saved requests and attempts, including retries; logical condition costs and message tokens are separate. Shared calls cannot be assigned exclusively to arms.
- Candidate ceilings apply to these selection readouts only. Annotated support absence is not semantic insufficiency.

## Tasks

- [x] Independently count original NoComm10/Broadcast34, abstention-aware NoComm62/Broadcast65 and any-correct67/66 from raw E5; snapshot historical artifacts before analysis.
- [x] Write failing tests for normalization, refusal markers, plurality versus strict majority, all-abstaining behavior, order-sensitive/independent ties and label-free interfaces. Implement `normalize`, `score_answer`, `is_abstention`, `aggregate(responses, rule)` in `analysis.py`.
- [x] Implement strict raw loading with source hashes and official-evaluator cross-checks. Materialize20 arms×300 questions×3 readouts, without using previous reconstructed outcomes as inputs. Test complete pairing and changed readout with frozen proposals.
- [x] Implement paired EM exact McNemar wins/losses/ties, question-bootstrap F1/EM intervals, readout-by-communication contrasts, declared multiplicity families, all pre-treatment subgroups and question-level failure tags. Test paired units and interaction arithmetic.
- [x] Implement `accounting.py`: verify unique cached responses, node order and raw provider text; join ledger/dispatch/receipt attempts by key; preserve lower-bound usage and uncertainty; save per-arm referenced physical demand and global union without double counting.
- [x] Run all-arm analysis. Generate metrics, exact transition memberships, exhaustive failure cases and cost frontiers; inspect the exact matched comparisons before interpreting them.
- [x] Verify required primary literature, including Free-MAD and Diagnostic Study; save a bounded comparison matrix and bibliography. Assess source-association feasibility without executing its prepared pilot.
- [x] Produce the seven requested reports, figures, reproducibility README, tests, source hashes and final preservation/completion audit. Decide GO/REVISE/STOP after matched results; no mixed-readout harm claim.

Verification commands: `python3 -m pytest WEBExtension/causal_study/readout_reassessment/tests -q`; `python3 -m WEBExtension.causal_study.readout_reassessment.analysis`; `python3 -m WEBExtension.causal_study.readout_reassessment.accounting`; `python3 -m WEBExtension.causal_study.readout_reassessment.reporting`. No command calls a model.

Final local verification:187 tests and12 subtests passed, including16 new tests. The initial combined test collection had a duplicate basename; renaming only the new test fixed it. All20 E5 arms are reaggregated under three fixed rules. Independent source/provenance verification and the final artifact snapshot are performed by verify.py. Decision:REVISE/pivot findings;0 paid calls, source-association pilot abandoned as the next primary WWW study.
