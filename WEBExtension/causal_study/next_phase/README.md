# Next research phase: offline mechanism and provenance audit

Start with [08_FINAL_SCIENTIFIC_DECISION.md](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/08_FINAL_SCIENTIFIC_DECISION.md). Decision: **REVISE DESIGN**, with no new paid inference authorized. The prior144-success ledger remains unchanged. This directory contains exploratory historical analysis and a prepared, mock-only two-arm diagnostic, not new experimental model results.

## Reports

1. [Deep transitions](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/01_E8_DEEP_TRANSITIONS.md): joint question transitions, subgroups, stronger abstention-aware baseline, raw IDs/lines and clustered intervals.
2. [Source/necessity audit](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/02_SOURCE_AND_NECESSITY_AUDIT.md): mechanical source checks, unresolved semantics and blinded annotation materials.
3. [Novelty comparison](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/03_NOVELTY_GAP_AND_RELATED_WORK.md):13 primary papers, nine-column matrix, dates/venues and bounded search evidence.
4. [Experiment blueprint](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/04_DECISIVE_EXPERIMENT_BLUEPRINT.md): compared alternatives and exactly one narrow conditional primary design.
5. [Budget/power](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/05_API_BUDGET_AND_POWER.md): measured reference usage,24-call conditional pilot, full proposal-screening cost and refined exact-power scenarios.
6. [Implementation audit](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/06_IMPLEMENTATION_AND_DRY_RUN_AUDIT.md): code, payload checks, tests, mock execution, resume and preserved artifacts.
7. [Adversarial review](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/07_ADVERSARIAL_REVIEW.md): three reviewer perspectives, concrete corrections and independence limits.
8. [Scientific decision](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/next_phase/08_FINAL_SCIENTIFIC_DECISION.md): required A–G answers and conditional WWW story.

## Reproduce local validation

Run from `/home/saurab/Documents/projectRLMAIN` using the existing Python environment with NumPy, SciPy, pytest and the locally preserved tokenizer. Nothing below downloads a model, loads credentials or calls Azure.

```bash
python3 -m WEBExtension.causal_study.next_phase.study prepare
python3 -m WEBExtension.causal_study.next_phase.study dry-run
python3 -m WEBExtension.causal_study.next_phase.offline_analysis
python3 -m WEBExtension.causal_study.next_phase.planning
python3 -m WEBExtension.causal_study.next_phase.power_refinement
python3 -m WEBExtension.causal_study.next_phase.audit_pipeline
python3 -m pytest WEBExtension/tests WEBExtension/causal_study/tests WEBExtension/causal_study/next_phase/tests -q
```

The planning command regenerates report 05's original coarse-grid section; run power refinement afterward to restore its exact-power supplement. Its825/1,100-question illustrative targets supersede the coarse doubling-grid1,600 values, without deleting the original planning artifacts. Reports03–04 and 07–08 contain scientific judgments that are not automatically revalidated by these commands.

`deep_transitions` regenerates report 01 and its artifacts; `source_audit` regenerates the data underlying the editorial report 02. Both use preserved raw inputs and are offline. Their scripts retain source hashes and official-score/source checks. Generated annotation templates contain pending independent labels; do not overwrite completed human annotation files on a future rerun. `make_figures` regenerates the observed readout comparison PNG/SVG from those saved paired results.

`study paid` deliberately raises `PermissionError`. No environment variable, approval flag or unused old quota can enable it. Changing the live execution path requires a new separately reviewed and explicitly authorized phase. The frozen execution sources (`binding_design.py`, `study.py`, `planning.py`) and all prepared artifacts are checked on resume. Never edit them casually or delete ledgers to make a rerun work; archive a new version before an authorized design change.

## Evidence and status files

- `transition_artifacts/`:12,898 observed records over 300 unique question identities; two not-run E3-v2 arm records for one ineligible ID; paired results and abstention-aware readouts.
- `source_artifacts/`:308 mechanical observations, bounded provisional agent reviews, independent annotation templates and protocol. Unknown semantic labels remain unknown.
- `literature/`:13-row matrix, verified identities/sections and search boundaries. Discovery-only leads are not verified findings.
- `prepared/`:12 fixed questions, frozen candidates/sources/configuration, private reference labels,24 full model packets and checksums. Private labels are never provider inputs.
- `mock/`:24 synthetic requests and responses, durable mock budget ledger,12 paired checkpoints, normal/resume validation and explicit `NOT_RUN_REAL_INFERENCE` analysis. These are not model-performance data.
- `planning_artifacts/`: Monte Carlo sensitivity grid, exact-power refinement, proposal-screening budgets and 6,809 gold-free fresh source-capable IDs. This is not an accepted semantic/conflict cohort.
- `PAYLOAD_AND_PRESERVATION_AUDIT.json`:947 historical hashes,900 original response texts,7,711 request source-sentence occurrences and 24 reviewed model packets.
- `COMPLETION_AUDIT.json` and `artifact_manifest.json`: final phase scope/status and hashes; neither authorizes inference. The manifest excludes transient caches, locks and itself. Regeneration changes hashes and requires an explicit refreshed snapshot for that revision.

Local validation:171 tests and 12 subtests passed. Remaining scientific risks are explicit in reports 06–08. No training, push, paid judge or additional E6 completion was run.
