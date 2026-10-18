# Research discovery loop

**Final decision: C — NO JUSTIFIABLE WWW MAIN-TRACK RESEARCH QUESTION FOUND.**

Start with 08 for the decision,01 for the evidence ledger,02 for primary literature,03/10 for nine rejected candidates and simulated criticism, and 09 for full-study feasibility. There are zero final survivors. The three files in 05 are rejected design records.06 preserves the two-round history;07 records scores with written reasons.

All work is offline saved-output analysis, public primary-source reading and research design. No paid inference, model judge, PPO training or historical modification is authorized or performed. The arXiv literature-search, codebase-memory and academic-writing workflows informed this investigation. Used primary-paper URLs and publication editions are listed in 02 and literature/references.json.

## Reproduce

From /home/saurab/Documents/projectRLMAIN, with Python, NumPy and SciPy installed:

```sh
python research_discovery_loop/analyses/offline_audit.py
python research_discovery_loop/analyses/history_checks.py
python research_discovery_loop/analyses/power_sensitivity.py
python research_discovery_loop/analyses/write_reports.py
python -m unittest discover -s research_discovery_loop/tests -v
python research_discovery_loop/analyses/verify.py
```

Do not regenerate the immutable historical baseline as part of reproduction. inventory.py was used once to capture initial hashes; replacing that baseline would weaken the preservation check. The analysis scripts read saved data and write only inside this directory. They do not import model runners or SDKs. Literature judgments are explicit curated inputs in research_assessment.py; regenerating reports does not rerun a literature search or establish automatic novelty.

Results include question-level scores, action/recipient pairs, E6 contrasts, machine-readable claim/candidate ledgers, input hashes, historical baseline, power sensitivity and completion verification. All scientific effects are exploratory, not confirmatory. New-bootstrap rounding can differ from preserved older reports. Effects in JSON are fractions unless named otherwise.

## Material limits

The inventory is not a claim every file body was read. Canonical task-relevant code and major raw research artifacts were checked; binary checkpoints, duplicate exports and unreferenced cache bodies were not exhaustively interpreted. Some legacy benchmark/checkpoint provenance remains unresolved. Source sufficiency/entailment/independence labels are incomplete. Reviewer personas are one investigator’s simulations, not independent peer review. No paper outcome was independently reproduced through new model execution.
