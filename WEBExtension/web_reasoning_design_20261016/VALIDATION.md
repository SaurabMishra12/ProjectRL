# Validation of the design artifacts

Date: 2026-10-16. Scope: documentation, inspected source evidence, and budget arithmetic. **No implementation or Web benchmark performance has been validated.**

## Checks performed

- Independently recomputed the historical E5 NoComm/Broadcast counts with `WEBExtension.causal_study.readout_reassessment.verify.independent_counts()`. The function checked its output against the saved independent artifact and returned partitioned original counts 10/34, partitioned abstention-aware counts 62/65, and shared abstention-aware counts 73/73.
- Parsed the legacy scaling summary and plotting source without executing the plotting script. Confirmed summary keys N=2,4,5 with 150 samples each, and the plot's supplied N=3 total-token value 1,680. The source marks that value as estimated.
- Checked local Markdown targets and balanced fenced code blocks in the proposal documents.
- Verified the proposed W1 call formula `2N+6`, giving 10,12,14,16 logical completions at N=2,3,4,5. The 20-question N=3 pilot is 240 logical completions. The full staged W1 package described in the specification is 5,360 after its nested N=3 evaluation reuse, before training, adaptive runs, ablations, retries, and any additional learned arm.
- Enumerated allowed gather/request counts at N=2,3,4,5 to check the design inequality `gather + 2*requested <= 3(N-1)`. This checks the mathematical schedule ceiling; runtime enforcement still needs implementation and tests. The separate hard control budget is six 96-token frames per agent.
- Hashed 16 existing local source/result artifacts in [source_manifest.json](source_manifest.json), including the exact historical E5 endpoint file and reference file used by the count reconstruction.
- Inspected graph coverage for evidence paths. The graph generation was 2026-10-13T17:28:45Z; Web paths were not tracked, so direct source/artifact reads were required.

## Not performed

No new provider inference, policy training, benchmark download, live model identity probe, deployed prompt/tokenizer trial, semantic source adjudication, or held-out evaluation. No performance improvement, statistical power, semantic sufficiency, implementation correctness, or conference novelty is established by these checks. The 192-token packet setting and 2,048-token output envelope remain proposed pilot settings.

The current deliverable consists of a reviewable [specification](EXPERIMENT_DESIGN.md), [prompt pack](PROMPT_PACK.md), [evidence audit](EVIDENCE_AUDIT.md), and source manifest. Existing runtime code, historical results, checkpoints, and manuscript are preserved.
