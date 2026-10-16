# SparseComm for distributed Web evidence: design for review

Date: 2026-10-16. Status: **design approved for implementation planning by the user's “Continue”; implementation plan awaiting review. Runtime changes and new experiments have not been implemented or executed.**

User-selected priority: **2WikiMultihopQA primary; FinQA secondary.** The requested outcome is a credible Web evaluation and concrete improvements to SparseComm, with compact prompts that make useful evidence easier to transmit. Performance improvements remain hypotheses.

## Recommended direction

1. Preserve partial facts even when a local agent cannot answer the whole question.
2. Transmit complete source atoms with stable citations; replace blind token-prefix cropping for the new method.
3. Route complementary evidence to a designated synthesis agent. Compare routing under identical recipient calls before adding adaptive rounds.
4. Use a controller that operates on bounded observable metadata and a set of agents, with explicit transmission limits. Low-rank edge parameters alone do not guarantee linear cost.
5. Separate supported answers, partial evidence, abstention, provider refusal, invalid output, and unresolved requests.
6. Give static, random, dense, and private-revision controls the same improved task instructions, output schema, model, and scoring rules.

## Review artifacts

- [Experiment and architecture specification](EXPERIMENT_DESIGN.md): hypotheses, algorithms, controls, sample design, accounting, budgets, implementation boundaries, and acceptance checks.
- [Prompt pack](PROMPT_PACK.md): proposed shared instructions, evidence extraction, synthesis, follow-up requests, and FinQA specialization.
- [Evidence audit](EVIDENCE_AUDIT.md): current local findings, corrections to the suggested narrative, official benchmark sources, and graph limitations.
- [Validation record](VALIDATION.md) and [source manifest](source_manifest.json): actual local checks and source hashes, distinct from proposed implementation tests.
- [Implementation plan](../../docs/superpowers/plans/2026-10-16-web-reasoning.md): 13 tasks, interfaces, required tests, and execution handoff.

## Alternatives considered

**A — prompts and readout only.** Cheapest engineering change, useful as a development ablation. It does not address routing, repeated evidence, or the quadratic controller state.

**B — compact evidence, demand-aware sparse routing, common readout, and hard budgets. Recommended.** First isolate a one-update routing effect; then evaluate adaptive consultation and a learned, agent-count-independent scorer. Keep a deterministic sparse router as a separate reproducible baseline.

**C — jointly learn retrieval and communication.** Potentially useful later, but it changes evidence availability and adds retrieval/training costs. It cannot resolve the current routing question cleanly. Do not include it in the first implementation.

## Important changes to the supplied strategy

The current evidence does not establish a positive new Web result or a quadratic-to-linear empirical scaling law. The matched-readout HotpotQA comparison is 62 versus 65 correct, not 62 versus 34. The legacy scaling summary contains N=2,4,5; the plotting script inserts N=3 from another benchmark and estimates its total tokens. The new design therefore tests these claims prospectively.

No old paper, result, checkpoint, existing experiment gate, or provider configuration is changed by this proposal. No new API requests or training runs were made. The next step is review of the implementation plan and selection of its execution method.
