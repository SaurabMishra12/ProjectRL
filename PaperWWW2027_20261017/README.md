# Who Gets the Last Word?

Working paper prepared on 17 October 2026 from the completed local evidence and relevant prior chats. This is an exploratory empirical audit, not a certified submission-ready paper or an acceptance prediction.

## Narrative

**A communication graph's value depends on how its updated answers reach the final output.** In the completed 2Wiki extension, a fixed sink makes In-Star and Dense identical by request reuse, while voting over all agents makes Dense more accurate. In historical HotpotQA traces, changing abstention handling strengthens the no-communication baseline and reduces the apparent communication gain. These are different, complementary effects; neither establishes that communication is generally unnecessary.

The proposed title is **Who Gets the Last Word? Auditing Communication Gains in Multi-Agent Web Question Answering**. Two alternatives are **The Last Word Changes the Verdict: Evaluating Communication in Web QA** and **Sparse Graphs, Different Answers: The Role of Readout in Multi-Agent Web QA**. The first title is recommended because it names the measured issue without promising adaptive routing superiority.

## Paper structure and paragraph roles

1. **Introduction:** problem (uneven evidence), challenge (graph and final answer rule interact), study, observations, bounded contribution and Web relevance.
2. **Related work:** debate and voting, graph optimization, evidence integration and abstention; precise distinction from the closest proposal/readout study.
3. **Evaluation framework:** define readouts and matched contrasts; explain the one-round fixed-sink identity without presenting it as a novel theorem.
4. **2Wiki study:** cohort provenance, seven arms, shared prompts, both readouts, measured outcomes, compute accounting and failures.
5. **HotpotQA reassessment:** matched readout results, conservative alternative and resource-control limits.
6. **Implications and limitations:** what to control before claiming communication efficiency, and what the results leave unresolved.
7. **Conclusion:** the specific observed dependence of communication comparisons on final answer selection.

## Scope and integrity

- The 2Wiki study is complete: 150 paired questions, with 20 previously inspected pilot questions reported separately from 130 additional questions. All seven arms and both readouts are retained.
- The original pilot gate failed. The extension was explicitly authorized after that failure. The narrative and highlighted comparisons are exploratory.
- The 300 HotpotQA questions were repeatedly inspected in earlier work. Offline rescoring is not a new independent sample.
- The new routing policies are deterministic. They do not validate learned SparseComm weights.
- In-Star/Dense equality at the fixed sink is structural request identity, not an independent equivalence or noninferiority result.
- Full-Union is the strongest point estimate in the additional cohort and uses fewer logical model-token references; its EM advantage over Dense voting is uncertain.
- Neither source completeness nor correct answers establish semantic source necessity or faithful evidence use.
- Existing manuscripts, experiments and original records are preserved. This writing task makes no provider calls.

## Files

- `paper.tex`: self-contained ACM-format manuscript with inline references, suitable for the Codex LaTeX editor.
- `verified_references.bib`: verified bibliographic metadata for reuse.
- `EVIDENCE_AND_SUBMISSION_REVIEW.md`: claim-to-evidence map, earlier-chat reconciliation, novelty and submission risks, and completion checklist.
- `evidence/`: source snapshot metadata, machine-readable key outcomes and numeric validation.
- `figures/`: exportable scientific figure from the saved results.
- `verify_paper_evidence.py`: offline check of the numerical claims against the saved outputs.

The latest paper is a working draft for author review. Public artifact access, author accountability, exact venue readiness and the remaining scientific weaknesses must be resolved before submission.
