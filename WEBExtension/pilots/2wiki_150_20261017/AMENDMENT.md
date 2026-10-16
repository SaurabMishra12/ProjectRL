# Authorized extension to 150 questions

The user requested “increase the 20 to 150” after the revised 20-question pilot failed its expansion gate. This authorizes 150 total questions, including the completed 20, without rewriting the historical decision.

Keep all seven arms, both readouts, common relaxed prompt/parser, deterministic routers, model identity, decoding, packet cap and tokenizer unchanged. Preserve the existing 100 frozen questions and assignments. Add 50 eligible publisher training questions with seeded, type-stratified sampling and no normalized question or source-title overlap with the preceding 100 or each other. Supporting labels establish completeness in the union; they do not control assignment or routing.

Copy the verified completed pilot cache and full physical ledger lineage into a separate run directory. Preserve the original directory and result envelopes. Importing the 171 completed responses incurs no provider calls. Reuse Round-0 and byte-identical Sink-1 requests during the additional 130 evaluations.

Hard cumulative ceilings, including imported usage: 1,471 physical completions, 1,531 attempts, 13,000,000 input tokens, 6,100,000 output tokens; per-call output cap 4,096. Thus at most 1,300 additional completions and 1,360 additional attempts. Missing usage uses the durable backend's conservative reservations. Uncertain transport halts and blocks restart; only confirmed provider rejections may retry within the fixed budget.

Before dispatch: test cache lineage, corruption rejection, result identity, retained failed gate, assigned denominators, source separation, and support completeness; freeze inputs and consumed source hashes; verify offline. During execution: persist every physical attempt and sealed question record. At completion: verify 150 unique paired rows, unchanged original artifacts, exact Sink-1 request identity, physical accounting, and all schema outcomes. Report the original 20, additional 130, and pooled 150 separately with exploratory paired intervals. No performance-based early stopping or outcome-driven tuning.

Run `python -m WEBExtension.pilots.2wiki_150_20261017.amendment --prepare`, then `--verify`, then `--execute` from the repository root.
