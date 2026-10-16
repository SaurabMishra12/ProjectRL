# One approved timeout replacement

The 150-question run stopped after 54 complete questions when an Azure Round-0 call for question 55 timed out. The stopped ledger contains 464 successful completions and one uncertain attempt. No response exists for that uncertain attempt. Its generation and billing status remain unknown.

The user explicitly answered **“Yes, allow one replacement attempt”** after being told about possible duplicate billing and the frozen stop rule. This child run permits at most one replacement dispatch for that exact request key, including confirmed provider rejections. A cached successful replacement can be reused. A further timeout still halts execution.

All original artifacts remain unchanged. The child imports 464 successful response files and resolved attempt events into a new working ledger, plus a complete copy and hash of the original ledger. The uncertain event is preserved separately rather than mislabeled as rejected or successful. All 54 complete question records and the two saved Round-0 responses for question 55 are reused. Inputs, assignments, prompts, parser, routers, readouts, model identity, decoding, and tokenizer remain unchanged.

The original combined ceilings remain 1,471 possible completions, 1,531 attempts, 13,000,000 input tokens, and 6,100,000 output tokens. The unknown attempt reserves one completion, one attempt, 576 input tokens, and 4,096 output tokens. Consequently the working ceilings are 1,470 completions, 1,530 attempts, 12,999,424 input tokens, and 6,095,904 output tokens. Working history includes the imported successful calls. Missing usage and every unresolved working attempt use full reservations in conservative totals.

Prepare and verify using `python -m WEBExtension.pilots.2wiki_150_resume_20261017.resume --prepare` and `--verify`; run with `--execute`. Final offline verification and result generation use `python -m WEBExtension.pilots.2wiki_150_resume_20261017.audit_resume`.
