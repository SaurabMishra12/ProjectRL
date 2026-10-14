# Independent code review

Read-only review of protocol.py, backend.py, experiments.py, reporting.py, robustness.py and notebook generation found two issues:

1. Ambiguous provider failures could be retried on resume without reconciliation; request-key tracking could hide later interrupted attempts.
2. E4 cumulative costs and payloads were paired with latest-round-only evidence-delivery diagnostics.

Both were corrected and independently rechecked. Recovery now uses attempt IDs, saved pre-call requests, uncertain-outcome blocking, raw-response checksum recovery and manual provider-evidence reconciliation. A per-run file lock prevents concurrent notebooks from duplicating inference. E4 records round-level delivery traces and cumulative distinct recipient facts for the same stopped/final prefix as its costs.

The recheck found no remaining critical or important issues in these fixes. This is code review, not evidence that model inference has been executed or that a research result is favorable.
