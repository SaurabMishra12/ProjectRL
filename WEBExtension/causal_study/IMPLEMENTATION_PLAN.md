# Controlled causal study implementation plan

The task specification is the user attachment `Pasted text.txt` supplied on 15 October 2026. The user authorizes autonomous design, implementation, local verification and a pilot only, capped at 150 successful completions and 180 HTTP attempts across E6/E7. Existing artifacts must remain untouched.

- [x] Audit WEBExtension source, old artifacts, manuscript alignment and E3-v2; hash evidence and compute E8 offline (audit_e8).
- [x] Freeze the corrected protocol, deterministic unexposed sample and counterbalanced six-document assignments; implement E6/E7 pure prompt construction and invariants (protocol).
- [x] Implement question-level paired statistics and conservative power sensitivity; independently challenge causal interpretation (review_stats).
- [x] Implement a serial, persistent paid budget, exact request cache, raw request/response provenance, safe retry and uncertain-request stop policy; test before inference (root).
- [x] Run an entirely mocked study including resume and refusal/error/budget cases. Resolve all critical validity issues before paid inference (root/team).
- [x] Freeze configuration, manifests, source snapshots and preregistration. Execute at most eight E6 questions with 18 calls each, preserving partial results (root).
- [x] Analyze successful complete question blocks, raw mechanisms, E8 and resource costs; generate the eight required reports, figures and reproducibility instructions; independent final critique (root/team).

## Design ruling before outcomes

Use eight questions with three initial calls, three private self-revision calls and three fixed broadcast revision calls per placement (18/question, 144 total). The original 12/question design confounds communication with revision computation. Deterministic initial majority remains a zero-extra-call anchor. E7 paid inference is deferred because six remaining completions cannot support a useful paired pilot. E7 implementation and distinctness checks remain required.

## Ownership and safeguards

All changes live under `WEBExtension/causal_study/`; no modifications to prior source or data. The graph generation is 2026-10-13T17:28:45Z and does not track current WEBExtension files; source fallback is required. No training, publishing, pushing, deployment switching or full evaluation. No credentials in artifacts or tool output. Mock and paid ledgers are separate; the paid study uses one persistent global ledger and exclusive lock. Unknown execution status blocks further paid calls, including on resume.
