# WEBExtension: experiment design and implementation plan

User goal: all experiments from the supplied blueprint as runnable notebooks, authentic datasets and inference, correctness checks, reproducible results. The supplied documents define candidate experimental designs, not measured findings or instructions to promise success. This implementation does not import legacy mock backends.

1. E0: checksum legacy W1–W4 and corrected W3/W4 separately; rescore all available predictions with official-compatible answer scoring; expose missing prompt/usage/source provenance.
2. Freeze authentic HotpotQA fullwiki validation input, proportional bridge/comparison sample (default 200), fixed candidate BM25 top six, 3 roles randomized to nodes, and seeds before new inference. This is candidate-based retrieval, not full-Wikipedia retrieval or browsing.
3. E1: redundant and complementary evidence assigned without gold; paired NoComm, clockwise and counterclockwise cycle recipient updates from identical initial states. Primary contrast: CW-versus-NoComm interaction P minus R. CCW is exploratory. 18 calls/question without cache reuse.
4. E2: CW versus CCW on redundant evidence; same source payloads, each sender/receiver exactly once. Reuses E1 R if identical configuration. 9 calls/question standalone including initial calls.
5. E3: one fixed 0→1 edge; equal tokenizer budget prefix versus seeded contiguous crop; same initial answer/evidence/prompt. 2 new receiver calls/question after R initial states. Exact slice failures stop rather than adding artificial padding.
6. E4: save a complete fixed-depth CW trajectory; replay first normalized unanimous non-abstaining answer. Gold cannot influence stopping. Report hypothetical avoided calls and wrong unanimity. No noninferiority claim without prespecified task-justified margin.
7. Export question-level F1/EM, support availability, validated citation diagnostics, actual delivered supporting facts, payload tokens, separate provider input/output usage and logical/physical calls, paired bootstrap intervals and sign randomization tests, and figures. Keep unknown usage/cost unknown.
8. Add seed/backbone replication and second-benchmark guidance without pooling incomparable protocols. Mean over seeds per question for clustered primary uncertainty; no selection of the best seed/model.

Validation: official scorer equivalence, no gold in prompt/partition interfaces, balanced cycles, exact token slices, paired completeness, cache/model identity and budget enforcement, notebook schema/execution in no-inference mode, real legacy rescore. No generated answers or synthetic datasets in experiment outputs. Tests exercise invariants only and are never reported as research results.

Full inference execution requires a real backend, pinned model/tokenizer identity, and declared call cap. Dataset/context limits and measured costs are inspected before scaling. Temperature zero is not a guarantee of provider determinism; saved raw responses are the reproducibility record.
