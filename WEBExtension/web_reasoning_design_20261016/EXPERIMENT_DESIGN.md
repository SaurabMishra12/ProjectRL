# Web reasoning experiment and SparseComm improvement specification

Date: 2026-10-16. Status: **design approved for implementation planning on 2026-10-16, not an implemented or frozen protocol**. Benchmark choice is approved: 2WikiMultihopQA primary, FinQA secondary. Numerical settings below remain versioned design defaults to lock in an exact run manifest before held-out evaluation.

## 1. Research question and success criteria

Can sparse communication deliver complementary, source-supported information between agents at lower measured cost than dense communication, while preserving answer quality? Separate this from improvements caused by stronger prompts, additional inference, or better aggregation.

Three questions have distinct evidence requirements:

1. **Routing:** with frozen initial responses, the same sink, sender count, packet cap, and one recipient update, does observable evidence-based sender selection outperform uniform random selection?
2. **System performance:** does adaptive sparse consultation offer a useful quality/cost trade-off against dense communication, static routing, strong initial aggregation, and a single full-evidence agent?
3. **Scaling:** under the same question cohort and fixed evidence union, how do physical input/output tokens, payload delivery, calls, and latency change at team sizes N=2,3,4,5?

No target accuracy or token reduction is assumed. Report a negative result if static routing, full-context single-agent reasoning, or private revision wins. A 200-question experiment is exploratory unless a prospective precision/power calculation supports a particular confirmatory claim.

## 2. Dataset protocol

### Primary: 2WikiMultihopQA

Use a publisher-linked corrected release with a compatible alias-aware official evaluator. Freeze archive hash, extracted-file hashes, source URL, release identity, evaluator commit/hash, and alias-file hash. Validate each annotated support locator against the unmodified context. Do not assume a constant paragraph count or universal gold completeness; publish observed counts and coverage.

Inference sees only `question` and publisher context title/sentence text, transformed to stable local document/sentence identifiers. The private evaluator owns answer strings, supporting facts, gold reasoning triples, gold entity IDs, aliases, and type/hop annotations. Do not pass the entire original row to prompt builders or routers.

The primary population uses the full supplied context, without BM25 top-k removal. Metadata-invalid records are recorded before inference with prespecified reasons. Support-coverage anomalies stay visible; do not quietly replace them to create a successful cohort. Use question types for prespecified sampling strata and post-hoc diagnostics, never as model-visible gold hints.

### Secondary: FinQA

Use original report text and complete tables with row labels, column/year headers, units, signs, percentages, and footnotes intact. Primary metric is official execution accuracy of a predicted program; program accuracy is secondary. A numerical answer without an executable program is not counted as an official execution-accuracy success.

Remove reference programs, executable answers, gold supporting indices, gold/model retrieval outputs, and label-dependent converted inputs from inference. Verify the table rendering against the official corrected format. Pin original sources and evaluator. Apply the same allowlisted arithmetic execution facility to every method; never execute arbitrary model-generated Python.

FinQA complements the textual multi-hop result with structured numerical evidence. It is a static financial-document task, not evidence of live website navigation or HTML comprehension.

### Splits and selection

- Engineering pilot: first 20 hash-selected training questions, stratified by available task type. Use it to check parsing, citation identity, truncation, and ledger recovery.
- Development: a disjoint 40-question training subset for prompts, packet caps, deterministic feature weights, and stopping thresholds. Report its results as development only.
- Proposed primary evaluation: 200 publisher development questions selected by deterministic type-proportional hashing, with 50 prespecified as a nested scaling cohort. Do not inspect their model outcomes during development.
- FinQA: start with a separate 20-question training pilot; propose a later held-out sample only after the primary pipeline works.
- Deduplicate exact/normalized questions across partitions. Keep connected source-title or financial-report groups within one partition where feasible; record any deviation and its sample-size consequences. Do not claim source-disjointness merely from different question IDs.
- The 200 questions are not automatically a powered noninferiority trial. If the precision analysis requires more questions, freeze a revised sample before reading primary outcomes.

Training a new learned controller requires a separate train/validation split, a model-selection rule, and an explicit inference budget. The small engineering/development subsets are not a substitute for a validated training cohort.

## 3. Evidence placement and fair task conditions

**Natural partition P, primary.** Hash-shuffle documents, then greedily assign whole documents to the least-loaded agent by pinned-tokenizer length; break ties by a seeded permutation. Repeat for N=2,3,4,5 with the same full evidence union. Do not use supporting labels to partition. Preserve document and sentence order within each assigned source. Publish actual per-agent loads and empty shards.

**Shared R, secondary.** Every agent sees the same full union. This measures redundancy and the cost of replication. It is a separate task condition; do not pool P and R in one headline score.

**Controlled split/co-located support, diagnostic only.** An offline allocator may use annotations to construct clearly named oracle placements, balanced over agents. The model and router never see those labels. Such placements measure sensitivity to annotated source allocation; they do not prove semantic necessity or represent natural retrieval quality. For FinQA, table/text separation is likewise a structured diagnostic, not the only primary partition.

Choose a sink by a frozen question-hash permutation, balanced across node indices, before initial responses. Its role is identical across methods for a question. Use the same local extraction instructions for all initial agents. Avoid assigning the strongest role or all useful evidence to the SparseComm sink. A dynamically selected sink is a later ablation, with its metadata and selection costs included.

## 4. Proposed method changes

### A. A persistent evidence ledger instead of answer-only state

Every agent returns a small ledger of source-backed facts, an optional candidate answer, missing relations, and a status. `partial` with useful evidence is a valid contribution. Local inability to answer does not delete the facts and does not create a competing answer string.

Stable fact identity is `(document_id, sentence_id)` for text, or an intact header/row locator for tables. Validate ownership and exact source text before delivery. Keep full names, negation, dates, comparators, units, and provenance. Distinguish original observations from derived claims; a citation locator being valid does not establish entailment.

Deduplicate by immutable source locator, not by vaguely similar wording. Preserve conflicts from different sources. Carry delivered evidence forward across rounds; do not replace it with only the latest candidate answer or silently remove contradictory earlier facts.

### B. Atomic packets instead of blind prefix truncation

Default proposed evidence-packet cap: **192 exact tokenizer tokens per directed delivery**, including routing envelope, source IDs, quotes, candidate claim, and uncertainty fields. Development alternatives are 96 and 384. Do not promise that full Web evidence fits in 15–50 tokens.

Build packets deterministically from the sender's validated ledger. Prioritize facts linked to an explicit requested entity/relation, then query overlap, then ledger order. Include complete atoms only. If an atom does not fit, skip it and log the reason; never split a sentence/row or drop its unit header. Omit the optional answer suggestion before dropping its supporting fact. If nothing fits, deliver an explicit unavailable packet and retain the question.

Different methods receive the same sender packet at the same frozen state. A cap is a ceiling, not a claim of exact realized-token equality. Log realized counts, complete atoms, omitted atoms, and source hashes. No padding, hidden expansion of a citation pointer, gold-selected quote, or automatic LLM compression call is allowed.

For multi-round use, a request and reply each consume a directed transmission and their full token counts. A receiver can cite only its original sources and source text actually delivered. The controller's full source store is never a free shared memory for the sink.

### C. Observable sender selection, with a simple baseline first

The first implementation should include a deterministic **SparseComm-Web heuristic**. Keep that name distinct from a learned policy. At fixed sink h, select `k(N)=max(1,floor(N/2))`, capped at N-1, distinct senders. N=2 therefore has no nontrivial sender-selection comparison.

The router receives bounded metadata per agent: status, valid-fact count, up to eight entity tags, up to two short missing-relation requests, and opaque source/fact hashes. It does not receive full source text, full rationales, gold fields, or correctness. Proposed development score for sender i is `2*J(entity_tags_i, sink_need_tokens) + J(entity_tags_i, question_tokens) + 0.5*has_valid_fact - 0.5*already_delivered`, where J is token-set Jaccard overlap. Token normalization, feature definitions, and tie permutation must be fixed in the implementation contract.

Each metadata frame has a proposed 96-token cap including headers and is charged as a control transmission. Limit each agent to six such control frames per question across all phases, including route notices; reject an over-budget schedule before executing it. A nonempty candidate answer never bypasses the evidence check. Unvalidated self-reported status can guide a routing heuristic but is not a correctness label.

The learned extension replaces hand-set scoring with a **shared per-agent encoder plus pooled team context and a shared sender-to-sink scorer**. For fixed feature width, pool dimension, and a fixed number of phases, it uses O(N) features/scoring. Maintain O(N) sink-relation state; do not flatten or reconstruct an N×N adjacency matrix. Use a linear-time top-k selection implementation if claiming linear selection complexity; a sorting implementation must be reported as O(N log N).

Train with labels/rewards on training questions only, select one checkpoint/rule on validation only, and freeze it before evaluation. Suggested reward combines official training answer quality with measured provider-token/call costs; all reward weights and training costs must be declared. Report the heuristic and learned policy separately. A legacy rank-two controller or the current three-agent categorical policy is an archival comparator, not this new architecture.

### D. Adaptive consultation after the fixed-call test

The adaptive system has at most three directed phases:

1. **Gather:** up to N-1 leaf-to-sink packets, normally k(N) selected senders; one sink update.
2. **Request:** if the sink reports unresolved needs/conflict and budget remains, send one bounded request to each selected leaf; each requested leaf gets one local update.
3. **Reply:** requested leaves return bounded source packets; one final sink update.

The sink generates request content within its previous update, without an extra planner call. A phase-2 request cannot contain hidden global source text. At most one request/reply per leaf; total data/request transmissions are bounded by **3(N-1)**. Scout-to-controller metadata and route notices are additional O(N) control traffic and must be included in total communication accounting.

With fixed packet cap L, metadata cap M, and fixed phases, communication is bounded by `3(N-1)L + 6*N*M`. This is a design bound, not a measured scaling law or a bound on all provider tokens. Initial source length, output reasoning, and number of questions also matter.

The proposed early-stop rule requires: a nonempty supported candidate, no unresolved request or explicit conflict, and at least one valid cited fact. These are observable model assertions plus structural checks, not a proof of sufficiency. Tune on development, freeze, and compare against always completing the scheduled phases. Never halt solely because all agents say `noanswer`, because answers match, or because an uncalibrated confidence is high. Hard budget exhaustion is logged separately from evidence-based stopping.

### E. Readout as a controlled factor

The designated sink's answer is the primary endpoint for the routing experiment. It can synthesize an answer absent from the original candidate set. If it is unresolved, retain an explicit abstention; do not replace it with a gold-informed alternate agent.

Compute the same locked abstention-aware vote offline for **every** method's endpoint states, including the original independent states. This strong NoComm comparator is mandatory. Use fixed normalized-answer ties and keep all-abstaining outputs explicit. Present sink and vote results side by side; do not select the better readout separately for each test arm. Also report the original vote as a historical diagnostic, not the principal baseline.

Status/refusal/invalid/transport failure are separate records. Answerability deletion experiments require a separate evaluation contract; do not treat all abstentions as wrong on an artificially unanswerable population.

## 5. Experiments and controls

### W1: one-update routing mechanism, N=3

Generate one shared initial state per question. Freeze the same sink and sender packets, then fork:

- **NoComm:** initial sink endpoint, plus the stronger all-agent abstention-aware initial readout; no new calls.
- **Private sink revision:** one sink update without peer facts.
- **Static sparse:** one predesignated sender into the sink, one update.
- **Uniform random sparse:** one seed-selected sender into the sink, one update.
- **Sparse observable router:** one selected sender into the sink, one update. Use the heuristic unless a trained policy has been frozen in advance; any additional learned arm is separately budgeted.
- **Complete In-Star:** all N-1 senders into the sink, one update.
- **Dense:** all-to-all packets, one update per agent. Score the same sink as primary and the common vote as secondary.
- **Single-agent full union:** one call over the full original source union, using the same task, source IDs, output schema, and arithmetic tools.

Private, static, random, selected-sparse, and complete-In-Star all have one recipient call. Sparse static/random/selected additionally match sender count and packet caps. Dense has more calls; its comparison is an end-to-end resource trade-off, not an isolated topology effect. At one frozen round the complete-In-Star and dense sink prompts should be identical; verify this equality and account for any physical cache reuse without pretending the logical dense system has fewer calls.

Primary routing contrast: selected-sparse minus random-sparse official answer EM at N=3. F1 and support coverage are secondary. The predetermined static baseline guards against an unnecessarily complicated router. If private revision is sufficient, report that result. This experiment by itself cannot establish multi-round routing or adaptive stopping gains.

### W2: adaptive system quality/cost

After W1 engineering validation, compare sparse gather/request/reply with complete In-Star, a static sparse consultation schedule, dense communication under the same maximum three phase slots, and private revision with a prespecified matched update-call budget. Freeze the private schedule independently of private outcomes. Report actual calls and input/output tokens even when ceilings match.

Also run a one-step paired packet-versus-private comparison at the exact same frozen receiver state and update count. For multi-round systems, trajectories diverge; do not call an outcome-dependent replay a clean topology intervention.

Both original and proposed sparse implementations may be reported, with prompt/readout changes explicitly labeled. The proposed learned version needs a new training/validation and serving provenance record. No architecture name alone establishes a performance result.

### W3: team-size scaling

Use the same prespecified 50 held-out questions at N=2,3,4,5 with fixed total source union and fixed caps. Evaluate the frozen policy without padding/tiling N=3 network weights. N=3-only training with evaluation at other N is a zero-shot transfer experiment; mixed-N training is a separate condition with declared training costs.

Measure transmissions, metadata bytes/tokens, recipient calls, total model input/output usage, local routing time/memory, wall time, and quality. Show actual dense results separately from the theoretical edge counts 2,6,12,20. At three dense rounds those are 6,18,36,60 deliveries; the proposed sparse hard data/request ceilings are 3,6,9,12, plus control traffic. Observed sparse counts may be lower.

Report paired question-bootstrap intervals at each N. Four small team sizes are a finite-range sensitivity study; fitted log-log slopes are descriptive and cannot establish a general asymptotic law. A linear hard transmission bound and empirically favorable token costs are distinct contributions.

### W4: development ablations and diagnostics

- Legacy task prompt versus new evidence-led prompt, crossed with sparse and dense routing. Do not give only SparseComm the stronger task prompt in the main comparison.
- Atomic packet selection versus prefix cropping, using the same raw sender state and cap; do not attribute gains solely to topology.
- Common vote versus common sink readout on every arm.
- Always-run versus observable stopping on the same allowed system; report any answer losses, not only savings.
- Natural versus controlled source placement; complete versus deliberately reduced evidence union as explicitly separate robustness tasks.
- Leave out answer suggestions from packets to distinguish source transfer from copying a peer answer.

The full factorial is not a mandatory first paid run. Choose the development ablations before inspecting held-out results. HotpotQA remains an offline diagnostic case study under the latest matched-readout analysis; do not silently rename or rerun the existing E9 study.

## 6. Statistics and claim policy

The independent unit is the question. Bootstrap entire question blocks, including all arms, N values, and assignment replicates, with 10,000 paired draws and a fixed recorded seed. Use exact paired McNemar for the prespecified EM routing contrast. Report absolute effects, wins/losses/ties, paired intervals, abstention/invalid rates, and per-type exploratory breakdowns. Repeated agents, seeds, readouts, and routed edges are not new samples.

If claiming preservation of quality against dense, preregister a meaningful margin, proposed **2 EM percentage points**, and require the appropriate one-sided lower confidence bound above -2, with a declared multiplicity rule for any family of such claims. Nonsignificance of a two-sided difference does not establish noninferiority. If the sample cannot resolve the margin, report descriptive trade-offs only. Use training/development discordance to plan power, not an assumed future gain.

Token savings are computed from physical serving input/output usage for the same questions, alongside logical isolated-arm cost and payload delivery. Keep training, validation, development, retrieval, packet construction, and retries visible. No currency estimate without an actual deployment tariff. Provider usage missingness is unknown, not zero.

A missing or uncertain request is not an invented `noanswer`. Preserve all assigned questions, report coverage and score bounds for unresolved outcomes, and mark the primary result incomplete while its cells are unresolved. Explicit valid abstentions on the original answerable benchmark are observed outputs, scored by the official rules. Keep model-quality, output-validity, and service-availability endpoints distinct.

## 7. Call budget: proposed stages, not spend authorization

For W1 at arbitrary N, logical completions/question are `N` shared initial + `1` full-union + `5` singleton updates (private/static/random/selected/full-In-Star) + `N` dense updates = **2N+6**. NoComm uses the initial states. This excludes training, additional policy seeds, prompt ablations, adaptive W2, retries, and provider probes.

- 20-question N=3 engineering pilot: **240 logical completions**.
- Separate 40-question N=3 development pass over this menu: **480**.
- 200-question N=3 evaluation: **2,400**.
- Scaling menu on 50 questions at all four N: `50*(10+12+14+16)` = **2,600** before reuse; its nested N=3 cell is already present in the 200-question pass, saving 600 logical evaluations if every contract/hash matches.
- Secondary FinQA 20-question pilot with the same N=3 menu: **240**.

Thus that staged W1 package totals **5,360 logical completions after the nested N=3 reuse**, before the excluded work. This is not a recommendation to run all stages immediately. Identical full-In-Star/dense sink prompts may reduce unique physical requests further; exact savings require compiled payload hashes. Logical per-arm cost still charges each arm's required computation. The actual token/currency ceiling must be computed from frozen payloads and output caps before execution.

The cheap first decision is the 20-question engineering pilot after implementation review, not an unqualified 600–800-call claim for a full study. Gate progression on parser/visibility/budget integrity rather than on whether an exploratory effect is positive. Paid execution, if later requested, needs one concrete frozen configuration and explicit limits; this design does not enable it.

## 8. Proposed implementation boundaries

Create a separate `WEBExtension/web_reasoning/` package after design review. Keep the existing E1–E9 code, gates, artifacts, checkpoints, and paper text intact.

- **Data adapters and schemas:** split public runtime inputs from private targets; official-source verification; arbitrary-N allocation; text/table atom identity.
- **Prompts and parsing:** common templates from the companion pack; versioned schemas; separate provider refusal, valid abstention, invalid output, and service failure.
- **Evidence transport:** validated ledger, atomic packet builder, visibility tracking, exact-token caps, control/data delivery log.
- **Policies:** private, fixed, seeded random, complete In-Star, dense, deterministic sparse, and optionally the trained pooled scorer. Every policy emits an explicit call/transmission schedule.
- **Runner and backend adapter:** immutable manifests, read-only legacy cache imports, request-key identity, append-only attempts, budget reservations, crash reconciliation, and mock-only local execution by default. Reuse well-tested persistence components only after checking their interfaces; no import should instantiate a provider.
- **Scoring and analysis:** pinned official evaluator adapters, common readouts, paired statistics, logical/physical/training cost ledgers, and source/result manifests.

These are interface responsibilities for review, not an executable implementation plan. A detailed coding plan follows approval of this specification.

## 9. Required acceptance evidence before any empirical claim

Meaningful local tests should cover public/private field separation; N=2..5 source allocation; permutation behavior of a learned set encoder; full-In-Star/dense sink payload equality; call-matched static/random/selected schedules; whole-atom caps including headers; no invisible source expansion; retention of partial/conflicting evidence; all-abstaining and refusal cases; output truncation; malformed source claims; official-evaluator parity; bounded transmission schedules; and deterministic resume without a second provider attempt.

Use mock responses only for engineering and mark them unscorable as empirical outcomes. Tests must detect a deliberately inserted gold field, split table unit, hidden-source citation, all-`noanswer` early stop, exceeded packet budget, and N×N state construction in the proposed scalable policy. Provider attempts remain disabled in local validation.

Before a real run, freeze exact source IDs, prompts, model/deployment identity, decoding settings, tokenizer hash, policy/checkpoint, all call/attempt/token ceilings, and run namespace. Start with a generous model-compatible output envelope, proposed 2,048 tokens, and check truncation in the pilot; do not reuse a short 384-token cap as if its adequacy were established. If engineering changes follow the pilot, issue a new version and keep the pilot out of held-out claims.

## 10. Paper use

The provisional narrative is **controlled Web reasoning → measured team-size trade-offs → HotpotQA diagnostic limitations**. Replace the first phrase with a positive showcase only after a clean result exists. Distinguish architectural guarantees, measured costs, exploratory quality estimates, and validated generalization. Any eight-page allocation is an editorial sketch until the actual venue rules are checked; no submission-policy claim is made here.
