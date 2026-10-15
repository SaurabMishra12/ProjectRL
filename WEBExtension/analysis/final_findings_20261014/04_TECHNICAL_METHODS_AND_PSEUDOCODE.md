# 4. Technical methods, pseudocode and reproducibility

**E3 record update — 15 October 2026:** E3-v2 is the current content-intervention record. Original E3 is retained as `E3_original` / historical 256-token answer-and-citation slicing, with 298/300 identical pairs. E3-v2 is a distinct protocol: 300 screened IDs, 299 eligible distinct exact-128-token evidence-bank pairs, and one explicit no-document case. These are the same question identities, not 299 additional independent test questions. It is an exploratory follow-up after original held-out results were inspected. The pre-update bundle is preserved separately.


Snapshot generated 2026-10-14T21:18:49.043266+00:00. Scope: completed frozen WEBExtension E1–E5 plus corrected exploratory E3-v2, controller training/validation/pilot, operational cost evidence, and the supplied 13-page manuscript. Original experiments use the same 300 held-out identities; E3-v2 screens all 300 and pairs 299 eligible evidence-bearing questions; the unit of inference is the question, not the arm, agent, seed, or repeated epoch. All analysis is offline; no inference calls were issued. Earlier raw artifacts and the 129-question audit are preserved. Display tables round to six significant digits; CSV/JSON files retain machine-readable precision. NA is unknown or inapplicable, never an invented zero.

Run identifiers: `run_f6aa61bac2c3dfae`, `controller_study_a0b3768357484b45`, split `controller_splits_2e025f8ae3d9815e`. Input hashes and final artifact hashes are in `final_manifest.json`.

## Scope and implementation map

This report describes the executed WEBExtension implementation, not the different continuous controller in the supplied paper. Source copies are under `sources/` and hashes in `final_manifest.json`. The knowledge graph generation 2026-10-13T17:28:45Z did not track WEBExtension; exact sources were read as fallback. No negative structural claim relies on graph absence.

| Component | Exact implementation |
|---|---|
| BM25, sharding, prompts, parsing, token slicing, majority, paired inference | `WEBExtension/protocol.py` |
| E1–E4 state construction, controls and scoring | `WEBExtension/experiments.py` |
| Source authentication and disjoint partitions | `WEBExtension/controller_splits.py` |
| 53-feature categorical policy and PPO update | `WEBExtension/learned_controller.py` |
| Rollouts, validation selection, static selection, E5 frozen evaluation | `WEBExtension/controller_study.py` |
| Main and controller reporting | `WEBExtension/reporting.py`, `controller_reporting.py` |
| Offline gold ownership and stratum inference | `WEBExtension/analysis/gold_strata.py` |
| Durable storage, worker, pacing and recovery | `WEBExtension/analysis/durable_storage.py`, `durable_runtime.py`, `durable_worker.py`, `rate_limited_machine.py`, `adaptive_pacing.py` |

## Frozen configuration

| field | value |
| --- | --- |
| base.provider | azure_v1 |
| base.model | gpt-oss-120b |
| base.model_revision | deployment_alias:gpt-oss-120b; immutable provider snapshot not exposed |
| base.api_version | v1 |
| base.tokenizer | openai/gpt-oss-120b |
| base.tokenizer_revision | b5c939de8f754692c1647ca79fbf85e8c1e70f8a |
| base.n_questions | 300 |
| base.sample_seed | 42 |
| base.partition_seed | 11 |
| base.role_seed | 17 |
| base.crop_seed | 23 |
| base.replicate_id | primary |
| base.top_k | 6 |
| base.payload_budget | 256 |
| base.temperature | 0 |
| base.max_tokens | 8192 |
| base.context_limit | 32768 |
| base.max_calls | 11400 |
| base.stop_depth | 3 |
| base.noninferiority_margin_f1 | NA (unknown) |
| base.noninferiority_justification | NA (unknown) |
| base.self_refinement_control | True |
| base.heldout_split_dir | artifacts/controller_splits_e219bbccd74a5888 |
| base.provider_refusal_policy | abstain_with_reported_usage |
| settings.split_seed | 20271014 |
| settings.train_questions | 100 |
| settings.validation_questions | 30 |
| settings.test_questions | 300 |
| settings.pilot_questions | 12 |
| settings.split_pool_per_type | 5000 |
| settings.training_seeds[0] | 7 |
| settings.training_seeds[1] | 19 |
| settings.training_seeds[2] | 31 |
| settings.collection_epochs | 3 |
| settings.ppo_epochs | 4 |
| settings.batch_size | 32 |
| settings.learning_rate | 0.0003 |
| settings.hidden | 64 |
| settings.clip | 0.2 |
| settings.entropy_coef | 0.01 |
| settings.target_kl | 0.03 |
| settings.quotas[0] | 64 |
| settings.quotas[1] | 128 |
| settings.quotas[2] | 256 |
| settings.lambda_payload | 0.05 |
| settings.lambda_calls | 0.02 |
| settings.regimes[0] | R |
| settings.regimes[1] | P |
| settings.static_candidates[0] | CW |
| settings.static_candidates[1] | CCW |
| settings.static_candidates[2] | OUTSTAR |
| settings.max_physical_calls | 46000 |

The max-token cap is a generation ceiling, not observed completion length. The tokenizer revision is pinned but the provider model is only a deployment alias; no immutable server snapshot was exposed. Provider refusals are explicit abstentions with reported usage retained. Other malformed/truncated responses stop execution and require evidence-preserving diagnosis. HTTP errors/timeouts are not made into incorrect benchmark answers.

## Data selection and leakage boundary

Publisher: hotpotqa/hotpot_qa, pinned revision 1908d6afbbead072334abe2965f91bd2709910ab. The source training files are checked against pinned hashes and 90,447 training records; the local dev candidate data contains 7,405 records. Selection uses type-stratified deterministic ID hashes, excludes the known legacy 100-item Web evaluation IDs from the held-out candidates, and blocks exact normalized candidate titles/questions between partitions. The held-out 300 questions were frozen before the controller study; the later user-requested 100/30 training/validation reduction did not change held-out bytes. The prior larger study's weights/rollouts are preserved separately; the active controller was trained afresh. Real saved responses can be reused by exact authenticated request identity.

```text
DATA_SPLITS(config):
    verify publisher file hashes and complete source row count
    heldout ← deterministic type-stratified dev selection excluding known legacy IDs
    pool ← deterministic source-train candidates with no heldout title/question overlap
    validation ← select 30 from pool, blocking heldout
    train ← select 100 from pool, blocking heldout ∪ validation
    pilot ← select 12, blocking all three
    assert pairwise disjoint IDs, normalized questions, normalized candidate titles
    save input-only files separately from annotation files; hash all files
```

Type strata are 80/20 bridge/comparison for train, 24/6 validation and 240/60 heldout. Filtering by candidate-title disjointness changes representativeness. Exact checks cannot establish semantic or pretraining isolation. Dataset type labels support stratified sampling; gold answers/support labels are not controller features.

## Common environment and metrics

```text
INITIALIZE(question, regime):
    pool ← stable top-6 BM25 ranking of publisher-provided candidate articles
    roles ← deterministic per-question permutation of retrieval/verification/reasoning
    if regime=R: give every node all six articles
    if regime=P: deterministically shuffle and assign documents round-robin to 3 nodes
    obtain/cache one structured real completion per agent
    parse answer + citations against allowed visible evidence
    return pool, assignment, roles, initial responses

DELIVER(state, directed_edges, source_payloads):
    attach each frozen sender payload to its specified recipients
    revise each recipient with at least one incoming edge exactly once
    retain initial responses at non-recipients
    record full delivery text, exact tokens, sender, recipient, source hash and facts
    return updated responses and deliveries

AGGREGATE(responses):
    normalize answers with Hotpot-style normalization
    select the most frequent answer; ties follow fixed node order
    return the first original answer string for the winning normalized answer
```

BM25 uses k1=1.5 and b=.75 over title plus sentences, with stable original-index tie breaking and log(1+(N−df+.5)/(df+.5)) IDF. There is no live network retrieval during question answering. Exact payload slicing decodes a contiguous token slice, rejects invalid Unicode or mismatched retokenization, and never pads a message. A short message transmits min(quota, length) tokens, explaining unused quota.

Answer F1 is 100×2PR/(P+R) after official normalization; EM is 100 for a normalized exact match, otherwise 0. Supporting recall is |predicted labels∩gold|/|gold| and precision divides by predicted-label count, with safe empty denominators. Additional support-set F1/EM and joint algebra are diagnostic citation-union measures, not leaderboard-validated faithfulness. For E3 the receiver's citations are used; for other experiments the union is over responses agreeing with the final ensemble answer. This final analysis corrects the earlier supplementary E3 union convention, preserving that historical report.

## E1: evidence regime × communication

Purpose: test whether the CW communication gain differs between shared and partitioned evidence. Each question has R/P × NoComm/SelfUpdate/CW/CCW (8 arms). NoComm requires 3 logical calls; the other arms require 6. The source responses and messages are frozen within a regime; semantic condition names are not inserted into model-visible prompts.

```text
FOR each frozen test question q:
    FOR regime IN {R,P}:
        s ← INITIALIZE(q,regime)
        NoComm ← AGGREGATE(s.initial_responses)
        SelfUpdate ← revise all nodes using own evidence and own prior answer, no peers
        messages[i] ← exact prefix up to 256 tokens from s.initial_response[i]
        CW ← DELIVER(s, {(0,1),(1,2),(2,0)}, messages)
        CCW ← DELIVER(s, {(0,2),(2,1),(1,0)}, messages)
        verify sender payload identity and exact total tokens between CW and CCW
        score and save all four arms
    d[q] ← (F1_P,CW − F1_P,NoComm) − (F1_R,CW − F1_R,NoComm)
PRIMARY ← mean_q d[q], paired bootstrap CI and sign-randomization p
```

SelfUpdate is a call-matched extra-computation control with zero peer transmission. R versus P also changes local evidence access; it is not an effect of topology alone. The primary interaction is fixed in advance; CCW and regime-specific contrasts remain secondary.

## E2: exact frozen-source destination intervention

Purpose: compare opposite directed cycles under R while keeping each sender's payload and each node's incoming/outgoing degree fixed. Three logical transmissions and three receiver updates are used in each arm. NoComm is retained as a reference. This uses the same authenticated calls/outcomes as E1 R and is not a second independent sample.

```text
FOR q:
    s ← the same R initialization used by E1
    freeze one prefix message per sender
    y_CW ← simultaneous one-step update over CW
    y_CCW ← simultaneous one-step update over CCW
    assert each sender transmits identical bytes/tokens in both arms
    assert total payload and logical calls match
    save F1_CW − F1_CCW
```

Both cycles are isomorphic graph motifs; the tested effect is destination assignment relative to roles/content, not general motif superiority. Updated outputs are not propagated to another node in the same round.

## Current E3-v2: executed protocol and pseudocode

Run: `e3_corrected_b5c4ff9db9b9b2bc`. Model, roles, initial answers, BM25 pool and receiver evidence are reused from the original frozen study. Sender 0 and receiver 1 remain fixed in R. The source bank contains every sender-visible Wikipedia sentence in retrieval rank and original sentence order; it does not depend on gold answers or successful generated citations. Cap 128 and all slices/eligibility/order are frozen before new inference. Seeded non-prefix candidates must be valid exact-token substrings and differ from the prefix; no padding is used. The source-empty ID `5a8cfee555429941ae14df5c` receives no fabricated message.

```text
verify original run completion and all scientific input hashes
load the same 300 gold-free inputs and saved R initial states
for each question:
    bank = all sentences actually visible to sender 0 in frozen order
    if no bank:
        freeze NO_SENDER_EVIDENCE_AVAILABLE; retain screened record
    else:
        prefix = valid first 128 exact tokenizer tokens
        crop = seeded nonidentical valid contiguous 128-token slice
        assert exact_count(prefix) = exact_count(crop) = 128
        freeze bank hash, slices, offsets and randomized arm order
for each unfinished screened question:
    for arm in frozen order, if eligible:
        reuse valid saved exact-request response if available
        otherwise persist request and dispatch real Azure completion
        persist raw HTTP body before parsing; count refusal as abstention
        update only receiver 1 with same own evidence and previous answer
    attach saved receiver NoComm and SelfUpdate anchors
    score after generation; fsync record and timings
verify 300 unique screened IDs, 299 distinct equal-token pairs and checksums
paired primary = mean(receiver_F1_prefix - receiver_F1_crop), n=299
bootstrap/sign-randomization draws=10000, seed=23
report all results and uncertainty; secondary EM/support are descriptive
```

Operational recovery resumed only unfinished questions after three explicit 429 rejections. Successful cached responses were reused; there are no unresolved attempts and no uncertain-cost replacements in this corrected run. A no-inference startup failure caused by a lazy-module import was corrected operationally; its evidence remains saved. Science sources and payloads are bound by the final run's manifest. Reports use saved outputs only, and token/frame counts do not imply exactly matched generated-token costs or latency.

Reproduce this report update offline with `HF_HUB_OFFLINE=1 python WEBExtension/analysis/final_findings_20261014/write_reports.py`, followed by `verify_bundle.py`. The generator calls `integrate_e3_v2.prepare_tables()` and `update_reports()`; neither creates a provider client or invokes inference. Frozen generation code is copied to `sources/e3_corrected.py` for audit, not execution by this analysis.


## Original E3 (historical): content selection at exact token count

Purpose: compare prefix and randomized contiguous token crop from sender 0 to receiver 1, with R evidence shared. Endpoint is receiver 1's answer, not the ensemble vote. Logical cost is three initial calls plus the one receiver update.

```text
FOR q:
    s ← INITIALIZE(q,R)
    text ← serialize sender 0 answer and cited evidence
    L ← min(256, tokenizer_length(text))
    prefix ← valid exact-L prefix
    crop ← deterministic randomized valid exact-L contiguous token slice
    FOR payload IN {prefix,crop}:
        update only receiver 1 using frozen initial state
        score receiver 1 answer and citations; save 4-call logical cost
    assert length(prefix)=length(crop)
    save whether the actual payload bytes are identical
```

The crop seed is 23 and keyed by text identity. In the realized data 298 pairs coincide, so intervention fidelity fails for a broad content-quality inference despite successful exact-length matching.

## E4: offline stopping on a shared trajectory

Purpose: measure what an exact nonempty unanimity rule would have saved on the same fully generated P trajectory. It does not train stopping and does not truncate actual inference spending in this evaluation.

```text
FOR q:
    s ← INITIALIZE(q,P); save depth-0 snapshot
    FOR depth=1..3:
        truncate current sender messages at 256 tokens
        perform simultaneous CW update; save snapshot and cumulative deliveries/calls
    chosen ← first depth whose 3 normalized answers are equal and not empty/noanswer
              (or depth 3 if none)
    stopped ← snapshot[chosen]; final ← snapshot[3]
    avoided_logical_calls ← 3*(3−chosen)
    wrong_unanimity ← chosen<3 and stopped.EM<100
    corrected_by_continuation ← wrong_unanimity and final.EM=100
    save all snapshots, not only the selected one
```

Initial-state stopping is allowed at depth 0. Counterfactual logical costs are separated from physical calls. Zero observed F1 differences are not a predeclared noninferiority result.

## E5: categorical contextual-bandit PPO

Action space: one NoComm action plus 18 supports × 3 quotas = 55 actions. The supports are all 15 two-edge subsets of six non-self directed edges, CW, CCW and the six-edge broadcast. Quotas are 64,128,256. The actor takes 53 observable features: two global, 9 per node (27), and 4 for each of six ordered sender–receiver pairs (24). Inputs include answer diversity, citation/invalid-label counts, own-document counts, message length, role indicators, lexical overlaps, citation novelty and evidence overlap. No answer correctness, gold support, question ID hash or running accuracy feature is supplied to the policy.

Network: 53→64→64 Tanh trunk; categorical actor with 55 logits and scalar critic. The Adam learning rate is .0003. The actor output is sampled in training and argmax at validation/test. This is a one-step contextual bandit, not the manuscript's multi-round Gaussian/GAE model.

Let F1_0 be NoComm F1 and V_full=2Σ_i len(message_i), counting full initial messages on both outgoing edges. Let U be the number of distinct updated recipients. The reward is

r = (F1_action−F1_0)/100 − 0.05 V_action/max(1,V_full) − 0.02 U/3.

This reward uses annotations offline. The denominator uses untruncated full sender messages, whereas the evaluated Broadcast baseline has quota 256 per edge; those are different quantities for long messages. NoComm has reward zero by construction. SelfUpdate incurs the three-update penalty without peer payload.

```text
TRAIN(seed):
    initialize actor/critic and optimizer using frozen seed
    FOR collection epoch=1..3:
        FOR each of 100 train questions and both R/P contexts:
            s ← INITIALIZE(q,regime)
            x ← observable_features(s)
            action, old_logprob, old_value ← sample actor(x)
            result ← execute action with real cached/provider responses
            reward ← F1 gain − payload penalty − recipient-call penalty
            durably save x, action, reward, old logprob/value, outputs and RNG state
        advantage ← standardize(reward−old_value), population standard deviation
        FOR up to 4 PPO passes over shuffled minibatches of 32:
            ratio ← exp(logprob−old_logprob)
            approximate_KL ← mean(exp(logratio)−1−logratio)
            stop update loop if KL>.03
            actor_loss ← −mean(min(ratio*A, clip(ratio,.8,1.2)*A))
            value_loss ← .5*mean((value−reward)^2)
            total_loss ← actor_loss + .5*value_loss − .01*entropy
            backpropagate; clip global gradient norm at .5; Adam step
            save optimizer/checkpoint diagnostics
        evaluate deterministic actor on 30 validation questions × R/P
        select checkpoint only if mean validation utility strictly improves
    freeze selected checkpoint hash

SELECT_STATIC:
    evaluate prespecified CW, CCW, OUTSTAR at quota256 on validation R/P
    choose highest mean validation utility (OUTSTAR in this run)

TEST:
    authenticate all checkpoints, split and static-selection hashes
    create/verify evaluation gate; permanently close training for this study
    FOR each of 300 heldout questions and both regimes:
        reuse one frozen initial state for every arm in this suite
        evaluate NoComm, Static, Broadcast, SelfUpdate
        FOR seed in {7,19,31}:
            learned_action ← argmax selected actor(observable_features)
            evaluate learned_action
            eligible ← action-space entries with same edge count and quota
            random_action ← deterministic choice keyed by question/regime/seed
            evaluate random_action
        durably save all 20 arm results for the question
    assert exact frozen 300-ID set, arm completeness and result checksum
```

Matching random edge count/quota can leave different recipient counts and realized tokens. Random selection can choose the same action as learned; cache reuse is correct in that case. Static selection spans three candidates, not exhaustive best-static validation. Three optimizer seeds do not provide independent backbone realizations. Frozen argmax replay matched every saved action in this audit.

## Statistical pseudocode and strata

```text
PAIRED_EFFECT(A,B):
    require exactly the same nonempty question IDs
    d ← array[A[q]−B[q]] in sorted question-ID order
    bootstrap 10,000 question samples with replacement; store mean(d*)
    CI ← empirical 2.5th/97.5th percentiles
    randomize signs independently per paired question 10,000 times
    p ← (1 + number(|randomized_mean| ≥ |observed_mean|−1e−12))/(10000+1)
    return mean, CI, p, helped/harmed/tied question counts

E5_CONTRAST:
    average learned scores across 3 seeds within each question
    do the same for MatchedRandom when used
    apply PAIRED_EFFECT to question-level means and baseline
    Holm-adjust the prespecified reporting family (20 F1/utility contrasts)

GOLD_STRATUM(q):
    derive the two annotated supporting article titles offline
    find their owners in the actual P assignment
    if either absent: Retrieval-Missing-Gold
    else if owners differ: Disjoint-Gold
    else: Co-located-Gold
```

Stratification is post hoc and observational. Preserve pooled primary results, all subgroup sizes, and the stored Holm families. Additional subgroup/type/trajectory diagnostics in this report use marginal 5,000-resample intervals and are explicitly descriptive. Do not interpret intervals as simultaneous or certify equivalence from a failure to reject zero.

## Persistence, recovery and usage accounting

Scientific manifests bind sources, settings, samples, prompts, tokenizer and selected weights. Model-visible prompts do not contain cache arm labels; internal cache namespace isolates controller and controlled suites. Canonical responses, staged HTTP bodies/receipts, request hashes and append-only call ledgers provide recovery before another physical request. Explicit authorization binds an uncertain original attempt before a replacement; original unknown costs survive. Runtime uses offline dataset access, bounded shared request slots and adaptive rate pacing; it never silently retrains after the held-out gate. This analysis consumed saved outputs only.

## Reproducing this report bundle

From the project root, with the original artifacts present:

```bash
HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sparsecomm_mpl python WEBExtension/analysis/final_findings_20261014/analyze_saved.py
HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sparsecomm_mpl python WEBExtension/analysis/final_findings_20261014/add_details.py
HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sparsecomm_mpl python WEBExtension/analysis/final_findings_20261014/deeper_analysis.py
HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sparsecomm_mpl python WEBExtension/analysis/final_findings_20261014/policy_diagnostics.py
python WEBExtension/analysis/final_findings_20261014/supplemental_register.py
python WEBExtension/analysis/final_findings_20261014/write_reports.py
```

These scripts regenerate derived reports in this new folder and construct no provider client. They require the preserved dataset, checkpoints, tokenizer cache and source manifests. Run in the listed order; preserve or copy the report folder before a future regeneration if you need immutable versions. The original experiment sources/results and the older 129-question publication audit are not edited. The supplied paper extraction is included in `sources/current_paper_extracted.txt`; future report generation can use that saved extraction if the original temporary extraction is unavailable.
