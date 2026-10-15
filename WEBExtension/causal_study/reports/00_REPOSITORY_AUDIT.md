# Repository and saved-artifact audit

Status: **COMPLETE_OFFLINE**, 15 October 2026. This audit made **zero inference requests** and changed no historical experiment outputs. Its scope is the current `WEBExtension` implementations, the completed frozen E1–E5 artifacts, corrected E3-v2, the `analysis/final_findings_20261014` bundle, split/provenance records, and the preserved manuscript extraction. Historical W/P experiments remain a separate evidence tier.

The parent performed a graph Verify preflight against `projectRLMAIN`, generation `2026-10-13T17:28:45Z`. Graph queries returned no WEBExtension symbols, and coverage reported `not_tracked`/`read_source_and_reindex` for the material files. This is outdated graph coverage, not evidence that the code is absent. All material findings below therefore use exact-source fallback and saved artifacts. Source and artifact SHA256 values are recorded in `../offline/provenance.json`; recomputed inventory facts are in `../offline/repository_inventory.json`.

## Ten required implementation questions

| Question | Source evidence | Finding and implication |
| --- | --- | --- |
| 1. What is one actual Azure request? | `backend.py:103–160`, `RealBackend.complete`; `analysis/durable_runtime.py`, transport capture | One invocation of `client.chat.completions.create(**params)` on a cache miss is one HTTP inference attempt; SDK retries are disabled. The request contains one message conversation and asks for one completion. Cache reads, votes and gold scoring are not physical requests. Retries are additional attempts even when their prompt key is unchanged. |
| 2. How many requests does each experiment use? | `experiments.py:63–72`, `preview`; `run_question:171–242`; `controller_study.py:31–40`, `forecast`; ledgers below | E1 has at most 24 unique logical generations per question with both regimes and SelfUpdate; E2 9 standalone and zero additional after E1; E3 5 standalone and at most 2 additional after E1; E4 12 for three rounds standalone; E5 at most 60 held-out generations per question for both regimes/three seeds before exact-cache reuse. Actual physical counts are lower and cannot be obtained by summing arm costs. Corrected E3-v2 generated 598 saved responses in 601 attempts for 299 eligible pairs. |
| 3. Does NoComm aggregation need an LLM? | `protocol.py:197–202`, `majority`; `experiments.py:132–158`, `condition` | No. It normalizes answer strings and selects an unweighted majority, resolving ties by first node occurrence in the frozen order. E3 uses receiver 1 rather than a final model or ensemble endpoint. No aggregation completion should be budgeted. |
| 4. Can initial responses safely be reused? | `backend.py:59–60,103–117`, cache identity; `experiments.initialize`; `e3_corrected.execute` | Only for exactly equal model identity, generation parameters, ordered messages and pinned tokenizer identity. Current cache also namespaces by replicate ID. Role, evidence assignment/order, own prior answer or peer text changes must change the key. Cache metadata alone is not part of identity. E6 must never reuse initial responses across different assignments. E8 validates the exact initial call-key prefix before importing E1 initial states into E3 analyses. |
| 5. Does communication change the number of agent updates? | `experiments.py:94–110`, `update`/`no_peer_update`; `controller_study.py:59–70`, `routed_cost` | Yes. Only distinct recipients update. A two-edge in-star updates one agent; a two-edge chain can update two; cycle and broadcast update three. Original NoComm makes zero updates, while SelfUpdate gives all three agents a no-peer revision. Thus Comm–NoComm bundles information and extra computation unless the additional computation control is used. |
| 6. Is there hidden shared source context? | `protocol.py:109–132`, `assign_evidence`/`build_prompt`; `experiments.py:74–106`, `visible_documents`/`initialize`/`update` | Model-visible evidence is exactly the assigned documents, its own previous answer, and delivered peer strings. Shared R gives all retained documents to all agents; P assigns disjoint documents. `visible_documents` references the union only to validate returned citations, not to add source text to the prompt. Final deterministic voting sees candidate answers; it does not give independent NoComm agents their peers' evidence. |
| 7. Do prompts include gold labels? | `experiments.py:80–90`, explicit whitelist; `protocol.py:124–132`, `build_prompt`; `condition` post-generation scoring; `controller_splits.py:18–20`, `inference_inputs` | The inspected inference builders do not include `answer` or `supporting_facts` from dataset labels. The `previous_answer` field is a model proposal, not the gold answer. Natural evidence may contain the answer and is not label leakage. Gold is used for scoring, training rewards and post-hoc availability diagnostics. Controlled E6/E7 may use support labels to assign evidence but must keep those annotations out of inference payloads. This is source-path validation, not a guarantee against benchmark pretraining contamination. |
| 8. Are repeated response states identical? | `backend.py` exact-cache key; `controller_study.py:42–45`, `runtime`; `final_findings.../audit_metrics.json`, `cross_suite_baseline_overlap` | Within E1/E2, scientific answers, payloads and provider keys are identical; cache/timing metadata can differ. E5 uses a separate controller replicate namespace and has no identical initial keys with E1; prior audit finds 106 answer disagreements among 1,200 same-question baseline comparisons. Temperature zero does not certify cross-run identity or immutable deployment state. Compare paired treatments within their frozen response state. |
| 9. Can fixed one-round communication run without retraining? | `experiments.run_question` E1/E2/E3; `protocol.CYCLES`; `controller_study.apply_action`, broadcast construction in `evaluate` | Yes. Directed CW/CCW updates and fixed broadcast are already implemented. New controlled placement can use fixed broadcast with three receiver updates per assignment and the same deterministic final vote. PPO is unnecessary for identifying placement effects and will not be trained for this causal pilot. |
| 10. Is token accounting reliable and concurrency bounded? | `backend.py:69,76,175–180`; `analysis/rate_limited_machine.py:26–34,68–90`; `analysis/adaptive_pacing.py`; frozen config | Provider usage is recorded, but some refusal responses omit output usage and uncertain requests have unknown liabilities. Token totals are therefore lower bounds where fields are missing. Existing operational pacing limits network requests to two and handles explicit 429 rejections separately; each scientific pipeline is serial. The base backend alone has no global concurrent success/attempt gate. The new pilot needs its own durable 150-success/180-attempt limiter. No verified deployment price or billing export exists, and the deployment is an alias whose immutable snapshot is not exposed. |

## Saved experiments and actual physical resource evidence

All E1–E5 result files have 300 unique, identical question IDs and match their completed status SHA256. Corrected E3-v2 screens the same 300 and has 299 treatment pairs plus one explicitly ineligible question. Their union is **300 independent question identities**, not 1,799 or 1,800 independent observations. E8 recomputed 90,286 answer-reference comparisons, including repeated references, against the preserved official HotpotQA evaluator: zero mismatches.

| Experiment | Distinct request keys referenced by saved arms | Sum of arm logical calls, including reused initials | Interpretation |
| --- | ---: | ---: | --- |
| E1 | 5,925 | 12,600 | Both evidence regimes, NoComm, SelfUpdate, CW, CCW. Upper bound 7,200 generations before cache reuse. |
| E2 | 2,001 | 4,500 | Same R calls as E1; no independent replication or additional necessary inference. |
| E3 original | 1,202 | 2,400 | Includes 900 R initials and 302 distinct receiver requests; 298/300 content pairs are identical. |
| E4 | 2,036 | 7,146 | Includes reused initials and stopped/final overlap; full trajectory originally generated. |
| E5 | 7,634 | 28,368 | Held-out referenced keys; controller ledger also contains training and validation. |
| E3-v2 | 1,798 | 4,492 | Includes reused initial/SelfUpdate anchors; only 598 new receiver responses. |

These columns are not invoice counts and must not be summed: exact keys recur across arms, rounds and E1–E4. Requests that failed or became uncertain may also be absent from completed scientific conditions. The original ledger rows provide the physical accounting:

| Ledger | Started HTTP attempts | Saved success events | Explicit rejected events | Known input tokens | Reported output tokens lower bound | Success rows missing output usage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `artifacts/run_f6aa61bac2c3dfae/calls.jsonl` | 6,153 | 6,134 | 11 | 5,960,203 | 1,582,163 | 29 |
| `artifacts/controller_study_a0b3768357484b45/inference/calls.jsonl` | 10,037 | 10,019 | 13 | 10,130,139 | 2,640,066 | 37 |
| `artifacts/e3_corrected_b5c4ff9db9b9b2bc/calls.jsonl` | 601 | 598 | 3 | 902,605 | 166,539 | 6 |

Input usage is present in every success row in these three ledgers. The controlled ledger retains eight historically uncertain liabilities and the controller ledger five, including interruption/recovery bookkeeping; successful authorized retries do not resolve the original attempt's bill. The older bundle's global, deduplicated inventory additionally includes archived runs: 18,568 attempts, 18,524 successes, 30 explicit rejections and 14 uncertain original liabilities. That number must not be presented as the new pilot's consumption. Monetary cost is **UNVERIFIABLE_FROM_SAVED_ARTIFACTS**.

## What the completed studies support

The source and completed output evidence supports the supplied qualitative context, with exact scope:

- E5 shared R: NoComm F1 32.3140, learned three-policy seed mean 32.1084, broadcast 31.6510. The mean is a statistical summary, not an implemented ensemble. There is no established shared-evidence learned benefit.
- E5 partitioned P: NoComm F1 5.0598, learned seed mean 10.5604, broadcast 14.9920. Learned–NoComm is +5.5006 and learned–broadcast −4.4316 F1 points. These are within-suite whole-policy comparisons.
- E1 primary interaction is +2.8815 F1 points, 95% bootstrap CI [−0.2600, 6.0109], paired randomization p=.07129. A positive P contrast alone does not identify a conventional significant interaction.
- Recomputing article availability from raw E1 assignments gives both supporting articles retained on 76/300 questions: 62 distributed and 14 co-located. At least one annotated article is missing on 224. Both were in the supplied candidate pool on 79; 221 questions already lacked at least one before top-six selection. Candidate absence and ranking loss are distinct. Missing annotations do not certify answer impossibility.
- E5 matched-random controls match edge count and quota on all 1,800 regime/seed/question comparisons, but actual logical calls differ on 912 and transmitted token totals differ on 254. This does not isolate routing from physical resources.
- P seed 7 chooses broadcast on 294/300 questions and matches broadcast final answers on all 300. P seed 31 uses one two-edge in-star on all 300. Frozen source/weights and near-static chosen actions do not establish learned topology superiority.
- E4 generates all trajectories, then replays the first eligible unanimity stop. Ten questions stop early, six early consensuses are wrong, and stopped/final answers coincide on all 300. The 54 logical calls avoided are hypothetical replay savings, not measured online latency or spending reductions.

Corrected E3-v2 is a completed exploratory follow-up, not a missing experiment to repeat. Its saved completion gate passes and the current result hash matches. It uses every sender-visible evidence sentence as its source bank, with distinct 128-token prefix/crop pairs on all 299 eligible questions. Prefix–crop receiver F1 is −1.7954, CI [−4.1973, 0.4708], p=.14019, with 8 helped/18 harmed/273 tied. Six explicit provider refusals remain abstentions. Prefix preserves 641 complete parseable fact lines versus crop's 500, but this is not proof of information use or entailment. The receiver already sees all R documents. Therefore E3-v2 identifies redundant presentation differences rather than genuinely novel evidence access; token cropping can break structured sentences. E7 must use semantically distinct intact source passages and frozen receiver state instead of repeating this manipulation.

## E8: proposal availability and aggregation are distinct bottlenecks

`../offline/e8_question_conditions.jsonl` contains 12,898 observed question/condition records across 43 groups, including E4 depths 0–3. `e8_summary.json` gives EM counts, continuous F1, question-level descriptive Wilson intervals, bootstrap intervals for F1 change, mutually exclusive transition categories, and explicit unobservable labels. Two missing E3-v2 treatment records are `NOT_RUN_INELIGIBLE_NO_SENDER_EVIDENCE`; the no-peer anchors remain observed. E3 aggregation rejection is `null` with `NOT_APPLICABLE_RECEIVER_ENDPOINT`, because its actual endpoint is receiver 1. E3's hypothetical majority diagnostic is labelled separately.

The E5 P NoComm state has an exactly correct agent on **67/300** questions, yet majority is exactly correct on **10/300**; it rejects a correct candidate on 57 questions. After broadcast, any-agent correctness is **66/300**, but majority correctness is **34/300**, with 32 rejected correct candidates. Of those 34 final exact answers, 32 questions already had an initially exact proposal; only two had none. Among 25 questions whose initially wrong vote becomes exact, 23 already had a correct candidate initially; one initially exact vote becomes wrong. Broadcast causes no observed expansion of question-level any-agent exact coverage here. Its higher endpoint accuracy is consistent with propagation/selection of existing correct answers, while new proposals and harmful revisions also occur.

Seed 19 and seed 31 leave 48 and 49 questions with a correct revised candidate rejected by majority, versus 32 for broadcast/seed 7. These counts do **not** isolate why broadcast succeeds: recipient calls, topology, transmitted information and state changes differ. Nor do they prove majority replacement will improve practical performance; identifying the correct candidate uses gold. The any-agent oracle EM and best-agent oracle F1 are diagnostic upper bounds only. E5 P NoComm's oracle best-agent F1 is 30.8152 versus practical majority F1 5.0598; broadcast's corresponding values are 30.5415 and 14.9920.

E1 shows a related pattern within its own state namespace: P initial any-exact 68, initial majority exact 11; CW revised any-exact 69, final majority exact 20. E1 R starts with any-exact 79, majority exact 68; CW ends at 76 and 73. Communication can improve selection while losing some individually correct proposals. Separate questions with new answers, agent-level harmful revisions, and final-vote rejection are retained in the raw E8 table. An observed new correct answer after a peer message is temporal evidence; actual use of the delivered facts is **UNOBSERVABLE_FROM_ANSWER_AND_CITATION_TRACE**.

## Manuscript and prior design alignment

The current supplied manuscript extraction is preserved at `analysis/final_findings_20261014/sources/current_paper_extracted.txt`; it describes historical W1–W4 role/prompts/calibration and is not an exact description of the current E-suite. The current E-suite uses retrieval/verification/reasoning roles, exact pinned-tokenizer payload counting, deterministic majority voting, and explicit source availability diagnostics. Historical word-proxy claims, stopping claims, or learned optimality cannot be promoted to new E-suite causal findings. The earlier `references/WWW2027_SparseComm_Research_Blueprint.md`, `CONTROLLER_RETRAINING_PLAN.md`, and `analysis/e3_correction_20261015/DESIGN_AND_NARRATIVE.md` motivate possible designs but do not provide new experiment outcomes or authorize confirmatory spending in this task.

Existing train/validation/held-out/pilot splits check exact IDs, normalized questions and normalized candidate article titles for overlap. They do not establish semantic duplicate absence, title alias disjointness or pretraining isolation. The held-out 300 have already been repeatedly inspected and used to motivate E3-v2 and the present narrative. New causal pilot questions must be deterministically selected without old performance conditioning; a later confirmatory sample must remain untouched during design and pilot selection.

## Sufficiency before paid pilot

The old code can supply prompts, data and fixed communication, but it does not itself enforce the new authorization boundary across experiments or distinguish 150 returned completions from 180 total attempts globally. The new protocol must locally validate equal unions/assigned document counts, absence of gold annotations, intended delivery, cache/resume identity, identical aggregation, question-level statistics and hard shared budget limits before paid inference. Missing prices do not authorize invented monetary costs; use completion/attempt caps and report provider tokens with missingness. No additional E1–E5 or E3-v2 calls are required for this audit or E8.

Reproduce the offline analysis with `python3 WEBExtension/causal_study/offline_e8.py` and test it with `python3 -m unittest discover -s WEBExtension/causal_study/tests -p test_e8.py -v`. Seven targeted guards pass. The code verifies completed result checksums, identical frozen IDs, linked initial call keys, implemented endpoint identity, official answer scores and question-level summary uniqueness; it does not instantiate a provider client. Old artifacts remain read-only.

## Exact source and artifact hashes

The following register is generated from the inspected files. The complete hash inventory, including result files, ledgers, previous analytical reports, frozen manifests and generated E8 artifacts, is `../offline/provenance.json`.

| Source or artifact, relative to WEBExtension | SHA256 |
| --- | --- |
| `protocol.py` | `30e18df0949f40b152131dca3c65d9a2a897080f0451a7f989dcf34a3fa28226` |
| `experiments.py` | `398842af07b16ef47eb26ca538a2479aa34cd0410d9bc567fbaa2d5bb1d37485` |
| `backend.py` | `5688bf903c81513af6ed38ea2435d77e3fb4a6d743a0a3da376c2ad560f29d34` |
| `controller_study.py` | `6479d85e4f450d6210af1c7b846768b98674a30d2389d82063607a2c84684a0b` |
| `learned_controller.py` | `ec648eea11ee2e724002c22a8f80d611e71fd3ae9dd80e65c0e5550b43359b17` |
| `controller_splits.py` | `06bf2ffbfebc6eb45f36e27c32c622e3f211a918bb93e487c84e7c6c789ac0b1` |
| `e3_corrected.py` | `f38aba3ee265517245fd891c5e209b85f12c85b0e2f032fcfa0b64078596f386` |
| `artifacts/run_f6aa61bac2c3dfae/E1.jsonl` | `1afe9ce6293a85acdcc743ee18b81c9ae79d57cd8258697cd715e77b4748a608` |
| `artifacts/run_f6aa61bac2c3dfae/E2.jsonl` | `dfae111876fc8fbcc3aaadd7ddff050909b3e76f0bb2e9ee51db7fa7490ed5c1` |
| `artifacts/run_f6aa61bac2c3dfae/E3.jsonl` | `793f10a77da7556cec6adec23e0bcbdd8e99cf5361ca977d202f61229cd2c8c6` |
| `artifacts/run_f6aa61bac2c3dfae/E4.jsonl` | `10ec23783228e25ab1002d0aeb241e7dd5f30f0b6e8a67e41d6f5b4cfe72084f` |
| `artifacts/controller_study_a0b3768357484b45/E5.jsonl` | `cf34e15527704024760f6ceaf11a34f62cfbdf000c08422bd6568f165ef7beca` |
| `artifacts/e3_corrected_b5c4ff9db9b9b2bc/E3.jsonl` | `6b6e325d198f35e7885ea6d4d832cb636c4d982b5885013b0d0d604797ef9779` |
| `analysis/final_findings_20261014/audit_metrics.json` | `367473404d0f9ca9868146536da51f7176a3bb11633f28d4a0b8f358fca53302` |
| `analysis/final_findings_20261014/E3_v2_updated_analysis.json` | `5034ba83c2807340a83095319dddf06b1399aa5b914a3edfd9fbdb62a4aa7d65` |
