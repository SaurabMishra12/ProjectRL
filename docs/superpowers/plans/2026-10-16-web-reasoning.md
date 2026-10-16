# SparseComm Web Reasoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independently testable Web reasoning experiment package with evidence-preserving sparse communication, strong matched controls, arbitrary team sizes, and auditable physical costs.

**Architecture:** A new `WEBExtension/web_reasoning/` package separates public evidence from private evaluation targets, validates local proposals, transports complete source atoms, and schedules explicit recipient updates. A durable backend records physical attempts independently from logical arm references. The fixed-call W1 runner is completed before adaptive W2, scaling W3, development ablations W4, and the optional learned policy are enabled.

**Tech Stack:** Python dataclasses and standard-library JSON/hash/path utilities; pytest; NumPy; existing filelock, transformers, and OpenAI-compatible infrastructure; PyTorch only for the optional learned scorer. Official evaluator dependencies are isolated from the offline package import path.

**Spec:** [Approved experiment specification](../../../WEBExtension/web_reasoning_design_20261016/EXPERIMENT_DESIGN.md), [prompt pack](../../../WEBExtension/web_reasoning_design_20261016/PROMPT_PACK.md), and [evidence audit](../../../WEBExtension/web_reasoning_design_20261016/EVIDENCE_AUDIT.md). The user's 2026-10-16 “Continue” approves the specification for implementation planning. This plan still requires review and an execution-method choice before product code is written.

## Global Constraints

- “2WikiMultihopQA primary, FinQA secondary.”
- “Keep the existing E1–E9 code, gates, artifacts, checkpoints, and paper text intact.”
- Evidence-packet cap: “192 exact tokenizer tokens per directed delivery”; development alternatives: “96 and 384”.
- Metadata cap: “96-token cap including headers”; “six such control frames per question” per agent.
- At most three directed adaptive phases; data/request transmissions bounded by “3(N-1)”; total bounded packet/control traffic: `3(N-1)L + 6*N*M`.
- Sender budget: `k(N)=max(1,floor(N/2))`, capped at N-1; evaluated team sizes N=2,3,4,5.
- Common task instructions and readout contract for every topology; no hidden global source expansion.
- “Provider attempts remain disabled in local validation.” Mock data is engineering evidence only.
- Primary pilot: 20 training questions; development: disjoint 40 training questions; proposed evaluation: 200 development questions; nested scaling cohort: 50; FinQA pilot: separate 20 training questions.
- W1 logical completions/question: `2N+6`; N=3 pilot: 240. Full staged W1 menu: 5,360 after nested N=3 reuse, excluding additional work listed in the specification.
- Paired question-level resampling: 10,000 draws; proposed noninferiority margin: 2 EM percentage points; no nonsignificance-to-equivalence inference.
- Proposed generation envelope: 2,048 output tokens; pilot must expose truncation rather than presume adequacy.
- No dataset/model downloads during unit tests. A pinned local tokenizer is required for real payload accounting; a fixture tokenizer must be labeled mock and rejected by scientific reporting.
- No provider credentials in manifests, logs, fixtures, prompts, or command output. A separate concrete run authorization is required before paid dispatch.

## Review Focus

1. Duplicate source titles or divergent sentence segmentation: reject ambiguous locators before allocation, rather than silently citing another paragraph. Tests: Task 2.
2. Tiny caps, multibyte text, and table units longer than the remaining packet: emit a bounded unavailable packet or explicit configuration failure; never split an atom or exceed the cap. Tests: Task 4.
3. A stale cache or two processes sharing a run directory: immutable identity and a serial lock must prevent double execution or reuse across models/tokenizers. Tests: Task 6.
4. Refusal, malformed successful output, and unknown transport status: preserve distinct outcomes and usage, with no fabricated answer or automatic uncertain retry. Tests: Tasks 3, 6, and 8.
5. Missing gold aliases, operands, or report unit context: fail validation or preserve an unresolved endpoint; never silently relax the official scoring contract. Tests: Tasks 2, 8, and 12.

## Execution and source evidence

Planning used graph project `projectRLMAIN`, Tier 2, generation `2026-10-13T17:28:45Z`. A targeted graph search did not return the relevant Web symbols. Coverage marked the inspected Web paths `not_tracked`, so exact source reads were used. This is a coverage limitation, not an absence claim.

The existing `analysis/durable_storage.py` imports top-level `backend`, and `analysis/durable_runtime.py` installs runtime hooks into old modules. Do not import those hooks into the new package or mutate their globals. Port the small canonical/atomic/journal primitives with attribution to their source, and test the independent state machine. Do not inherit E6/E9 authorization or attempt budgets.

At execution time, use the worktree skill to choose suitable isolation. The proposal files are currently outside the tracked source set inspected by `git ls-files`; preserve/copy these exact approved documents into the implementation checkout if needed. Do not copy unrelated untracked work. All creation paths below are relative to the implementation checkout. Existing modified external-validation files are outside this plan.

For every task: read its source references, write the indicated failing tests, run them to confirm the intended failure, implement the minimum behavior, rerun the owning tests, then commit only that task's explicit files. Test snippets specify required cases and public behavior, not the entire suite. Tests must not contact a provider. Expected PASS/FAIL statements below are future acceptance criteria, not results from this planning turn.

## File map and dependency order

- `WEBExtension/web_reasoning/contracts.py`: immutable public/private evidence and state records.
- `config.py`: versioned settings and cross-field validation.
- `data.py`, `splits.py`: verified adapters, placement, and frozen cohort selection.
- `tokenization.py`, `prompts.py`, `parsing.py`: token identity, shared instructions, and strict output contracts.
- `evidence.py`, `packets.py`: validated persistent ledgers, bounded packets, and recipient visibility.
- `policies.py`: fixed/random/heuristic/complete/dense schedules and bounded metadata.
- `storage.py`, `backend.py`, `accounting.py`: immutable run identity, physical attempts, reservations, and logical references.
- `runner.py`: W1 frozen-state branching and shared initialization.
- `evaluators.py`, `readout.py`, `statistics.py`: official scoring, common aggregation, and paired analysis.
- `adaptive.py`, `scaling.py`, `ablations.py`: W2/W3/W4 orchestration, without implicit new inference.
- `learned.py`, `training.py`: optional shared encoder, pooled context, validation selection, and declared training cost.
- `finqa.py`: bounded program interpreter and table-operand provenance.
- `cli.py`, `__main__.py`, `reporting.py`, `README.md`: runnable local workflow, preparation artifacts, and reports.
- `tests/`: owning tests for the corresponding modules; `tests/fixtures.py` contains explicitly synthetic engineering inputs only.

Tasks 1–8 produce the fixed-call textual evidence experiment. Tasks 9–10 add adaptive/scaling/ablation support. Task 11 adds the optional learned policy. Task 12 completes numerical execution. Task 13 integrates CLI, real-input preparation, preservation verification, and release documentation. No step requires an empirical effect to be positive.

---

### Task 1: Public/private records and immutable configuration

**Files:** Create `WEBExtension/web_reasoning/{__init__,contracts,config}.py` and `tests/{__init__,fixtures,test_contracts}.py`.

**Interfaces:** Define frozen dataclasses: `Atom(atom_id, document_id, sentence_id, kind, text, title, context)`; `Source(document_id, title, atoms)`; `PublicExample(key, dataset, question, sources)`; `PrivateTargets(answer, aliases, supporting_locators, program, strata)`; `Example(public, targets)`; `Need(entity, relation)`; `Fact(atom_id, quote, claim)`; `Proposal(status, answer, facts, entity_tags, needs, derivation, conflicts, program, operand_sources)`; `Outcome(kind, proposal, raw_sha256, usage, reason)`; `AgentState(agent_id, visible_atoms, ledger, outcome)`. Collection fields are tuples; `context` and `strata` serialize through explicit allowlists. `PublicExample.key` is internal and never rendered. Status is one of supported/partial/no_evidence/conflict; outcome kind is valid/refusal/invalid/unresolved.

`RunConfig` contains schema/prompt version, experiment, seed, team sizes, regime, packet/control caps, control-frame ceiling, output cap, model/deployment/tokenizer identity, phase limits, positive completion/attempt/input/output ceilings, and mode. `validate_config(config)->RunConfig` rejects unknown fields and inconsistent values. `runtime_object(public)->dict` emits only question and public source data, never keys or private fields.

- [ ] **Step 1:** Add actual minimal publisher-shaped fixtures:

```python
def wiki_row():
    return {"_id": "fixture-1", "question": "Where was Mira born?",
            "context": [["Mira", ["Mira was born in Delta."]]],
            "answer": "Delta", "supporting_facts": [["Mira", 0]],
            "evidences": [["Mira", "birthplace", "Delta"]], "type": "inference"}

def finqa_row():
    return {"id": "fixture-report-1", "pre_text": ["Amounts are in millions."],
            "post_text": [], "table": [["", "2020", "2019"], ["Revenue", "12", "10"]],
            "qa": {"question": "What was the revenue increase?", "exe_ans": 2,
                   "program": "subtract(12, 10)", "gold_inds": {"table_1": "Revenue"}}}
```

- [ ] **Step 2:** Write failing contract tests; include an added `answer` key in a public mapping and an attempt ceiling smaller than the completion ceiling. Assert explicit `ValueError` rather than dropping unexpected fields. Run `python -m pytest WEBExtension/web_reasoning/tests/test_contracts.py -q`; expected failure: package/contracts absent.
- [ ] **Step 3:** Implement dataclasses, strict constructors, explicit serializers, and cross-field validation. Use this serialization rule at all runtime boundaries:

```python
def runtime_object(public):
    return {"question": public.question,
            "visible_sources": [source.to_public_dict() for source in public.sources]}
```

`Source.to_public_dict()` emits document_id/title and ordered atoms only; each atom emits its source locator, kind, text, and necessary unit/header context. No generic `vars(row)`/`asdict(Example)` enters prompts.

- [ ] **Step 4:** Add a test asserting source order is stable and `runtime_object` has neither `key` nor evaluation fields; rerun the owning tests.
- [ ] **Step 5:** Commit only the three modules, package initializers, fixture file, and contract tests: `git commit -m "feat: define isolated Web evidence contracts"` after explicitly staging those files.

### Task 2: Verified dataset adapters, placement, and cohort freeze

**Files:** Create `data.py`, `splits.py`, `tests/test_data.py`, `tests/test_splits.py` in the new package.

**Interfaces:** `load_verified(path: Path, expected_sha256: str, dataset: str, aliases: dict)->tuple[Example,...]`; `from_2wiki(row: dict, aliases: dict)->Example`; `from_finqa(row: dict)->Example`; `allocate(public, n_agents, regime, seed, token_count)->tuple[tuple[Atom,...],...]`; `freeze_splits(examples, sizes, seed)->dict[str,tuple[Example,...]]`; `balanced_sinks(keys, n_agents, seed)->dict[str,int]`.

- [ ] **Step 1:** Write tests using the Task 1 fixtures. Assert gold triples/programs exist only in `PrivateTargets`; input sources remain unchanged. Parameterize N=2,3,4,5, shared/partitioned placement, fewer documents than agents, duplicate titles with ambiguous support locators, and an out-of-range supporting sentence. All source atoms must retain stable locators.

```python
def test_gold_is_not_runtime_input():
    from WEBExtension.web_reasoning.contracts import runtime_object
    from WEBExtension.web_reasoning.data import from_2wiki
    from WEBExtension.web_reasoning.tests.fixtures import wiki_row
    public = runtime_object(from_2wiki(wiki_row(), aliases={}).public)
    assert set(public) == {"question", "visible_sources"}
    assert "evidences" not in public and "supporting_facts" not in public
```

- [ ] **Step 2:** Run both new test files; confirm missing adapters rather than an unrelated fixture/import problem.
- [ ] **Step 3:** Implement SHA-256 verification before JSON parsing; reject ambiguous original titles and invalid annotations before inference. Use source-local deterministic IDs independent of support labels. FinQA table atoms retain the original row and every associated header/unit note. Inspect and isolate publisher auxiliary fields using allowlists.

Allocation algorithm: seeded document shuffle, choose the lowest current token load with frozen seeded node-order ties, append whole documents, restore original source order within each shard. Shared placement copies the same ordered union. The sink allocator hash-sorts keys and cycles through a seed-permuted node list to obtain counts differing by at most one.

Split algorithm: normalize question strings, group identical questions and connected title/report identities, choose groups by deterministic hash with type quotas, and freeze counts/IDs/hashes. Reject duplicate question membership and record quota/group-size deviations. Do not silently break a source group to reach an exact sample count; require an explicit recorded relaxation before release.

The allocation implementation uses this deterministic recipient rule after the seeded node permutation:

```python
def least_loaded(loads, tie_order):
    rank = {node: i for i, node in enumerate(tie_order)}
    return min(range(len(loads)), key=lambda node: (loads[node], rank[node]))
```

- [ ] **Step 4:** Test hash mismatch, missing required alias input for alias-bearing release rows, disjoint split identities, grouped source separation, and order-independent split reproduction. Run `python -m pytest WEBExtension/web_reasoning/tests/test_data.py WEBExtension/web_reasoning/tests/test_splits.py -q`.
- [ ] **Step 5:** Stage the four new files and commit `feat: add verified Web dataset and allocation pipeline`.

### Task 3: Shared prompts, local token identity, and strict parsing

**Files:** Create `tokenization.py`, `prompts.py`, `parsing.py`, `tests/test_prompts.py`, `tests/test_parsing.py`, `tests/test_tokenization.py`; extend `tests/fixtures.py` with `proposal_body()` returning every required field.

**Interfaces:** `PinnedTokenizer(path, revision, expected_hash)` with `count(text)->int` and `.identity`; `UTF8FixtureTokenizer` with the same interface and identity kind `mock`. `build_messages(question, visible_atoms, own_state, incoming_packets, request, dataset, stage)->list[dict[str,str]]`; `parse_outcome(raw, visible_atoms, dataset)->Outcome`.

- [ ] **Step 1:** Add failing tests for unknown statuses, invalid status/answer pairs, non-JSON text, code fences, empty supported answers, invisible locators, wrong quotations, provider refusals, finish_reason=`length`, oversized lists, and conflicting fact IDs. Keep a raw provider body and usage reference for every invalid/refused result.

```python
def test_partial_facts_are_not_votes():
    from WEBExtension.web_reasoning.parsing import parse_outcome
    body = {"status": "partial", "answer": "Delta", "facts": [],
            "entity_tags": [], "needs": [], "derivation": [], "conflicts": []}
    import json
    raw = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(body)}}]}
    assert parse_outcome(raw, (), "2wiki").kind == "invalid"
```

- [ ] **Step 2:** Run the three test files and observe the missing implementation failures.
- [ ] **Step 3:** Extract the exact common/initial/synthesis/request/FinQA instructions from the approved prompt pack into versioned constants. Serialize user data canonically. Render an actual schema, not the descriptive example. Stage distinguishes extraction/synthesis/request; topology names and private condition labels never enter model text. Build visible sources from the caller's supplied atoms only.

Parsing validates exact types, duplicate fields, required keys, bounded claims, status coherence, source ownership, and exact full-atom quote equality. A syntactically valid partial proposal retains validated facts. Invalid outcomes retain their raw record but do not acquire a fabricated proposal. `noanswer`/empty/null supported answers are invalid; a valid unresolved proposal has answer null. Do not salvage invalid citations into a silently different scored prediction.

Load Hugging Face tokenizers with a pinned revision, verified backend hash, `local_files_only=True`, and `trust_remote_code=False`; never fall back to fixture counts outside mode MOCK. The fixture tokenizer counts UTF-8 bytes and explicitly declares that these are engineering units, not model tokens.

Use explicit schema/visibility checks before constructing a valid Outcome:

```python
def validate_answer_status(status, answer, facts):
    if status == "supported":
        if not isinstance(answer, str) or not answer.strip() or not facts:
            raise ValueError("supported_requires_answer_and_fact")
        if answer.strip().lower() == "noanswer":
            raise ValueError("abstention_is_not_supported")
    elif status in {"partial", "no_evidence", "conflict"}:
        if answer is not None:
            raise ValueError("unresolved_answer_must_be_null")
    else:
        raise ValueError("unknown_status")
```

- [ ] **Step 4:** Assert byte-identical prompt text for identical evidence/state across topology labels in internal metadata; test Unicode exact preservation and local tokenizer identity mismatch without network access. Rerun the three files.
- [ ] **Step 5:** Commit the explicit new modules, owning tests, and fixture update as `feat: add common evidence prompts and strict response parsing`.

### Task 4: Persistent fact ledger, atomic packets, and visibility

**Files:** Create `evidence.py`, `packets.py`, `tests/test_evidence.py`, `tests/test_packets.py`.

**Interfaces:** `merge_ledger(previous: tuple[Fact,...], incoming: tuple[Fact,...])->tuple[Fact,...]`; `Packet(sender, recipient, text, tokens, facts, omitted_atom_ids, unavailable, kind)`; `pack_evidence(state, recipient, request, cap, tokenizer, include_answer=True)->Packet`; `apply_delivery(receiver: AgentState, packet: Packet)->AgentState`; `pack_control(agent_id, payload, cap, tokenizer)->str`.

- [ ] **Step 1:** Write failing tests that preserve two contradictory sources while deduplicating the same source locator, and that transfer only the text actually included. Exercise tiny caps, oversized first atom, multibyte names, unit notes, and optional answer removal before evidence removal.

```python
def test_ledger_preserves_different_source_claims():
    from WEBExtension.web_reasoning.contracts import Fact
    from WEBExtension.web_reasoning.evidence import merge_ledger
    a = Fact("d0:0", "Mira was born in Delta.", "Birthplace is Delta")
    b = Fact("d1:0", "Mira was born in Epsilon.", "Birthplace is Epsilon")
    assert merge_ledger((a,), (a,b)) == (a,b)
```

- [ ] **Step 2:** Run both files and confirm absent evidence/packet functions.
- [ ] **Step 3:** Implement identity-keyed merge with exact-content consistency checks. Rank atoms by explicit request overlap, query overlap, then original ledger order. Build and tokenize the entire serialized candidate packet after each complete atom; include it only if the full message fits. Strip an optional answer suggestion before removing a source atom. If the minimal unavailable envelope itself exceeds the cap, raise `ValueError` before dispatch; otherwise emit the bounded unavailable envelope.

`apply_delivery` extends recipient visibility only with the packet's validated facts and exact text. No access to a global source registry is exposed to prompt builders. Headers/unit context are part of a table atom, never separate detachable fragments. Record transport kind `evidence`, `request`, `reply`, or `control` explicitly.

Use whole-message counting when considering every candidate atom, with the canonical envelope already populated:

```python
import json

def serialized_packet(envelope, atoms):
    return json.dumps({**envelope, "facts": atoms}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)

def fits(envelope, atoms, cap, tokenizer):
    return tokenizer.count(serialized_packet(envelope, atoms)) <= cap
```

- [ ] **Step 4:** Add a negative test attempting to cite a source hash without receiving its text; verify rejection by Task 3 parsing after delivery. Assert `packet.tokens == tokenizer.count(packet.text) <= cap` for every case. Rerun both files and parsing tests.
- [ ] **Step 5:** Commit these four files as `feat: preserve complete source atoms across sparse deliveries`.

### Task 5: Call-matched policies and charged observable metadata

**Files:** Create `policies.py`, `tests/test_policies.py`.

**Interfaces:** `Metadata(agent_id, status, valid_fact_count, entity_tags, needs, fact_hashes)`; `metadata_frame(state, tokenizer, cap=96)->tuple[Metadata,str]`; `select_senders(kind, metadata, sink, k, question, seed, already_delivered)->tuple[int,...]`; `Schedule(edges, recipients, control_frames)`; `one_round_schedule(kind, n_agents, sink, senders)->Schedule`.

- [ ] **Step 1:** Parameterize policy tests over N=2..5, sink positions, zero valid facts, duplicate tags, tied relevance, and permuted node indices. Test static/random/heuristic each choose exactly k distinct nonsink senders and one recipient, and complete In-Star/dense deliver identical ordered incoming edges to the sink.

```python
def test_singleton_update_schedules_are_call_matched():
    from WEBExtension.web_reasoning.policies import one_round_schedule
    for policy in ("static", "random", "heuristic"):
        schedule = one_round_schedule(policy, n_agents=3, sink=1, senders=(0,))
        assert schedule.edges == ((0,1),)
        assert schedule.recipients == (1,)
```

- [ ] **Step 2:** Run the owning tests; ensure the failure identifies absent policy code.
- [ ] **Step 3:** Implement static via frozen node permutation, random via seeded uniform sample without replacement, and the approved Jaccard heuristic. Normalize tags/needs/question with lowercase Unicode alphanumeric tokenization; union token sets before Jaccard, empty union gives zero. `has_valid_fact` is a Boolean; `already_delivered` indicates all of that sender's advertised validated facts have already reached the sink. Scores use only bounded parsed metadata.

Pack control metadata by dropping trailing tags/needs/hash entries until the entire frame fits; retain mandatory status/agent/count fields. Full-source text, candidate answers, and rationales are not valid Metadata fields. Every emitted frame is charged. Preserve a frozen tie permutation for static/random/heuristic comparisons; do not choose by source-support labels.

```python
def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left | right else 0.0

def heuristic_score(tags, needs, question, has_fact, already_delivered):
    return (2*jaccard(tags, needs) + jaccard(tags, question)
            + 0.5*bool(has_fact) - 0.5*bool(already_delivered))
```

- [ ] **Step 4:** Add a test rejecting source text or correctness in router metadata. Check deterministic replay, fixed sink/recipient equality, no diagonal edges, no N×N allocation in heuristic scoring, and six-frame budget enforcement before dispatch. Rerun policy tests.
- [ ] **Step 5:** Commit policy module/tests as `feat: add matched sparse routing schedules and metadata costs`.

### Task 6: Durable request identity, budgets, and physical/logical accounting

**Files:** Create `storage.py`, `backend.py`, `accounting.py`, `tests/test_storage.py`, `tests/test_backend.py`, `tests/test_accounting.py`.

**Interfaces:** `canonical(obj)->str`; `digest(obj)->str`; `atomic_json(path,obj)->None`; `Journal(path).append(event)`; `Request(messages, model_identity, tokenizer_identity, decoding)`; `Backend.complete(request, logical_slot)->Outcome`; `DurableBackend(directory, config, transport, tokenizer)`; `BudgetExceeded`, `UncertainRequest`, `ProviderRejected`; `cost_summary(events, logical_refs)->dict`.

The injected transport signature is `transport(request: dict, attempt_id: str)->tuple[int,dict,dict]`. Constructors/imports never construct an SDK client. Request identity hashes canonical model-visible parameters, deployment, tokenizer, prompt/schema version, and stochastic replicate ID; arm labels are separate logical references. A semantic schema change creates a different key even if prompt bytes accidentally match.

- [ ] **Step 1:** Write backend tests with an in-memory transport that records invocations. Verify two logical slots with an identical request produce one physical completion; changed model/tokenizer/replicate produces a new request; corrupted cache hashes and concurrent directory locks fail explicitly.

```python
def test_unknown_attempt_blocks_restart(tmp_path):
    from WEBExtension.web_reasoning.storage import Journal
    from WEBExtension.web_reasoning.backend import DurableBackend, UncertainRequest
    from WEBExtension.web_reasoning.tests.fixtures import backend_config
    import pytest
    Journal(tmp_path / "calls.jsonl").append(
        {"event":"started", "attempt_id":"a1", "key":"k1", "request_sha256":"r1"})
    with pytest.raises(UncertainRequest):
        DurableBackend(tmp_path, backend_config(), transport=None, tokenizer=None)
```

Add `backend_config()` in fixtures here, returning a complete mode-MOCK `RunConfig` with finite positive ceilings, mock model/tokenizer identity, and zero automatic SDK retries. It is not a real-run authorization.

- [ ] **Step 2:** Run the three new files and observe expected absent-module failures.
- [ ] **Step 3:** Implement file/directory-fsynced atomic writes and a serial file lock. Freeze configuration before any request; verify it on restart. State order is request save → reserve budgets → fsynced `started` → raw response/body save → fsynced terminal event → cache. Recover a saved raw response after a crash; an unmatched started attempt without recoverable evidence blocks the run.

Reserve pinned input count plus framing allowance and maximum output before dispatch. Track logical completions, started attempts, known tokens, missing usage, cache references, refusals, malformed successes, and uncertain liabilities separately. A successful malformed response consumes a completion and is never automatically rerun. Only a clearly rejected ungenerated 429 may retry, bounded by config and remaining attempts; unknown/timeout does not.

Request identity deliberately excludes internal arm labels:

```python
def request_key(request):
    return digest({"messages": request.messages,
                   "model": request.model_identity,
                   "tokenizer": request.tokenizer_identity,
                   "decoding": request.decoding})
```

The decoding identity includes prompt/schema versions and stochastic replicate ID as specified above. `digest` is the Task 6 SHA-256 of canonical UTF-8 JSON. A unique immutable request file still carries the separate logical slot provenance.

- [ ] **Step 4:** Fault-inject at each storage boundary; test cache recovery without a second transport call, rejected retry ceilings, incomplete/negative usage, and budget stop before dispatch. Account for `usage=None` as unknown rather than zero. Rerun all three files.
- [ ] **Step 5:** Commit the six files plus the explicit fixture update as `feat: add durable isolated attempt and resource accounting`.

### Task 7: Frozen-state W1 experiment runner

**Files:** Create `runner.py`, `tests/test_runner.py`.

**Interfaces:** `initialize(public, assignments, backend, config)->tuple[AgentState,...]`; `run_w1(public, assignments, sink, backend, config)->QuestionRun`; `QuestionRun(question_key, dataset, n_agents, initial, arms, logical_refs, transmissions, mode)`; `ArmResult(name, states, sink, outcome, references, transmissions)`. Private targets are not runner arguments. `QuestionRun.mode` propagates MOCK or REAL unchanged into every exported result. Add test helper `w1_case(n_agents, directory)` in `tests/fixtures.py`, returning `(public, assignments, sink, backend, config)` from an explicitly synthetic six-document fixture, Task 2 allocation, Task 6 mock backend, and deterministic Task 3-valid response bodies. It must not read benchmark files or credentials.

- [ ] **Step 1:** Write a mock-transport fixture with genuinely different source visibility per agent and deterministic valid partial/supported outputs. Verify a shared initialization is generated once, the private update has no peer source, and the complete-In-Star/dense sink messages are identical.

```python
def test_w1_records_actual_logical_slots(tmp_path):
    from WEBExtension.web_reasoning.runner import run_w1
    from WEBExtension.web_reasoning.tests.fixtures import w1_case
    for n in (2,3,4,5):
        public, assignments, sink, backend, config = w1_case(n, tmp_path / str(n))
        result = run_w1(public, assignments, sink, backend, config)
        assert len(result.logical_refs) == 2*n + 6
```

Extend this integration test to compare stages/recipients and confirm dense still owns N update references even when its sink physically hits the complete-In-Star cache. Commit the fixture update with this task.

- [ ] **Step 2:** Run `python -m pytest WEBExtension/web_reasoning/tests/test_runner.py -q`; confirm runner absence rather than transport/parser errors.
- [ ] **Step 3:** Implement the eight named W1 arms. Fork immutable state for each arm; use one packet version per sender/frozen state and canonical sender order. Full-union uses one call with the same schema/task and original source union. NoComm adds no call. Private/static/random/selected/full-In-Star add one sink update each; dense adds N update references. Carry all nonupdated agent endpoints unchanged for common offline voting.

The new runner's recipient loop must aggregate incoming senders before updating each recipient once:

```python
def incoming_by_recipient(edges):
    recipients = sorted({recipient for _, recipient in edges})
    return {recipient: tuple(sorted(sender for sender, target in edges if target == recipient))
            for recipient in recipients}
```

Construct every arm from the same initial immutable tuple; never feed one arm's revised endpoint into another arm's update. Store the original tuple hash and assert it remains unchanged after all arms.
- [ ] **Step 4:** Add tests that shuffled arm execution order preserves prompt hashes and that an invalid initial outcome remains visible without invented evidence. Verify the actual `2N+6` formula, cache-versus-logical counts, visibility, and union equality at N=2..5. Rerun runner/policy/backend tests.
- [ ] **Step 5:** Commit runner/tests as `feat: implement fixed-state Web routing comparisons`.

### Task 8: Official evaluation adapters, common readout, and paired statistics

**Files:** Create `evaluators.py`, `readout.py`, `statistics.py`, `references/sources.json`, `tests/test_evaluators.py`, `tests/test_readout.py`, `tests/test_statistics.py`.

**Interfaces:** `abstention_vote(states, node_order)->str`; `sink_answer(arm)->str|None`; `OfficialEvaluator(dataset, source_manifest, alias_file).score(public_key, outcome, targets)->dict`; `paired_summary(question_rows, contrast, draws=10000, seed=20261016)->dict`; `exact_mcnemar(wins,losses)->float`; `identification_bounds(question_rows)->dict`.

- [ ] **Step 1:** Write tests for a concrete answer surviving two valid null abstentions, deterministic substantive ties, all-null states, refusal/invalid versus abstention, missing paired cells, and a noninferiority interval crossing -2. Use whole questions as resampling rows.

```python
def test_exact_paired_em_test():
    from WEBExtension.web_reasoning.statistics import exact_mcnemar
    assert exact_mcnemar(0,0) == 1.0
    assert exact_mcnemar(5,0) == 0.0625
```

- [ ] **Step 2:** Run the three files and confirm missing scoring modules.
- [ ] **Step 3:** Pin official source commits and hashes from the publisher repositories linked in the audit. The source manifest must record upstream URLs, commit, license, raw hashes, and any mechanical local adapter change. Download only these requested public sources during execution, with required network escalation if blocked; never substitute a remembered evaluator. The current environment lacks `ujson`; resolve that exact dependency before requiring upstream parity tests, or report the unavailable dependency explicitly. No silent skipped parity check may support an official-score claim.

Alias-aware 2Wiki answer EM/F1 uses the compatible release alias file, held privately. Convert visible sentence citations back to original titles/indices for support metrics. Do not report official joint evidence scores unless predicted reasoning triples are explicitly present and converted. FinQA official scoring consumes predicted program tokens, not a free-form numeric answer.

Readout excludes only valid unresolved proposals and explicit provider abstentions/refusals from substantive voting; any invalid/unresolved request makes the scientific cell incomplete, rather than silently disappearing. Implement exact McNemar with a two-sided binomial tail. Bootstrap paired question blocks; return effect/CI/wins/losses/ties plus unresolved bounds. Report the -2 margin only when a prespecified one-sided procedure and sample justify it.

```python
import math

def exact_mcnemar(wins, losses):
    if type(wins) is not int or type(losses) is not int or min(wins, losses) < 0:
        raise ValueError("nonnegative_integer_discordances_required")
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2*tail)
```

- [ ] **Step 4:** Compare wrapper scores against upstream evaluator output on fixed fixtures for aliases, Boolean special cases, punctuation, missing predictions, and incorrect support indices. Test that MOCK rows are rejected from scientific summaries even if their answers match. Rerun readout/statistics/parity tests.
- [ ] **Step 5:** Commit explicit scorer/readout/statistics/reference files and tests as `feat: add official scoring and paired Web analysis`.

### Task 9: Bounded gather/request/reply consultation

**Files:** Create `adaptive.py`, `tests/test_adaptive.py`.

**Interfaces:** `should_stop(state: AgentState)->bool`; `run_adaptive(public, initial, sink, backend, config, policy)->ArmResult`; `validate_transmission_budget(transmissions, n_agents, packet_cap, metadata_cap, max_control_frames=6)->None`.

- [ ] **Step 1:** Test partial evidence, all-null initial proposals, explicit conflicts, a supported sink with unresolved needs, supported complete sink, empty/oversized requests, no remaining budget, and a leaf that cannot satisfy a request. Supported status alone must not halt.

```python
def test_stop_requires_all_observable_conditions():
    from WEBExtension.web_reasoning.adaptive import stop_fields
    assert not stop_fields("partial", None, 2, 0, 0)
    assert not stop_fields("supported", "Delta", 1, 1, 0)
    assert not stop_fields("supported", "Delta", 1, 0, 1)
    assert stop_fields("supported", "Delta", 1, 0, 0)
```

Define the public pure helper `stop_fields(status, answer, valid_fact_count, needs_count, conflicts_count)->bool`; `should_stop` delegates after validating the Outcome.

- [ ] **Step 2:** Run the owning file and verify missing adaptive functions.
- [ ] **Step 3:** Implement gather at k senders, then at most one request/reply per selected leaf. Request content comes from the preceding sink proposal; replies get one local specialist update. Preserve evidence ledger across every phase. Reject schedules that exceed `3(N-1)` data/request deliveries or six metadata frames per agent before performing the next backend call. Record `supported_stop`, `budget_stop`, `phase_limit`, `unresolved`, and `conflict` separately.

```python
def stop_fields(status, answer, valid_fact_count, needs_count, conflicts_count):
    return (status == "supported" and isinstance(answer, str) and bool(answer.strip())
            and valid_fact_count > 0 and needs_count == 0 and conflicts_count == 0)
```

`should_stop` first requires outcome.kind=`valid`; it cannot reuse a supported proposal from a previous failed update without explicitly recording that endpoint/state choice.
- [ ] **Step 4:** Enumerate N=2..5 and every leaf request subset; assert delivery bounds, actual token caps, recipient-call counts, and no hidden planner call. Compare always-run and stopping schedules using mock states, preserving losses as well as savings. Add the prespecified private-update schedule with the same maximum update budget and no peer packets. Rerun adaptive/runner tests.
- [ ] **Step 5:** Commit adaptive/tests as `feat: add bounded evidence-demand consultation`.

### Task 10: Scaling and controlled development ablations

**Files:** Create `scaling.py`, `ablations.py`, `tests/test_scaling.py`, `tests/test_ablations.py`.

**Interfaces:** `run_scaling(public_examples, nested_keys, backend_factory, config)->tuple[QuestionRun,...]`; `ablation_variants(base_config)->tuple[RunConfig,...]`; `compile_variant(base_state, variant, config)->dict`.

- [ ] **Step 1:** Test same-source-union placement at N=2..5, exact nested N=3 reuse only with identical manifests/requests, and rejection of a legacy estimated N=3 plotting value as an empirical record.
- [ ] **Step 2:** Run the two files and confirm absent orchestrators.
- [ ] **Step 3:** Orchestrate existing W1/W2 functions without modifying their scientific contracts. Record both theoretical dense edges and actual transmissions/calls/control/payload/input/output/latency. Count shared physical artifacts once but retain per-arm isolated costs. No curve point may be synthesized from a different cohort.

Define development-only variants: common legacy/new prompt crossed with selected/dense; whole-atom versus prefix cap; answer suggestion present/absent; always-run/stopping; natural/shared/oracle placement. Oracle placement explicitly consults private support labels offline and exports only the resulting public assignment. It is not part of the natural-population primary estimate. Freeze which variants are run before held-out outcomes; do not automatically launch the factorial.

Each scaling record includes this separate theoretical reference, never substituted for measured events:

```python
def theoretical_reference(n, rounds):
    if n not in (2,3,4,5) or rounds not in (1,2,3):
        raise ValueError("unsupported_team_or_depth")
    return {"dense_deliveries": rounds*n*(n-1),
            "sparse_three_phase_ceiling": 3*(n-1),
            "reference_only": True}
```

- [ ] **Step 4:** Add tests preventing a single arm from receiving a different task-prompt version in the primary comparison. Verify prefix treatment is labeled as a content ablation, oracle metadata never enters prompts, control traffic remains counted, and N=2 has no claimed nontrivial sender-selection effect. Rerun both files.
- [ ] **Step 5:** Commit these modules/tests as `feat: add paired team-size and prompt ablation orchestration`.

### Task 11: Optional shared encoder and train/validation-only selection

**Files:** Create `learned.py`, `training.py`, `tests/test_learned.py`, `tests/test_training.py`.

**Interfaces:** `PooledSenderScorer(feature_dim=32, hidden_dim=64)` with `forward(features: Tensor, sink_features: Tensor)->Tensor[N-1]`; `sender_features(metadata, question, delivered)->array[N-1,32]`; `train_router(training_records, validation_records, settings)->Checkpoint`; `choose_checkpoint(validation_scores, tie_rule)->Checkpoint`. No gold argument exists in serving methods.

- [ ] **Step 1:** Write permutation-equivariance, N=2..5 serving, fixed parameter-count, and no N×N state tests. Use generated engineering tensors for these tests, never label them benchmark results.

```python
def test_scorer_is_equivariant():
    import torch
    from WEBExtension.web_reasoning.learned import PooledSenderScorer
    torch.manual_seed(7)
    model = PooledSenderScorer(32,64).eval()
    x, sink = torch.randn(4,32), torch.randn(32)
    p = torch.tensor([2,0,3,1])
    assert torch.allclose(model(x[p],sink), model(x,sink)[p], atol=1e-6)
```

- [ ] **Step 2:** Run the two files and confirm the learned extension is absent.
- [ ] **Step 3:** Implement a shared per-sender MLP, mean-pooled team context, sink embedding, and shared scalar scorer. Features contain bounded hashed lexical tags, status one-hot, valid-fact count, request overlap, and delivered fraction; dimensions and hash seed are versioned. No source text, test correctness, full adjacency, or N-specific weight tiling. Use an O(N) selection algorithm if reporting linear selection cost; otherwise explicitly report the implemented O(N log N) sort.

```python
import torch
from torch import nn

class PooledSenderScorer(nn.Module):
    def __init__(self, feature_dim=32, hidden_dim=64):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.score = nn.Sequential(nn.Linear(3*hidden_dim, hidden_dim), nn.Tanh(),
                                   nn.Linear(hidden_dim, 1))

    def forward(self, features, sink_features):
        encoded = self.encode(features)
        pooled = encoded.mean(dim=0, keepdim=True).expand_as(encoded)
        sink = self.encode(sink_features).expand_as(encoded)
        return self.score(torch.cat((encoded, pooled, sink), dim=1)).squeeze(-1)
```

Fix the 32 feature columns: 0–3 status one-hot; 4 `log1p(valid_fact_count)/log(5)` clipped to [0,1]; 5 request/entity Jaccard; 6 query/entity Jaccard; 7 delivered-fact fraction; 8–15 signed entity-tag hash bins; 16–23 signed need-token hash bins; 24–31 sink entity-tag hash bins. Hash with SHA-256 and the manifest seed; normalize each eight-bin block by its nonzero input-token count. Use the same feature contract for sink encoding. Concatenate sender, mean-team, and sink hidden vectors before shared scalar scoring; neither position nor gold labels enter serving.

The first learned training mode uses complete, matched one-sender outcomes on N=3 training questions: both possible sender updates from the same frozen state must be present. Fit predicted utility by mean-squared error to observed official training EM minus the manifest's required `lambda_input_tokens`, `lambda_output_tokens`, and `lambda_calls` penalties, with declared positive normalizers. Reject incomplete counterfactual pairs; do not fill unobserved rewards with model guesses or generate missing outcomes automatically. Require optimizer, learning rate, epoch count, and seeds in settings. Choose one checkpoint by mean validation utility, with smallest checkpoint hash breaking exact ties. This is supervised utility regression, not a claim of PPO replication. Mixed-N training is a separate declared mode.

Training records contain only authorized training prompt states and measured outcomes/costs. Record every training generation or imported cache provenance and all rejected/uncertain attempts. Do not train on the 200/50 evaluation identities. No training inference is launched by this coding task.

- [ ] **Step 4:** Test that putting an evaluation key in training raises, a changed feature/prompt version rejects the checkpoint, validation ties have a fixed rule, and training costs remain in a separate included ledger. Save/reload a tiny local tensor-only fixture checkpoint and reproduce serving scores. Rerun both files.
- [ ] **Step 5:** Commit explicit modules/tests as `feat: add optional agent-count-independent sender scorer`.

### Task 12: FinQA executable programs and operand provenance

**Files:** Create `finqa.py`, `tests/test_finqa.py`; extend `data.py` and `evaluators.py` only where their already-defined interfaces require numerical specialization.

**Interfaces:** `execute_program(tokens, visible_table_atoms, max_steps=32)->ProgramResult`; `validate_operands(program, operand_sources, visible_atoms)->None`; `ProgramResult(valid, value, reason, steps)`; `finqa_prediction(proposal)->dict` producing the official ID/program prediction format.

- [ ] **Step 1:** Write tests for comma-separated values, parenthesized negatives, ratios versus percentages, year/column ambiguity, missing unit notes, division by zero, invalid prior-step references, oversized exponent/step budgets, and an operand cited only by a hidden source. A direct numeric answer with no program must not pass official execution accuracy.

```python
def test_arithmetic_is_allowlisted():
    from WEBExtension.web_reasoning.finqa import execute_program
    result = execute_program(["__import__(", "os", "0", ")", "EOF"], (), max_steps=32)
    assert not result.valid
    assert result.reason == "unsupported_operation"
```

- [ ] **Step 2:** Run `python -m pytest WEBExtension/web_reasoning/tests/test_finqa.py -q`; confirm absent interpreter.
- [ ] **Step 3:** Implement only the pinned official operation/constant/reference syntax. Use typed operands, no Python `eval`/`exec`; bound steps and numerical magnitude. Validate row identifiers and unit context against visible table atoms. Keep program normalization consistent with the official parser; do not invent conversion shortcuts. The final scored program and raw output are both retained.

The arithmetic dispatch starts from explicit callables, with table aggregation, bounded exponent, and comparison implemented as separately validated handlers matching the pinned evaluator:

```python
import operator

BINARY_ARITHMETIC = {"add": operator.add, "subtract": operator.sub,
                     "multiply": operator.mul, "divide": operator.truediv}

def arithmetic(operation, left, right):
    if operation not in BINARY_ARITHMETIC:
        raise ValueError("unsupported_operation")
    if operation == "divide" and right == 0:
        raise ValueError("division_by_zero")
    return BINARY_ARITHMETIC[operation](left, right)
```
- [ ] **Step 4:** Compare arithmetic/program outputs with the pinned upstream FinQA evaluator on valid, malformed, table-aggregation, and percent cases. The same tool is available to every arm. Verify that incomplete agents can transmit operands while returning a null answer and empty program. Rerun FinQA/data/scoring tests.
- [ ] **Step 5:** Commit the explicit numerical files and targeted adapter edits as `feat: support source-grounded FinQA program execution`.

### Task 13: CLI, release preparation, reports, and preservation audit

**Files:** Create `cli.py`, `__main__.py`, `reporting.py`, `README.md`, `requirements.txt`, `tests/test_cli.py`, `tests/test_reporting.py`; update the proposal README with the implementation link after tests actually pass.

**Interfaces:** `main(argv: list[str]|None)->int`; `prepare_run(config_path, sources, output_dir)->Path`; `write_report(question_runs, targets, output_dir, scientific=False)->Path`; `verify_run(run_dir)->dict`.

- [ ] **Step 1:** Add CLI tests with subprocess network access blocked. `--help`, `prepare`, `run --mode mock`, `verify`, `report --engineering`, and `forecast` must work without credentials. `report --scientific` rejects mock provenance; `run --mode live` without a reviewed run permit stops before constructing a client.
- [ ] **Step 2:** Run the two files and confirm missing CLI/reporting code.
- [ ] **Step 3:** Implement these commands:

```text
python -m WEBExtension.web_reasoning prepare --config PATH --sources PATH --output PATH
python -m WEBExtension.web_reasoning forecast --manifest PATH
python -m WEBExtension.web_reasoning run --manifest PATH --mode mock
python -m WEBExtension.web_reasoning verify --run PATH
python -m WEBExtension.web_reasoning report --run PATH --engineering
```

The CLI accepts real local verified data for preparation and synthetic fixtures only for explicitly MOCK engineering execution. `prepare` writes hashes, public/private partitions, prompt/model/tokenizer identity, schedule ceilings, and budget estimates; it makes no inference. `forecast` distinguishes logical calls from unique hashes, theoretical bounds from observed usage, and known lower bounds from missing usage. The live backend factory is lazy and outside normal imports; it requires an exact permit matching manifest hash, finite call/attempt/token ceilings, and separately supplied credentials. The coding/validation workflow never issues a live command or creates that spend permit.

Check the reviewed permit before any SDK import or credential access; a matching file is an operational guard, not evidence that human authorization exists by itself:

```python
def require_live_permit(manifest_hash, permit):
    required = ("max_completions", "max_attempts", "max_input_tokens", "max_output_tokens")
    if permit is None or permit.get("manifest_sha256") != manifest_hash:
        raise PermissionError("reviewed_run_permit_required")
    if any(type(permit.get(k)) is not int or permit[k] <= 0 for k in required):
        raise PermissionError("finite_positive_live_ceilings_required")
```

Reports show every arm under sink and common vote, matched comparisons, visibility/packet diagnostics, physical union versus per-arm logical cost, training overhead, errors/unknowns, per-N paired intervals, and exact source hashes. No scientific chart contains mock or estimated legacy curve points. If upstream evaluation, tokenizer, or source validation is unavailable, show the specific incomplete status rather than an official score. Preserve raw provider output and request identity links.

Use an isolated minimal requirements file for direct dependencies, with versions compatible with the inspected environment; do not rewrite repository-wide requirements. Add an example mode-MOCK configuration with explicit finite ceilings and synthetic-source provenance. Document how to prepare a real pilot manifest, then stop for review of its actual spend limits. The first live study remains the separately authorized 20-question pilot, not a full automatic sequence.

- [ ] **Step 4:** Run complete local verification:

```bash
python -m pytest WEBExtension/web_reasoning/tests -q
python -m compileall -q WEBExtension/web_reasoning
python -m WEBExtension.web_reasoning --help
python -m pytest WEBExtension/tests/test_protocol.py -q
```

Run mock W1 at every N=2..5 and an adaptive mock case; rerun the same manifests and assert zero additional transport invocations. Verify original design source-manifest hashes and independent E5 counts from the original source checkout. If an isolated worktree lacks preserved ignored artifacts, perform that read-only historical check in the source checkout and record the distinction. Inspect `git status` and the exact diff; no unrelated edits may enter commits.

- [ ] **Step 5:** Perform the selected execution workflow's final fresh review. Address findings with new focused tests where material. Commit explicit CLI/report/docs/tests files as `feat: expose auditable Web experiment preparation and reports`. Report local validation, incomplete external dependencies, release status, and absence of paid runs accurately. Do not claim Web accuracy or linear empirical scaling from these checks.

## Spec-to-task coverage and release boundaries

- Data provenance, public/private separation, source placement, cohort grouping: Tasks 1–2.
- Improved shared prompts, partial evidence, statuses/refusals, token identity: Tasks 3–4.
- Routing and readout fairness, call-matched W1, full-union single-agent baseline: Tasks 5, 7–8.
- Durable attempts, missing usage, cache reuse, all cost classes: Task 6 and final reports.
- Adaptive consultation/stopping and transmission/control bounds: Task 9.
- Common-cohort scaling, prompt/content/readout/oracle ablations: Task 10.
- Optional learned policy, arbitrary-N serving, train/validation selection: Task 11.
- Numerical evidence, program safety, official execution metrics: Task 12.
- Official parity, paired inference, missing-cell bounds: Task 8, completed for FinQA in Task 12.
- Workflow, forecast, manifests, all acceptance checks and preservation: Task 13.
- Paid execution, human semantic adjudication, confirmatory power certification, and paper claims are separate empirical/review actions, not implied by software completion.

## Plan review and recommended execution

Recommended method: **Native**. The tasks share contracts, parser visibility, backend identity, and accounting interfaces; implementing them sequentially in one session reduces interface handoff errors and repeated context. A fresh whole-change review follows the selected execution workflow. Subagent-driven execution is an alternative if the user prefers per-task independent implementation/review overhead.

Before implementation: the user reviews this saved plan and chooses Native or Subagent-driven. After that choice, use the corresponding execution skill, establish isolation, and proceed without asking again for ordinary reversible coding decisions. No new paid calls are authorized by plan approval.
