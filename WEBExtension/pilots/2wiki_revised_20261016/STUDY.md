# Revised 2Wiki partitioned evaluation: frozen gate

The user explicitly authorized this seven-arm N=3 evaluation, a 20-question pilot first, and expansion to 100 questions only if the pilot gate passes. This is a bounded extension of the existing harness. No unrelated benchmark or post-outcome prompt tuning is authorized.

The original twenty publisher training questions and whole-article assignments are retained, including the old pilot's unfinished question. All annotated gold supporting facts are in the union. Gold is not used to assign or select senders. The eighty potential expansion questions are sampled before new inference by seeded, type-stratified hash ranking from source-valid publisher training rows; they share no normalized question or article title with the first twenty or with each other. This does not establish independence from all older research artifacts. The run is exploratory, not a held-out test claim.

Common prompt: evidence-only reasoning with a relaxed three-field JSON object (`facts`, `missing_entities`, `answer`). No list-size ceiling or source-locator verification rejects an otherwise usable answer. Source/fact accuracy is a separate scientific question. Fenced JSON is accepted; missing or non-list metadata is retained with diagnostics. Missing/invalid answer types, malformed JSON, refusals and truncated outputs remain failures. Explicit null/noanswer is a valid abstention. No output repair call is made.

All seven arms use the same initial prompt, task contract and 512-token complete-item evidence packet cap. Full union makes one call over all sources. NoComm reuses the three initial responses. Random and heuristic each update Sink 1 with one sender. The complementary router selects k=0,1,2 using the sink's unresolved entity names and public source-title metadata. It greedily covers distinct needs, retains the initial sink if it reports a concrete answer with no unresolved needs, and conservatively takes both senders for unresolved needs that cannot be matched. In-Star updates Sink 1 from senders 0 and 2. Dense updates all three agents from the frozen initial messages. Sender order is canonical; In-Star and Dense share the exact Sink-1 request. Metadata transmission costs are recorded separately from evidence packets and model tokens.

Both fixed Sink-1 and abstention-aware agent-order-tie voting are applied to every arm. The single full-union answer is identical under both readouts. Official publisher alias-aware EM/F1 is used for answer scoring; failures count as zero operational success and are never dropped. No supporting-fact or semantic-faithfulness score is claimed.

Gate, frozen before inference:

1. All twenty question records complete; no unresolved provider attempt.
2. Fewer than 2% of unique physical returned responses are invalid, refused, or truncated. These combined operational contract failures are stricter than malformed-schema failures alone.
3. NoComm EM is at most 15% under **both** readouts (at most 3/20 correct). Questions and prompts will not be changed to force that result.
4. In-Star and the complementary router each strictly exceed NoComm, random one-sender and heuristic one-sender in **both EM and mean F1 under both readouts**.
5. Neither In-Star nor complementary has lower Sink-1 EM or mean F1 than Dense. In-Star must additionally share every Dense Sink-1 request key. Question-level disagreements are preserved.

These are operational pilot conditions, not significance or noninferiority tests. A failure stops the study. If they all pass, the same frozen implementation evaluates the remaining eighty without tuning. Report those eighty separately; pooled 100 is conditional on pilot selection.

Maximum new pilot spend: 200 completions, 220 attempts, 2,000,000 input tokens and 820,000 output tokens. Maximum through conditional expansion: 1,000 completions, 1,050 attempts, 10,000,000 input tokens and 4,100,000 output tokens. Every call has a 4,096-token output ceiling. These are upper limits, not promised usage or currency prices. SDK retries are disabled; only explicitly rejected 429 responses can be retried within the persistent attempt budget. An uncertain transport status stops execution without automatic retry.

The existing Azure gpt-oss-120b deployment alias and pinned cached tokenizer are retained. Physical cache keys include exact messages, serving identity, decoding parameters, tokenizer identity and a new internal replicate namespace. No arm labels enter model-visible prompts. Changing the prompt invalidates reuse of the original strict-schema generations; the old unresolved attempt and ledger remain unchanged. Reuse occurs within the revised common contract, including Round 0, coincident one-sender requests, and the In-Star/Dense sink.

Implementation and consumed inputs are hashed before credential access. Raw requests/responses, attempt ledger, physical usage, complete question records and gate decision are saved under `WEBExtension/web_reasoning/artifacts/2wiki_revised_20261016`. No credentials are saved. Publisher source archive SHA256 remains `95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef`.

Commands from repository root:

```bash
python -m pytest -q WEBExtension/pilots/2wiki_revised_20261016/test_revised.py WEBExtension/web_reasoning/tests
python -m WEBExtension.pilots.2wiki_revised_20261016.prepare
python -u -m WEBExtension.pilots.2wiki_revised_20261016.run --execute
```

The final command performs the pilot and only conditionally expands. Never reset an uncertain ledger, edit the frozen inputs, or change the gate after observing outcomes.
