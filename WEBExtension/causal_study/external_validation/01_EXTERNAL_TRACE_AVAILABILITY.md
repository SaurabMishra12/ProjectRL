# External trace availability

Audit date: 16 October 2026. **No suitable independent Web-QA trace set preserving the required abstention semantics and reconstructable communication comparison was established in this bounded search.** An independent numeric-QA pipeline is partially usable, and is analyzed rather than discarded because it does not fit the proposed narrative.

The five eligibility requirements were checked separately: initial independent proposals, retained abstention/refusal/missing categories, individual outputs, reconstructable final readout, and accepted labels. A release of code or input data is not a release of generated traces. An output containing parser `None` is not automatically a model expressing uncertainty.

## Local evidence

The direct local inventory inspected 18 files in the explicit scope recorded in [results/local_inventory.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/results/local_inventory.json). It includes E1–E5, corrected E3, old W1–W4 predictions and older task summaries. Existing source-transition work covers E3-v 2. The E5 states give strong within-pipeline reconstruction, but E1–E5 reuse the same 300 identities; summaries from other local tasks have different schemas, missing required output distinctions, or unestablished independent generation provenance. They are not independent evidence for the source-bound abstention interaction. The separate new E5 implementation reproduces all 60 preserved first-three-readout counts; this verifies computation, not population generality.

Graph evidence was Tier 2, project `projectRLMAIN`, generation `2026-10-13T17:28:45Z`. Targeted WEBExtension aggregation lookup returned no graph matches; exact coverage was `not_tracked`. Direct source inspection and bounded file inventory therefore supply the evidence. A clean coverage flag is not a completeness guarantee.

## Public releases inspected

- **Debate or Vote**, commit `82c929ea773d534cfb3fb0ddc4b7d14d245ab549`. 38-entry tree; code and evaluator, no saved per-agent evaluation files identified in this pinned tree. **NO compatible traces found in inspected tree**. [Public repository](https://github.com/deeplearning-wisc/debate-or-vote).

- **When Debate Helps**, commit `899b2834acd13f7b3298eb88233d39ab419d7a15`. 49-entry tree; construction/test examples are dataset examples, not generated multi-agent response histories. **NO compatible traces found in inspected tree**. [Public repository](https://github.com/Wang-ML-Lab/when-debate-helps).

- **Free-MAD linked software**, commit `4c62ac3c79b6420ace5d671f7db6de47202afa71`. 326-entry current tree is a broader evolving software project; inspected regular JSON files are configurations; no paper-result trace archive identified. **UNAVAILABLE in inspected tree; current software is not itself paper-run provenance**. [Public repository](https://github.com/jonathansantilli/freemad).

- **MADAM-RAG / RAMDocs**, commit `d44454c9ebb0bf513d8236a03decc9fb58704cad`. 5 blobs: data, code, README, requirements and license. RAMDocs_test contains documents/labels, not saved agent completions. **NO compatible saved traces in inspected tree**. [Public repository](https://github.com/HanNight/RAMDocs).

- **Don't Hallucinate, Abstain**, commit `2f2b73d5b9e035b6935c55b1ef06713b4c5de781`. 30-entry tree contains input datasets and approach scripts; no published agent-run output histories identified. **NO compatible traces found in inspected tree**. [Public repository](https://github.com/BunsenFeng/AbstainQA).

- **MuSiQue**, commit `922ac98f19a201998dbdae6d7f2887a5258dbdeb`. 94-entry code tree; README links official four-model predictions externally; those are predictions, not communication histories. Linked prediction archive was not downloaded. **NOT a communication replication; full external prediction archive not audited**. [Public repository](https://github.com/stonybrooknlp/musique).

- **Public exploratory GSM8K log**, commit `b0ad106f5ba26ad1646e3bff875f9c911864aec6`. 212-entry tree; prespecified all eight regime model directories, seed 1,N 3,R 2; raw text and parsed predictions across three rounds. **PARTIAL compatibility: reconstructable numeric votes/labels; no separate semantic abstention or provider-refusal field, no matched three-seat private control**. [Public repository](https://github.com/moozwen/sakana-debate).

The actual trees, relevant code and acquisition receipts are retained in [sources/](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/sources); [results/public_blob_verification.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/results/public_blob_verification.json) verifies downloaded Git blob hashes against pinned trees. GitHub/Hugging Face metadata retrieval is public HTTP activity; **new paid inference calls=0**. Repository trees were complete for these commits, but linked storage, other branches, private data and all internet releases were not exhaustively audited. No authors were contacted.

The GSM8K selection was committed to a local selection record before downloading outcomes: every regime model directory with `N3_R2_seed1`, rather than whichever model showed a desirable result. Other seeds and the superseded 512-token archive were outside this bounded analysis. [sources/external_trace_selection.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/sources/external_trace_selection.json) preserves this rule. There are 500 unique questions shared by all eight configurations, not 4000 independent questions. This exploratory personal research repository is independent public evidence, not a peer-reviewed or independently provider-audited benchmark release.

MuSiQue was acquired from a pinned 59.4 MB mirror with published SHA256; author-hosted archive byte equality is unverified. The author Drive HEAD probe returned HTTP 200 with HTML content, which is not evidence of successful dataset download. Definition/license/evaluator evidence comes from the original author repository and paper; mirror provenance is reported separately. [sources/musique_mirror_acquisition.json](/home/saurab/Documents/projectRLMAIN/WEBExtension/causal_study/external_validation/sources/musique_mirror_acquisition.json).

No search absence statement here means that suitable traces do not exist anywhere. The supported statement is that none were established in the inspected sources.
