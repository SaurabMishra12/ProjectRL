# Measured 150-question 2Wiki evaluation

150 paired questions completed: 20 reused pilot questions and 130 additional questions. All seven arms and both readouts are retained.

**The original 20-question gate failed.** The user subsequently authorized expansion to 150. No prompt, parser, router, assignment of the first 100, or model setting changed.

![Additional 130 outcomes and logical token costs](additional_results.png)

The new 130 provide the primary extension results. Pooled 150 results and pilot results are separately available in summary.json. These publisher training-split results and all paired contrasts are exploratory.

## Additional 130 questions

- full_union: abstention-aware 93/130 correct, F1 80.24; Sink-1 93/130 correct, F1 80.24.
- nocomm: abstention-aware 25/130 correct, F1 22.17; Sink-1 8/130 correct, F1 7.05.
- random: abstention-aware 44/130 correct, F1 36.85; Sink-1 34/130 correct, F1 27.12.
- heuristic: abstention-aware 60/130 correct, F1 50.91; Sink-1 51/130 correct, F1 43.55.
- complementary: abstention-aware 54/130 correct, F1 45.44; Sink-1 45/130 correct, F1 37.56.
- instar: abstention-aware 71/130 correct, F1 59.83; Sink-1 66/130 correct, F1 54.96.
- dense: abstention-aware 91/130 correct, F1 77.26; Sink-1 66/130 correct, F1 54.96.

For these additional 130 questions, In-Star uses 260 directed deliveries and 130 recipient calls, versus 780 and 390 for Dense. Known logical model-token references are 708,467 versus 1,109,317 (36.13% fewer). These are distinct cost measures; the fixed-sink equality does not imply equality under abstention-aware voting.

## Audited scope

- Reconstructed all arm requests and officially rescored 2100 readouts from saved responses; 150/150 In-Star and Dense Sink-1 requests match byte for byte.
- Sink equality is guaranteed by the one-round incoming message identity and physical response reuse. It is not an independent noninferiority result.
- Physical parse outcomes: {'valid': 1281, 'invalid': 1}. Refusal, malformed and unfinished outputs remain operational failures; explicit abstentions remain in all assigned denominators.
- Per-arm logical token references overlap across physically reused requests. Payload, router metadata, and provider token usage are separately reported. Currency cost is unavailable.
- This study evaluates deterministic routing and fixed-sink synthesis, not newly trained SparseComm controller weights. It does not establish asymptotic scaling or superiority of learned routing.
- The relaxed common parser removes source-visibility and list-size schema rejection by design; it does not verify every generated fact.
- Questions have gold-complete evidence in the union. Whole-article partitioning is independent of gold labels. Gold completeness does not establish that every example semantically requires communication.
- Shared training provenance and previous pilot inspection limit confirmatory claims. No performance-based early stopping or post-outcome prompt tuning was used.

[Full metrics and exploratory paired intervals](summary.json) · [All predictions](predictions.json) · [Offline audit](audit.json) · [Failure cases](failure_cases.json) · [Authorization amendment](authorization.json)

One original Round-0 Azure request timed out at question 55. Its generation and billing status remain unknown in the unchanged parent ledger. The user explicitly authorized one replacement attempt. The two other saved Round-0 responses and all 54 completed questions were reused. The timeout's full reservation remains counted against the original combined ceiling; no unresolved attempt was relabeled as a provider rejection.
