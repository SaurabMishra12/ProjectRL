# Pre-inference amendment001: unique question units

Final structural inspection found2410 normalized question strings across2417 paired MuSiQue IDs: seven cross-ID duplicate groups. No inference or source annotation has run. The previous offline specification and affected artifacts are preserved here. They are superseded drafts, not new scientific results.

To enforce the unique-question unit, rank all IDs by SHA256("external-validation-v1/"+ID), retain the first ID for each question string normalized by casefold and Unicode word-token whitespace, then assign20 pilot,24 annotation and the remaining2366 confirmation-pool units. Select the first1000 confirmation units as before. This exclusion uses question identity only, not source sufficiency, model answers, correctness, disagreement or beneficial outcome. Preserve both source contexts of every retained ID. No silent replacement or model-output filtering occurred. All4834 source records remain unchanged; seven duplicate IDs are excluded prospectively.

This amendment corrects the design/feasibility denominator; it does not turn2410 question strings into independent natural facts or guarantee absence of shared subquestions. Generalization remains conditional on this benchmark.
