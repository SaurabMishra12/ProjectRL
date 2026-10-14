# Gold-location diagnostics

Added at the user's request before held-out predictions were inspected. This analysis does not alter the frozen training or inference protocol. Gold annotations enter offline analysis only.

For each question, reconstruct the actual deterministic P assignment after the frozen BM25 top-six retrieval. Deduplicate supporting-fact article titles. With exactly two gold titles present, classify by whether their unique initial owners differ (Disjoint-Gold) or coincide (Co-located-Gold). Keep Retrieval-Missing-Gold and non-two-title Undefined-Gold questions as separate groups. Retain every question in the pooled primary analysis.

Report stratum counts, bridge/comparison composition, condition F1/EM and measured costs, paired communication effects with question bootstrap intervals, and the difference in paired effects between Disjoint-Gold and Co-located-Gold. E1 retains its pooled primary interaction; these strata are additional exploratory diagnostics. E5 averages optimizer seeds within questions. Holm correction covers paired effect tests within each stratified report; heterogeneity confidence intervals remain exploratory and unadjusted. Empty strata are explicitly unestimable.

For uniform 3/3/4 assignment of ten documents, co-location is (C(3,2)+C(3,2)+C(4,2))/C(10,2)=12/45=26.67%. With the actual top-six 2/2/2 assignment, it is 3/15=20%, conditional on retaining both articles. Observed proportions need not equal these expectations. No gold-aware resharding or exclusion is performed, and no magnitude or sign of the communication effect is assumed.

The implementation is kept in this analysis subdirectory so it does not change the frozen root-level training source manifest. Each completed stratified report records this analysis source hash and the underlying inference result hash. Supporting-article ownership describes evidence location, not a guarantee that a question requires communication or cannot be answered from model pretraining knowledge.
