# K5 — REJECTED: Sufficiency-conditioned communication/readout interaction

Working title: *Communication Gains Under Source-Aware Evaluation*. This is an investigated design, not a recommended paper.

Question/hypothesis: does changing readout alter the utility of peer revision differently when context suffices versus when abstention is warranted? Hypothesis: stronger minority utilization improves answerable accuracy but can increase unsupported answering. Novelty verdict: **POTENTIALLY NOVEL only after a substantive mechanism gap is demonstrated; current formulation is insufficiently distinct**.

Three closest works: L8 fixes proposals and changes verification-aware readout; L10 measures sufficient/insufficient context and selective generation; L12 implements collaborative abstention and retrieval. The proposed cross joins existing questions. It must yield a new, consequential source-conditioned prediction beyond their combined known tradeoff; that is not established.

Evidence: H15 shows original/aware P communication gain 8/1 pp, interaction -7 pp CI [-10.333,-4]. Against: strict majority retains 7.667 pp; external numeric original/aware are identical; semantic sufficiency and entailment labels remain pending; E6 does not supply a qualified interaction. No universal readout repair is proposed.

Estimand: per model, J_q is the mean over paired sufficient/insufficient contexts of (U_peer,aware-U_private,aware)-(U_peer,original-U_private,original). U rewards supported correct answers in answerable contexts and valid abstention in insufficient contexts; invalid/upstream failure is scored by a fixed intention-to-treat rule. Also report false-answer risk, harmful flips, coverage and benchmark-native sufficiency, not just the composite. Both contexts/models cluster by normalized question. Gold/sufficiency labels are never inputs to readout or routing.

Smallest contrast: for each context/model,3 shared initial states +3 own-state revisions +3 simultaneous peer revisions. Fixed readouts cost 0 additional model calls. Two contexts =>18; two backbones=>36 per independent question. All sources fully shared initially, so this is not the earlier partitioned regime and transport requires evidence. Human source annotations must precede outputs; distinguish provider refusals, malformed responses and semantic abstentions.

Population: the pinned MuSiQue mirror has 2410 normalized question units,2366 remaining after existing 20/24 reservations; all 48 context annotations in the prior packet are pending. Deduplicate before selection; keep both contexts. Source identity/semantic sufficiency/serving revision qualification can reduce this pool. Do not treat 4834 contexts or eight model directories as independent questions.

Power:5 pp meaningful interaction is a planning choice, not an estimated effect. With alpha.025/model,80% normal sensitivity, assumed SD 0.25/.5/1/2 imply N=238/951/3803/15209. Last two exceed the pool. SD 2 is a loose range bound, not an estimated attainable variance or a simulated law. A source-realizable bounded sensitivity is available for smaller SD in 09. A20-question run cannot guarantee variance small enough for a meaningful confirmatory null.

Full study: N=1000 =>36,000 successes plus the prior technical 20 reserve 720 if ever newly authorized; at N=951 =>34,236. Reference E6 mix for 36,000 is 23,148,500 input +8,641,000 output tokens; serial summed request latency 9.811 h versus throttled amortized wall 20.101 h. Those are historical scaling references, not forecasts for MuSiQue long contexts or both backbones. Price is unverified; use token-price formula. Retry/failure budget and a prespecified variance gate are required before any future launch.

Positive result: a qualified interaction only for those registered contexts/models. Opposite result: competent readout can preserve beneficial communication and appropriate abstention; that would contradict a simplistic disappearance story. Tight null: bound registered interaction; wide null or abstention-free law: unresolved/ineligible. Final gate: **REJECT now** for unresolved substantive novelty, qualification and full-study power.
