# K1 — REJECTED: Activity-controlled evidence routing

Working title: *What Changes When Web Agents Communicate?* No paper is recommended under this title.

Problem/question: estimate the value of routing peer evidence after holding recipient revision activity fixed. Hypothesis: much apparent sparse-policy cost benefit is activation reduction, while source assignment has a smaller quality effect. This is falsifiable but its general premise is already explicit in L1/L2/L4/L5. Novelty verdict: **INCREMENTAL as presently formulated**.

Closest comparisons: L1 intervenes on organization under fixed budget limits but allows execution to change; L2 masks/reexecutes channels and explicitly analyzes node/edge cost; L5 removes nodes and edges across rounds. Our all-recipient own-state control is more tightly defined for this question, but a tighter control alone is procedural unless it changes a consequential general finding.

Evidence: ledger H08/H14. P learned seeds 19/31 each activate one receiver on 300 questions;912/1800 random pairs differ in recipient count. Against: W4 learned-static interval includes positive and negative effects; existing pruning papers already discuss participation. Neither evidence identifies a general activation-mediated share of quality or spending.

Estimand: for frozen initial states Z, fixed activation schedule A, fixed readout R and evidence regime E, tau(E)=E_q[Y_q(peer-route g,A,R,E)-Y_q(own-state,A,R,E)]. A source-assignment contrast compares g and g-prime at the same A and payload ceilings; it is distinct from a natural indirect-effect decomposition. Randomize before outcome and average all assigned questions. Measure task utility, false answers, prompt/completion tokens, failures and latency separately. Gold is only for scoring.

Smallest discriminating intervention: force every receiver to revise in both peer and private arms; freeze source bundles and initial states. For meaningful two-edge in-star rewiring while keeping one recipient, use at least four agents: three candidate senders allow three possible two-sender subsets. N=3 has none. Multiple graph choices, semantic evidence exposure and source overlap must be prespecified. Output lengths may change; equal caps do not guarantee equal realized cost.

Feasibility: four initial generations + four own-state revisions + four peer revisions =12 per question/model/regime for a single routed contrast, before alternative graph arms. Two regimes/two backbones =>48N; at illustrative N=951 =>45,648 calls. A20-question technical run would be 960 calls but is not recommended or a scientific pilot. A5 pp four-cell interaction with SD 0.5 requires approximately 951 questions at alpha.025/80% sensitivity; the relevant SD for this changed instrument is unknown. Current N=300 outcomes cannot establish it.

Full-study budget: actual prompt/output tokenization is unavailable. E6 reference 642.014 input/240.028 output per success implies a 29.352m input/10.957m output at 45,648 calls. This is not a forecast for four-agent long contexts. Request latency reference is 0.9811 s/call and throttled amortized wall 2.0101 s/call; serial HTTP and wall references must stay separate. Prices unknown. A valid future budget needs actual prompt upper bounds, immutable serving identities, retry cap and new explicit authorization.

Adverse outcomes: positive routing value would reject a claim that activation explains everything. A tight null would bound this registered contrast only. A wide null or failed evidence exposure is unresolved. Changes caused by local PPO defects would not constitute the proposed general mechanism. Final gate: **REJECT**; distinct contribution and realistic generalization not established.
