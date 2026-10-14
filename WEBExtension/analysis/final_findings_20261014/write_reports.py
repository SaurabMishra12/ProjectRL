"""Build four final reports from the complete offline analysis and supplied paper."""
import json,math,html,hashlib,shutil,re,sys
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd,numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];A=json.loads((R/'artifacts/active_machine_runs.json').read_text());C=Path(A['controller_study']);B=Path(A['controlled_run']);SPLIT=Path(A['controller_split_dir'])
a=json.loads((O/'audit_metrics.json').read_text());S=pd.read_csv(O/'tables/condition_summary.csv');F=pd.read_csv(O/'tables/all_condition_question_metrics.csv');M=pd.read_csv(O/'tables/E5_complete_seed_averages.csv');e5=json.loads((C/'reports/E5_statistics.json').read_text());plan=json.loads((C/'manifest.json').read_text());now=datetime.now(timezone.utc).isoformat()
def num(v):
 if v is None or (isinstance(v,(float,np.floating)) and not np.isfinite(v)):return 'NA (unknown)'
 if isinstance(v,(dict,list,tuple)):return json.dumps(v,ensure_ascii=False)
 if isinstance(v,(float,np.floating)):return format(v,'.6g')
 return str(v)
def table(d):
 d=pd.DataFrame(d)
 if not len(d):return 'No applicable records.'
 esc=lambda v:num(v).replace('|','\\|').replace('\n',' ')
 return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(esc(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))
def load(n):return pd.read_csv(O/'tables'/n)
def figure(n,caption):return f'\n![{caption}](figures/{n}.png)\n\n{caption}\n'
def effecttable(effects):
 return table([{'experiment':x.get('experiment','E5'),'regime':x.get('regime',''),'metric':x.get('metric','answer F1'),'contrast':x['contrast'],'n':x['n'],'effect':x['effect'],'CI95':x['ci95'],'raw_p':x['randomization_p'],'Holm_p':x.get('holm_p'),'helped':x.get('helped'),'harmed':x.get('harmed'),'ties':x.get('unchanged')} for x in effects])
def scalarflatten(x,path=''):
 if isinstance(x,dict):
  for k,v in x.items():yield from scalarflatten(v,path+'.'+str(k) if path else str(k))
 elif isinstance(x,list):
  for k,v in enumerate(x):yield from scalarflatten(v,f'{path}[{k}]')
 else:yield {'field':path,'value':x}
common=f'''Snapshot generated {now}. Scope: completed frozen WEBExtension E1–E5, controller training/validation/pilot, operational cost evidence, and the supplied 13-page manuscript. Each experiment uses the same 300 held-out question identities; the unit of inference is the question, not the arm, agent, seed, or repeated epoch. All analysis is offline; no inference calls were issued. Earlier raw artifacts and the 129-question audit are preserved. Display tables round to six significant digits; CSV/JSON files retain machine-readable precision. NA is unknown or inapplicable, never an invented zero.

Run identifiers: `run_f6aa61bac2c3dfae`, `controller_study_a0b3768357484b45`, split `controller_splits_2e025f8ae3d9815e`. Input hashes and final artifact hashes are in `final_manifest.json`.
'''
# All original numerical report fields, without flattening away inference metadata.
allfields=[]
for p in (O/'original_reports').rglob('*.json'):
 for x in scalarflatten(json.loads(p.read_text())):allfields.append({'source':str(p.relative_to(O)),**x})
pd.DataFrame(allfields).to_csv(O/'tables/all_original_report_fields.csv',index=False)
# Preserve full manuscript text, page boundaries, and all reported numbers without asserting their correctness.
paper=Path('/home/saurab/Downloads/Association_for_Computing_Machinery__ACM____SIG_Proceedings_Template (2).pdf')
raw=(Path('/tmp/sparsecomm_paper_raw.txt') if Path('/tmp/sparsecomm_paper_raw.txt').exists() else O/'sources/current_paper_extracted.txt').read_text();(O/'sources/current_paper_extracted.txt').write_text(raw)
claims=[]
for page,text in enumerate(raw.split('\f'),1):
 for line in text.splitlines():
  if re.search(r'\d',line) and not line.strip().isdigit():claims.append({'pdf_page':page,'extracted_line':line,'verification_status':'MANUSCRIPT_REPORTED_NOT_REVALIDATED'})
pd.DataFrame(claims).to_csv(O/'tables/manuscript_numeric_claim_inventory.csv',index=False)
# Report 1: complete quantitative record, scalar distributions and all original inference tables.
r1='# 1. Complete results and metric register\n\n'+common+'''
## Evidence status and interpretation

The completed E1–E5 files have 300 distinct test IDs each, identical question sets, matching completion SHA256 values, and unchanged frozen scientific source/checkpoint hashes. Answer F1 and EM were recomputed with the locally preserved official HotpotQA evaluator; all stored scores matched. All distinct transmitted payloads were recounted with the pinned tokenizer. This is strong internal consistency/provenance evidence, not external certification of the Azure deployment snapshot or an independent rerun.

The old paper's P1–P5/W1–W4 numbers are a separate historical evidence tier. The supplied manuscript is preserved in full text and its numerical lines catalogued. Those historical runs have not been newly reproduced or certified by the current E-suite; their numbers must not be relabeled as E1–E5 results. The result catalog below is exhaustive for the completed E-suite's recorded scalar condition fields and the explicitly listed recomputed diagnostics. It cannot supply unmeasured entailment, true provider bills, server compute time, or semantic/pretraining leakage.

## Data, arms and metric units

R means every agent sees the same BM25 top-six candidate articles. P means those six articles are exclusively partitioned across three agents (normally two each). F1 and EM use 0–100 scales. Payload is actual inter-agent token transmission per question; a broadcast copy counts separately per directed edge. Logical calls count the arm's required generations including shared initials, not new provider charges. Supporting-label metrics measure citation-set overlap with annotations, not textual entailment. E3 scores a designated receiver; all other main arms use normalized majority voting.

'''+table([{'experiment':e,**v} for e,v in a['integrity'].items()])+ '\n\n## All main arm results\n\n'+table(S[['experiment','condition','n','F1','F1_CI_low','F1_CI_high','EM','EM_Wilson_low','EM_Wilson_high','payload_tokens_known_mean','logical_calls_known_mean']])+'''

F1 summary intervals use 5,000 question-bootstrap resamples (seed 42); EM uses a Wilson interval. A per-arm interval is not a paired contrast test. Below, canonical completed contrast reports use 10,000 resamples and 10,000 Monte Carlo sign randomizations, with the (1+k)/(B+1) p-value correction. These p-values are not exact enumeration. Original family-specific Holm corrections are preserved.

## E1–E4 canonical paired effects

'''+effecttable(a['complete_effects'])+'\n\n## E5 canonical whole-policy effects: all F1 and utility contrasts\n\n'+effecttable(e5['effects'])+'''

Only the frozen E1 interaction is the primary endpoint. All E5 contrasts are exploratory relative to that primary question, despite the now-complete test. Utility is the prespecified F1-gain fraction minus payload and recipient-call penalties; it is not answer accuracy. Seeds are averaged within each question before uncertainty estimation. No noninferiority conclusion is licensed by a nonsignificant difference: no E-suite margin was predeclared.

## E5 seed-mean resource tradeoffs

'''+table(load('E5_complete_quality_resource_tradeoffs.csv'))+'\n\n## Training and validation: every epoch\n\n'+table(load('controller_learning_curves.csv'))+'''

There are 100 unique training and 30 unique validation questions, evaluated in both R and P. Per seed: 3 collection epochs × 200 contexts = 600 training rollouts and 3 × 60 = 180 validation records. Across seeds 7/19/31: 1,800 rollouts, 540 validation records and 252 optimizer minibatch updates (28 per epoch per seed). Validation-selected epochs are 1, 3, 3. These are not 1,800 independently sampled training questions. The prior larger study is archived; it is included only in global cost accounting.

'''+table(load('selected_seed_performance.csv'))+'\n\n## Pilot and static validation (not held-out efficacy results)\n\n'+table(load('pilot_static_summary.csv'))+'\n\nThe 12-question pilot checks parsing, costs and invariants. Static selection uses 30 validation questions in both regimes; it is not a test-set comparison. Full nested numeric fields and source links are retained in the supplemental tables.\n\n## Frozen selected actor distributions\n\n'+table(load('frozen_actor_probability_summary.csv'))+'\n\n## Split overlap and availability\n\n'+table([{'pair':k,**v} for k,v in a['leakage']['pairwise_checks'].items()])+'\n\n'+table([{'stratum':k,'n':v} for k,v in a['stratum_counts'].items()])+'''

Exact IDs, normalized questions and normalized candidate titles have no measured cross-partition overlap. Title aliases, semantic duplicates, pretraining contamination and all historical exposures were not exhaustively detectable. The same test questions are intentionally reused across E1–E5; they must not be counted as 1,500 independent observations.

## Physical usage, refusals and timing

'''+table(load('active_pipeline_physical_usage.csv'))+'\n\n'+table(list(scalarflatten(a['physical_usage_global'])))+'''

Known input and output dimensions are summed independently. Original report functions withheld both totals when either dimension was incomplete; originals are preserved, and these corrected known lower bounds are supplied alongside them. Unknown potentially billed original attempts remain unknown even when a replacement succeeds. Global attempt-ID-deduplicated totals include archived caps, pilots and the prior controller study; they are not only the final evaluation's costs. Logical arm totals must not be summed into a provider bill because arms share cached responses.

No verified deployment price or billing export is present, so currency spending and monetary savings are unmeasured. Missing token usage is not zero. Refusals remain abstentions in headline quality metrics. Timing covers captured local sessions and network intervals; nested timers overlap and do not estimate server compute or a randomized per-arm latency benefit.

'''+table(load('network_timing_summary.csv'))+'\n\n'+table(load('worker_session_timings.csv'))+'\n\n## Every scalar metric, distribution and missing count\n\nThe following includes all recorded scalar condition/cost fields and the recomputed support/joint-label diagnostics. Boolean means are proportions. `sum_known` for an arbitrary metric is descriptive and not necessarily a scientifically meaningful aggregate. Missingness is explicit.\n\n'
D=load('all_scalar_distribution_metrics.csv')
for metric,d in D.groupby('metric'):
 r1+='### '+metric+'\n\n'+table(d.drop(columns='metric'))+'\n\n'
r1+='## Every original report field\n\n[Full-precision field register](tables/all_original_report_fields.csv) retains all pooled/stratified effects, heterogeneity checks, action frequencies, diagnostics and metadata without selectively omitting inconvenient contrasts. The entire original report tree is copied under `original_reports/`.\n\n'
for p in sorted((O/'original_reports').rglob('*statistics*.json')):
 j=json.loads(p.read_text());r1+='### '+str(p.relative_to(O))+'\n\n```json\n'+json.dumps(j,indent=2)+'\n```\n\n'
r1+='## Full-precision files and original paper numbers\n\n'
for p in sorted((O/'tables').glob('*.csv')):r1+=f'- [{p.name}](tables/{p.name})\n'
r1+='\n[Supplied paper full extraction](sources/current_paper_extracted.txt) preserves all paper tables/figures text and numbers, including P1–P5/W1–W4. The numeric claim inventory labels them manuscript-reported. This report does not blend them into verified E-suite evidence.\n'
# Report 2: deeper findings and case studies.
r2='# 2. Findings from the complete raw outputs\n\n'+common+'''
## Main finding: communication usefulness and sparsity are conditional

In E5, the R learned seed mean has F1 32.1084, compared with 32.3140 for NoComm and 31.6510 for Broadcast. It sends 60.05 payload tokens and uses 3.97778 calls per question, versus 293.293 tokens and 6 calls for Broadcast. This is a 79.5256% payload reduction and 33.7037% logical-call reduction, with an inconclusive F1 difference (+0.45735, CI −0.68787 to 1.60985). It does not establish noninferiority. NoComm already supplies similar quality at zero transmission and three calls; learned utility is significantly worse than NoComm (−0.02002, Holm p=0.0019998).

In P, the learned seed mean reaches F1 10.5604 versus NoComm 5.05979, SelfUpdate 6.21606 and Static 6.42086. The gains are +5.50062, +4.34435 and +4.13955 points, respectively, and pass the stored E5 Holm family. Broadcast reaches 14.9920: learned is worse by 4.43158 points (CI −6.31219 to −2.70728; Holm p=0.0019998), while transmitting 46.2780% fewer payload tokens and requiring 22.3519% fewer logical calls. That is a measured quality/resource tradeoff. Fewer transmitted tokens do not imply the same percent reduction in billed input/output tokens or latency.

E5 utility against NoComm in P is +0.01624 with CI spanning zero; improvement in F1 is therefore stronger evidence than improvement in the prespecified reward. Static means the validation-selected OUTSTAR from only three candidates, not the best of all 15 two-edge supports. A claim of superiority over all static topologies would exceed the experiment.

## The main moderator hypothesis remains inconclusive

E1's frozen primary difference-in-differences is +2.88151 F1 points, CI [−0.259961, 6.01089], p=0.0712929. Communication improves P in secondary comparisons, but the pooled experiment does not confirm that communication benefits P more than R. The distinction matters: a significant effect in one group and a nonsignificant effect in another is not itself a significant interaction. E5 is a different whole-policy comparison and cannot retrospectively replace the E1 primary endpoint.

'''+figure('02_paired_effects','E1–E4 paired effects. Intervals and tests retain their original analysis families.')+'''
## Evidence availability is a major scope limitation

The fullwiki publisher candidates contain both annotated gold articles in 79/300 questions; BM25 top-six retains both in 76/300. Of the final pools, 62 are Disjoint-Gold, 14 Co-located-Gold and 224 Retrieval-Missing-Gold. Thus 74.6667% lack at least one labelled gold article after selection. Only three additional both-gold cases are lost by top-six selection. Most missing-gold cases are already missing in the supplied candidate set; calling all of this BM25 failure would be inaccurate.

For actual six-document 2/2/2 allocation, the chance that two retained distinct gold articles co-locate under uniform shuffling is 3×C(2,2)/C(6,2)=1/5. For hypothetical 3/3/4 allocation of ten documents it is [C(3,2)+C(3,2)+C(4,2)]/C(10,2)=12/45=26.6667%, rather than exactly 30%. The observed 14/76 co-location rate is descriptive. Fullwiki candidate retrieval is different from the guaranteed two-gold distractor setting.

Gold-missing does not mean no possible answer information exists: alternative passages or model memory can answer questions. Gold labels are used only for offline analysis/reward, not to choose deployment shards. The subgroup findings below are observational and post-hoc, with only 14 co-located cases. Larger point estimates do not establish a causal effect of belonging to a stratum.

'''+figure('04_retrieval_audit','Availability of annotated gold articles before and after BM25 top-six selection.')+table(load('E5_subgroup_diagnostics.csv'))+'''

In P, learned-minus-NoComm is +11.4316 points on Disjoint-Gold and +19.0476 on Co-located-Gold, compared with +3.01233 on Missing-Gold. The data do not support the advance prediction that the disjoint stratum must have the largest improvement. Against Broadcast the learned deficit is −9.95315 on Disjoint-Gold and −9.52381 on Co-located-Gold. Missing evidence and constrained transfer both matter; this experiment cannot isolate their causal shares.

'''+figure('15_E5_subgroups','Post-hoc P subgroup contrasts; marginal bootstrap intervals are not multiplicity-confirmed primary tests.')+'''
## The learned policy is often almost static

P seed 31 selects action 14 on all 300 questions: the two-edge in-star 0→1, 2→1 with quota 128. P seed 7 selects action 54 (six-edge broadcast with quota 256) on 294/300 questions. It produces exactly the same final answer as the Broadcast baseline on all 300 questions. Its high score therefore does not demonstrate a new routing advantage. P seed 19 mostly chooses another in-star. R seed 31 chooses NoComm on 241/300 questions. Across seeds, selected deterministic policies use only one to four actions per regime.

'''+table(load('E5_complete_action_adaptivity.csv'))+'\n\n'+table(load('learned_broadcast_identity.csv'))+'''

The three-seed mean is a statistical summary of three trained policies, not an implemented per-question mixture, voting ensemble or oracle selector. It should not be presented as a deployable single policy. Do not pick seed 7 post hoc as the headline router because it ties Broadcast. Selection was by validation utility; test-conditioned reselection would invalidate that decision rule.

We independently reconstructed the 53 observable features from saved initial responses and frozen assignments, then reevaluated the selected weights offline. All 1,800 regime×seed×question argmax actions matched their recorded choices. Mean maximum action probability is only 0.01878–0.02144, compared with uniform 1/55=0.01818, and mean entropy is 5.77835–5.78118 bits against log2(55)=5.78136. Thus near-uniform stochastic scores yield concentrated greedy choices. This is evidence of weak probability separation, not proof that training failed or an invitation to tune on test data. The next table distinguishes entropy of the actor's categorical probabilities from entropy of the selected action frequencies. A nearly uniform probability vector can still have the same slightly-largest action on every question; deterministic selection frequency alone does not reveal policy confidence.

'''+table(load('frozen_actor_probability_summary.csv'))+'''

## Matching edge count and quota does not match resources

Of 1,800 learned/random paired evaluations, edge count and quota match in all 1,800. Actual recipient-call counts differ in 912 (50.6667%) and transmitted token totals differ in 254 (14.1111%). An in-star updates one recipient, whereas another two-edge graph may update two. Short messages also leave different unused quota. Consequently, the P Learned–MatchedRandom F1 difference +1.01577 (raw p=.0456954; Holm p=.502650) is not multiplicity-robust and cannot establish pure routing superiority. Even a significant result would need the actual resource mismatch disclosed.

## E2 and E3 expose the limits of component attribution

E2 reverses one-step cyclic destinations while preserving the exact sender payloads, edge count and receiver calls in shared evidence. The CW–CCW difference is −0.106522 F1 points with a CI spanning zero. This is a valid narrow destination-allocation contrast, not evidence that all graph structures are interchangeable. E2 shares E1's R answers, payloads and provider call keys; it is a reanalysis of overlapping experimental data, not independent replication.

E3 uses a designated receiver and exact token-count matching, but 298/300 prefix/crop inputs are identical because messages fit the 256-token cap. Only two questions undergo an actual content change. The observed +0.144928 receiver F1 difference cannot establish general content-insensitivity, prefix superiority, compression equivalence or safety. The null result mainly diagnoses an ineffective manipulation. Removing the 298 ties after observing results cannot turn the remaining two into a confirmatory experiment.

## Stopping is a small replay saving in this protocol

E4 stops early on 10/300 questions: 3 at depth 0, 2 at depth 1 and 5 at depth 2; 290 run all three rounds. Six early consensuses are wrong, and continuation corrects none of those six here. Stopped and final answers/F1 coincide on all questions. Replay avoids 54/3,600 logical calls (1.5%) and reduces mean transmitted payload from 216.31 to 200.6267 (7.2504%). Full trajectories were actually generated, so this does not measure realized spending/time saved by an online stopping service. Unanimity is not a correctness certificate.

'''+figure('08_stopping_trajectory','E4 shared-trajectory replay: stopping depth and answer quality across rounds.')+'''
## Training, seed instability and execution-order drift

Training F1/reward and validation utility are separate quantities. Seed 7 has its best validation utility in epoch 1; later training reward improves without improving the selection criterion. Seeds 19 and 31 select epoch 3. Validation has only 30 unique questions (60 correlated R/P contexts), so selected policy instability is unsurprising but its cause is not established experimentally. All nine collection epochs completed four PPO passes with 28 minibatch updates each and no KL early stop. Training stochastic-action entropy remains near log2(55), while greedy test actions concentrate strongly; no convergence or optimality guarantee follows from completing three epochs.

'''+figure('09_controller_training','Recorded training rewards and validation curves; repeated questions are not new independent samples.')+'''

The historical 129-question P Learned–NoComm estimate was +7.83555 points; the final effect is +5.50062. The Broadcast deficit changed from −5.33519 to −4.43158. These changes reflect the evaluated prefix, not model learning during testing. Checkpoints were frozen. All prefixes and their construction are preserved below; no stopping rule or model selection used these post-hoc curves.

'''+table(load('execution_prefix_trends.csv'))+figure('16_prefix_drift','Diagnostic cumulative effects in frozen execution order, not a training curve.')+'''
## Real examples: gains, harms and scoring effects

The examples below are deliberately selected from the three largest and three smallest paired seed-mean F1 changes in each listed comparison. They illustrate mechanisms and failure types; they are not a representative sample or causal explanation. Gold answers come from the publisher. Exact question IDs link back to the saved E5 record. No new LLM judge was used. Benchmark F1 changes may reflect answer granularity, not only factual correctness: for example, “American Civil War” versus gold “War” receives lower token-overlap F1 even though the longer phrase may be semantically related.

'''
for case in json.loads((O/'raw_output_casebook.json').read_text()):
 r2+=f"### {case['comparison']} — {case['id']}\n\nQuestion: {case['question']}\n\nGold: **{case['gold_answer']}**. Type: {case['type']}; stratum: {case['stratum']}. Baseline answer: `{case['baseline_answer']}` (F1 {case['baseline_f1']:.6g}); seed-mean change {case['delta']:+.6g} points.\n\n"+table([{'seed':s,'answer':x['answer'],'F1':x['f1'],'edges':x['action']['edges'],'quota':x['action']['quota'],'actual_payload':x['payload']} for s,x in case['learned'].items()])+'\n\n'
r2+='''## What the data cannot settle

Citations were checked as annotated label sets, not entailment. Refusal incidence is retained in primary scores; conditioning on refusal status is treatment-dependent and only descriptive. The deployment alias is not an immutable model snapshot. One benchmark, one evidence partition/role/crop seed and three optimizer seeds do not establish cross-model, cross-partition or live-Web robustness. No new external baseline systems were run by this analysis. These boundaries should shape the paper, not be hidden in a footnote.
'''
# Report3 narrative changes and mathematically precise repair.
r3='# 3. Narrative support, manuscript corrections and replacement prose\n\n'+common+'''
## Verdict

The completed experiments support a careful empirical paper about the conditions under which communication helps and the difficulty of attributing savings to routing, content selection and stopping. They do not support the current abstract's broad claim that each component independently preserves reasoning fidelity, or a general claim of learned routing superiority. Completion removes the missing-E5 concern; it does not remove the ineffective E3 manipulation, quality loss versus Broadcast in P, weak routing-specific evidence, single-backbone scope or legacy-provenance questions.

The strongest defensible story is: **evaluate communication as an information-delivery decision, audit whether intended interventions actually change inputs, and distinguish whole-policy quality/resource tradeoffs from isolated structural effects.** E1–E5 provide useful positive and negative evidence for this story. A strong final paper needs claims that survive those negative findings. Acceptance and novelty cannot be guaranteed by finishing an experiment checklist.

## Claim-by-claim changes to the supplied 13-page PDF

| Location | Current claim or presentation | Evidence and required action |
|---|---|---|
| Abstract, p.1 | 50.4–90.2% transmission reduction with statistical noninferiority across primary benchmarks | Those are historical P-series claims, not E-suite findings. Retain only with separately verified legacy raw provenance and NI protocol. For E5 report 79.53% R / 46.28% P payload reductions against its quota-capped Broadcast, alongside CIs and the P quality deficit. No E-suite NI margin exists. |
| Abstract, pp.1 and conclusion p.10 | Each mechanism independently reduces overhead while preserving fidelity; stopping/rationing are primary drivers | E3 has two actual changes, E4 saves only 1.5% hypothetical calls here, E5 changes support/quota/recipient calls together. Replace with protocol-specific measured tradeoffs and unresolved attribution. |
| Introduction, p.1 | One controller framework and common protocol across all results | Clearly distinguish legacy continuous SparseComm and the new categorical one-step SparseComm-Web instrument. The new study does not validate the old architecture or GAE/halting implementation. |
| Section 3, pp.3–4 | “measure-zero” equal-volume set | Equality may be systematic for discrete lengths, symmetric supports and sender-preserving cycles. Require a continuous-density assumption or remove this claim. E2 explicitly constructs equality. |
| Proposition 2 | Volume shifts demonstrate a large mediated quality effect | Algebraic decomposition is valid given well-defined potential outcomes, but a 3.4× volume change measures volume, not the size of the indirect outcome effect. Avoid claiming identified mediation. |
| Proposition 3 and proof | Every support induces a proper simplex face; fixed total alone establishes pure routing | Nonempty proper supports yield proper faces; full support yields the whole simplex, and the empty support is infeasible for positive mass. Specify message content/allocation, receiver calls and update rules as part of the intervention. |
| Proposition 4, p.4 | Uses arm mean volume in the observed causal contrast and says equal mean recovers fixed-mass identification | The displayed identity is for deterministic interventions at scalar volumes, not automatically the actual average outcome under question-varying realized volumes. Equal means do not imply pairwise equality. Replace as below. |
| Section 4.2, p.4; Algorithm 1, p.12; Table 14, p.13 | 20-D state, 61-D Gaussian action, low-rank intensities, MiniLM, GAE, MAP/Platt and learned halt | E5 has 53 observable inputs, 55 categorical actions, two 64-unit Tanh layers, Adam 3e-4, one-step reward, deterministic argmax at test, unweighted normalized majority. No MiniLM, GAE, calibrated MAP or trained stopping in E5. Give separate methods/configuration tables. |
| Section 5.1, p.5 | Common 512-token cap and shared calibrated aggregator for all protocols | E-suite freezes max_tokens=8192, context limit=32768, temperature=0 and a pinned exact tokenizer. Top-p should only be stated if explicitly fixed in the request. Do not inherit old 512-token settings. |
| Table 1, p.5 | Existing P/W map | Add E1–E5 with exact intervention, primary endpoint, n=300, evidence regime and inferential tier. E2 is overlapping reanalysis. Label old W1–W4 as historical. |
| Section 6.3, p.6 | Common caps remove the volume confound | Common caps need not match realized messages or receiver work. Describe capped whole-policy comparisons; actual per-question equality is necessary for the stated narrow structural estimand. |
| Section 7.1, p.8 | Prefix “demonstrating” superiority over arbitrary crops | The old W2 CI crosses zero and Holm p=.750; new E3 is degenerate. Change to an uncertain point estimate in the old protocol and an ineffective content intervention in E3. |
| Section 7.2, pp.8–9 | W4 proves sparse gains require fresh multi-round reasoning propagation | No randomized fresh-vs-replayed payload comparison identifies this necessity. State only that the fixed replayed support did not outperform its baseline. |
| Discussion, p.9 | Efficient coordination is primarily governed by adaptive depth rather than per-question rewiring | This is not established across protocols; E4 shows small savings, and E5 is one-step. Explain when each effect was observed and when evidence is absent. |
| Section 9, p.10 | Web word-proxy costs and condition tags | Keep this as a historical limitation of W1–W4. E-suite uses exact pinned-token counts and internal cache namespaces. Clearly separate these evidence tiers. |
| Table 3 vs Table 11, pp.6/13 | Peak PPO and five-seed summaries | Audit identities and denominator: cap-100 Table 11 lists a 94.0 seed but Table 3 calls 92.67 the “peak”. If different runs are intended, label them; otherwise correct from originals. Current E-suite does not validate either number. |
| Table 8 vs Table 2 and text, pp.12/6 | Hotpot NI p-values at margins 5 and 10 | Main text gives .0012 at margin 5; table transcriptions differ (.001 and <.0012). Recompute from legacy paired records and use one rounding convention. |
| Artifact URL, p.1/p.10 | All code/checkpoints/raw traces available | This analysis did not verify the anonymous repository contents. Publish a sanitized manifest and verify accessibility/completeness before retaining that assertion. |

## Corrected estimand statement

A scalar token total does not specify which text is delivered to which recipient. Let a fixed, fully specified allocation/content rule define the intervention Y_q(t,v). If t changes message versions or recipient calls beyond the intended destination intervention, name those changes rather than assuming them away. For a common reference volume v0(q) feasible for each compared arm, define:

TE = E_q[Y_q(t,V_t(q)) − Y_q(t′,V_t′(q))].

CDE(v0) = E_q[Y_q(t,v0(q)) − Y_q(t′,v0(q))].

Then TE = CDE(v0) + R_t − R_t′, where R_t = E_q[Y_q(t,V_t(q)) − Y_q(t,v0(q))]. If each question/arm has a valid Lipschitz bound L_q,t over the relevant feasible interval, then

|TE − CDE(v0)| ≤ E_q[L_q,t |V_t(q)−v0(q)| + L_q,t′ |V_t′(q)−v0(q)|].

This is a conditional sensitivity bound, not an empirically estimated correction. No L values were measured in E1–E5. Under exact question-level equality V_t(q)=V_t′(q)=v0(q), the residual terms vanish, provided the intervention is well-defined and other mechanisms are controlled. Equal dataset means are insufficient. A common upper cap alone supplies neither equality nor an informative outcome bound. The scalar-volume decomposition does not control content quality or recipient computation by itself.

For fixed binary A, the projection argument in Proposition 1 is elementary and correct for fixed B≥0. In a multi-round system, B can itself change with routing, so the fixed-B support subtraction formula should not be presented as the entire policy-induced change in generated volume. Avoid treating these bookkeeping identities as a novel causal identification theorem without a precise assumptions/contribution comparison to prior mediation work.

## Recommended paper structure and figure replacements

1. Abstract/introduction: information availability, interference and resource-attribution challenge in candidate-based Web QA; state the one-backbone scope on the first page.
2. Estimands and protocol validity: define controlled destination allocation versus whole-policy effects, include a diagram of shared versus partitioned evidence and a check of whether message interventions actually differ.
3. Methods: categorical controller, saved-data split, reward and validation-only selection; separate legacy continuous architecture if retained.
4. Results: E1 primary interaction first, followed by conditional P gains; E2/E3 validity findings; E4 stopping replay; E5 full policy tradeoffs with all seeds.
5. Evidence availability and failure analysis: the three gold strata, missing-candidate issue, real harmed/helped examples, seed-dependent static behavior.
6. Discussion/limitations: partial support for the narrative, no global NI claim, single backbone/benchmark/partition, unknown bills, legacy results requiring their own audit.

Replace a “paper in one figure” that implies uniform superiority with (a) evidence-availability counts, (b) primary and E5 paired effects with uncertainty, and (c) payload/call tradeoffs against both Broadcast and NoComm. Show all selected seeds rather than only a peak. Move full scalar/stratum tables to an appendix linked to Report 1. State that E2 shares E1 data. Keep E3's ineffective manipulation visible instead of deleting its null result.

## Suggested title

**When Does Communication Help? Evidence Availability and Resource Attribution in Multi-Agent Question Answering**

A version preserving the current title is **Signal over Chatter: Auditing Evidence Delivery and Communication Tradeoffs in Multi-Agent Question Answering**. “Decomposing routing, rationing and stopping” should be qualified if used, because the new suite does not fully identify all three components independently.

## Replacement abstract (for the verified E-suite scope)

Multi-agent question-answering systems exchange evidence through messages whose routing, content and length jointly affect both predictions and resource use. We study these effects in a controlled three-agent setting using 300 paired HotpotQA fullwiki candidate-retrieval questions and a frozen gpt-oss-120b deployment. Our protocols compare shared and partitioned evidence, reverse cyclic destinations while preserving sender payloads, test exact-length content selection, replay consensus stopping, and evaluate a validation-selected categorical PPO controller. The primary interaction between communication and evidence partitioning is inconclusive: 2.88 answer-F1 points (95% CI −0.26 to 6.01). In partitioned evidence, the learned policies improve over NoComm by 5.50 points, but underperform quota-capped broadcast by 4.43 points while transmitting 46.28% fewer payload tokens. Under shared evidence, they transmit 79.53% fewer payload tokens than broadcast without an established quality difference, although NoComm remains a strong lower-cost baseline. Input-level audits reveal that 224 of 300 selected pools lack at least one annotated gold article, and 298 prefix/crop pairs are identical. These findings support conditional communication gains and highlight the need to verify intervention strength and realized resources before attributing performance to graph structure.

This abstract intentionally describes only authenticated new results. Incorporating legacy multi-benchmark claims requires independent evidence validation and an explicit two-instrument methods section.

## Replacement results paragraph

On partitioned evidence, the three selected controller policies achieve a seed-mean answer F1 of 10.56, compared with 5.06 for NoComm, 6.22 for SelfUpdate and 6.42 for the validation-selected static support. Paired F1 gains over these baselines remain significant under the report's Holm correction. Broadcast achieves 14.99 F1; the learned policies therefore trade 4.43 F1 points (95% CI −6.31 to −2.71 for learned minus broadcast) for 46.28% fewer transmitted payload tokens and 22.35% fewer logical calls. The learned policies do not establish an F1 advantage over the edge-count/quota-matched random control after multiplicity adjustment. This control also differs in realized recipient calls and payload on some questions, so it is a whole-policy comparison rather than a pure routing intervention.

## Replacement discussion and limitations paragraph

The results support communication when agents hold restricted evidence, but do not imply that a learned router is uniformly preferable. Shared-evidence NoComm already performs comparably to communicating policies with fewer calls, while the partitioned controller's quality is lower than broadcast. Deterministic routing is strongly seed-dependent: one selected policy broadcasts on 98% of partitioned questions, and another uses a single in-star throughout. The apparent content comparison is also weak because most paired payloads are identical. These findings motivate reporting realized resources, policy diversity and intervention fidelity together with answer quality. Our evidence is limited to one deployment, one benchmark sample and three optimizer seeds; it does not establish cross-backbone robustness, citation entailment, calibrated noninferiority, or realized monetary savings.

## Remaining work, in priority order

**Can be fixed from current evidence:** replace unsupported claims, use the correct categorical method/settings, keep the primary null visible, separate old/new protocols and token units, report all seeds and resource mismatches, repair Proposition 4, add costs/missingness, and supply a reproducibility manifest.

**Requires additional evidence if those claims remain:** a content intervention that changes meaningful text on enough questions; recipient/call/payload-matched routing controls; independent partition/role seeds and preferably a second backbone or benchmark; an online stopping run if actual latency/billing savings are claimed; separately authenticated P-series/W-series raw records and preregistration for historical NI claims. These were not silently run or fabricated. Define new protocols before new data collection and retain the present test results unchanged; now-observed held-out data must not become a tuning target for a new confirmatory claim.

**Literature/venue validation:** this report audits claim-evidence alignment, not an exhaustive novelty ranking or every bibliographic entry. The official WWW2027 research-track call requires substantive Web relevance on the first page; candidate Wikipedia retrieval alone does not establish it. Explain the scientific information-access challenge, and avoid describing offline candidate reranking as live-Web search or retrieval over all Wikipedia. Primary references checked: [WWW2027 research track](https://www2027.thewebconf.org/research-track-papers/), [HotpotQA](https://hotpotqa.github.io/), [official evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).

## Adversarial self-review

- Contribution: a useful controlled empirical audit is supported; broad algorithmic superiority and full causal decomposition are not.
- Clarity: distinguish E from P/W, F1 from thresholded accuracy, actual tokens from proxies, and policy means from a deployed ensemble.
- Experimental strength: all planned E-suite rows exist; that does not repair E3's ineffective manipulation or small subgroups.
- Evaluation completeness: reports are complete within the frozen scope; independent robustness and legacy provenance remain separate work.
- Method soundness: exact split checks and frozen actor replay pass; arbitrary-volume potential outcomes require explicit implementation assumptions, and random controls do not equalize all realized resources.
'''
# Report4 exact protocol.
r4='# 4. Technical methods, pseudocode and reproducibility\n\n'+common+'''
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

'''+table(list(scalarflatten({'base':{k:v for k,v in plan['base'].items() if k not in ['base_url','credential_source']},'settings':plan['settings']})))+'''

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

## E3: content selection at exact token count

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
'''
reports={'01_COMPLETE_RESULTS_AND_METRICS.md':r1,'02_DEEP_RAW_OUTPUT_FINDINGS.md':r2,'03_PAPER_NARRATIVE_AND_REVISION_PLAN.md':r3,'04_TECHNICAL_METHODS_AND_PSEUDOCODE.md':r4}
for name,text in reports.items():(O/name).write_text(text)
# HTML exports use a local markdown renderer when available; safe raw text fallback remains readable.
try:
 from markdown_it import MarkdownIt
 md=MarkdownIt('commonmark',{'html':False}).enable('table').render
except ImportError:md=lambda x:'<pre>'+html.escape(x)+'</pre>'
css='body{font:16px/1.6 system-ui,sans-serif;color:#18242f;max-width:1400px;margin:40px auto;padding:0 28px}h1{font-size:30px}h2{border-top:1px solid #d5dce1;padding-top:24px;margin-top:40px}table{border-collapse:collapse;display:block;overflow-x:auto;margin:20px 0;font-size:13px}th,td{border:1px solid #d5dce1;padding:7px 10px;text-align:left;vertical-align:top}th{background:#edf2f6}img{max-width:100%;height:auto}pre{background:#f4f6f8;padding:18px;overflow:auto;white-space:pre-wrap}a{color:#175b86}nav{display:flex;flex-wrap:wrap;gap:18px}code{font-size:.9em}@media print{body{margin:0;font-size:10pt}table{font-size:8pt}h2{break-before:auto}img{max-height:230mm;object-fit:contain}}'
for name,text in reports.items():
 body=md(text);(O/name.replace('.md','.html')).write_text('<!doctype html><html><head><meta charset="utf-8"><title>'+name+'</title><style>'+css+'</style></head><body><nav><a href="index.html">All reports</a></nav>'+body+'</body></html>')
index='# SparseComm complete findings bundle\n\nFour reports based on all completed E1–E5 results. No fresh inference.\n\n'
for name in reports:index+=f'- [{name[3:-3].replace("_"," ").title()}]({name.replace(".md",".html")}) — [editable Markdown]({name})\n'
index+='\nThe verified conclusion is conditional communication benefit with resource/quality tradeoffs, not uniform router superiority. The historical partial audit is preserved separately.\n\n'
index+='## Complete data and figures\n\n'
for p in sorted((O/'tables').glob('*.csv')):index+=f'- [{p.name}](tables/{p.name})\n'
index+='\n## Figure gallery\n\n'
for p in sorted((O/'figures').glob('*.png')):index+=f'### {p.stem}\n\n![{p.stem}](figures/{p.name})\n\n[Vector PDF](figures/{p.stem}.pdf)\n\n'
(O/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>SparseComm final findings</title><style>'+css+'</style></head><body>'+md(index)+'</body></html>')
# Compact read-only notebook, preserving the user's notebook workflow.
import nbformat
nb=nbformat.v4.new_notebook();nb.cells=[nbformat.v4.new_markdown_cell('# Complete E1–E5 findings\n\nThis notebook reads saved reports and figures only. Recalculation commands are in Report 4. No model requests are made.'),nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\nroot=Path('/home/saurab/Documents/projectRLMAIN/WEBExtension/analysis/final_findings_20261014')\ndisplay(pd.read_csv(root/'tables/condition_summary.csv'))")]
for name in reports:nb.cells.append(nbformat.v4.new_markdown_cell(f'[{name}]({name})'))
for fig in ['02_paired_effects','10_E5_complete_quality','15_E5_subgroups','16_prefix_drift']:nb.cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(root/'figures/{fig}.png')))"))
nbformat.write(nb,O/'FINAL_FINDINGS.ipynb')
# Snapshot all scientific input references; fail if they changed during analysis.
inputs={}
for e in ['E1','E2','E3','E4','E5']:
 p=(C if e=='E5' else B)/(e+'.jsonl');h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==a['integrity'][e]['sha256'];inputs[str(p)]=h
for seed,h in json.loads((C/'evaluation_gate.json').read_text())['checkpoint_hashes'].items():
 p=C/f'seed_{seed}/selected.pt';assert hashlib.sha256(p.read_bytes()).hexdigest()==h;inputs[str(p)]=h
for name,h in plan['sources'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
for name in ['durable_storage.py','durable_runtime.py','durable_worker.py','rate_limited_machine.py','adaptive_pacing.py']:shutil.copyfile(R/'analysis'/name,O/'sources'/('analysis__'+name))
inputs[str(paper)]=hashlib.sha256(paper.read_bytes()).hexdigest()
manifest={'created_utc':now,'scope':'Four final reports from complete E1–E5; legacy manuscript values separately labeled','inference_calls_issued':0,'result_counts':{e:a['integrity'][e]['n'] for e in a['integrity']},'score_mismatches':len(a['score_mismatches']),'payload_token_mismatches':len(a['token_count_mismatches']),'inputs_sha256':inputs,'derived_artifacts':{str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in O.rglob('*') if p.is_file() and p.name not in ['final_manifest.json','analysis_artifact_manifest.json']}}
(O/'final_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'reports':list(reports),'figures':len(list((O/'figures').glob('*.png'))),'tables':len(list((O/'tables').glob('*.csv'))),'score_mismatches':manifest['score_mismatches'],'token_mismatches':manifest['payload_token_mismatches']},indent=2))
