"""Render evidence-led Markdown and standard scientific figures from saved results."""
import csv,json,os,re
from pathlib import Path
from .reanalyse import ROOT,save

def link(name,label=None):return f'[{label or name}]({ROOT/name})'
def write(name,text):
    chunks=re.split(r'(`[^`]*`|https?://[^\s)]+|/[^\s)]+)',text.strip())
    nouns=r'(?:questions|question|contexts|identities|files|records|models|backbones|successes|calls|paired|shared|readouts|draws|replicates|agents|paragraphs|tokens|configurations|preprint|correct|states|fields|MB|pp)\b'
    for i in range(0,len(chunks),2):
        held={}
        def hold(match):
            key='PRESERVED'+chr(0xe000+len(held))+'TOKEN';held[key]=match.group();return key
        chunks[i]=re.sub(r'\b(?:GSM8K|SHA256|E[1-8]|W[1-4])\b|\b\d+(?:\.\d+)?e[+-]?\d+\b',hold,chunks[i])
        chunks[i]=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',chunks[i])
        chunks[i]=re.sub(r'(?<=[A-Za-z])(?=[+−]\d)|(?<=%)(?=[A-Za-z0-9])',' ',chunks[i])
        chunks[i]=re.sub(r';(?=\S)','; ',chunks[i])
        chunks[i]=re.sub(r'(\d)(?='+nouns+r')',r'\1 ',chunks[i])
        chunks[i]=re.sub(r'\b(October|NeurIPS|ICML|ACL|TACL|COLM|ICLR|FindingsACL)(?=20\d\d)',r'\1 ',chunks[i])
        chunks[i]=re.sub(r':(?=[A-Za-z0-9])',': ',chunks[i])
        for key,value in held.items():chunks[i]=chunks[i].replace(key,value)
    (ROOT/name).write_text(''.join(chunks)+'\n')
def main():
    local=json.loads((ROOT/'results/local_inventory.json').read_text())
    hist=json.loads((ROOT/'results/historical_four_readouts.json').read_text())
    ext=json.loads((ROOT/'results/external_gsm_results.json').read_text())
    mus=json.loads((ROOT/'results/musique_inspection.json').read_text())
    power=json.loads((ROOT/'results/power_sensitivity.json').read_text())
    budget=json.loads((ROOT/'results/api_budget.json').read_text())
    protocol=json.loads((ROOT/'frozen/protocol.json').read_text())
    hm={(r['arm'],r['rule']):r for r in hist['metrics']}
    source_rows=[]
    descriptions={
        'deeplearning-wisc__debate-or-vote':('Debate or Vote','38-entry tree; code and evaluator, no saved per-agent evaluation files identified in this pinned tree','NO compatible traces found in inspected tree'),
        'Wang-ML-Lab__when-debate-helps':('When Debate Helps','49-entry tree; construction/test examples are dataset examples, not generated multi-agent response histories','NO compatible traces found in inspected tree'),
        'jonathansantilli__freemad':('Free-MAD linked software','326-entry current tree is a broader evolving software project; inspected regular JSON files are configurations; no paper-result trace archive identified','UNAVAILABLE in inspected tree; current software is not itself paper-run provenance'),
        'HanNight__RAMDocs':('MADAM-RAG / RAMDocs','5 blobs: data, code, README, requirements and license. RAMDocs_test contains documents/labels, not saved agent completions','NO compatible saved traces in inspected tree'),
        'BunsenFeng__AbstainQA':('Don\'t Hallucinate, Abstain','30-entry tree contains input datasets and approach scripts; no published agent-run output histories identified','NO compatible traces found in inspected tree'),
        'stonybrooknlp__musique':('MuSiQue','94-entry code tree; README links official four-model predictions externally; those are predictions, not communication histories. Linked prediction archive was not downloaded','NOT a communication replication; full external prediction archive not audited'),
        'moozwen__sakana-debate':('Public exploratory GSM8K log','212-entry tree; prespecified all eight regime model directories, seed1,N3,R2; raw text and parsed predictions across three rounds','PARTIAL compatibility: reconstructable numeric votes/labels; no separate semantic abstention or provider-refusal field, no matched three-seat private control')}
    for key,(title,scope,status) in descriptions.items():
        d=json.loads((ROOT/'sources'/(key+'_tree.json')).read_text());source_rows.append({'source':title,'repo':key.replace('__','/'),'commit':d['sha'],'tree_entries':len(d['tree']),'tree_truncated':d.get('truncated',False),'scope':scope,'eligibility':status})
    save('results/trace_eligibility.json',source_rows)
    items='\n\n'.join(f"- **{r['source']}**, commit `{r['commit']}`. {r['scope']}. **{r['eligibility']}**. [Public repository](https://github.com/{r['repo']})." for r in source_rows)
    write('01_EXTERNAL_TRACE_AVAILABILITY.md',f'''
# External trace availability

Audit date:16 October2026. **No suitable independent Web-QA trace set preserving the required abstention semantics and reconstructable communication comparison was established in this bounded search.** An independent numeric-QA pipeline is partially usable, and is analyzed rather than discarded because it does not fit the proposed narrative.

The five eligibility requirements were checked separately: initial independent proposals, retained abstention/refusal/missing categories, individual outputs, reconstructable final readout, and accepted labels. A release of code or input data is not a release of generated traces. An output containing parser `None` is not automatically a model expressing uncertainty.

## Local evidence

The direct local inventory inspected{len(local['files'])}files in the explicit scope recorded in {link('results/local_inventory.json')}. It includes E1–E5, corrected E3, old W1–W4 predictions and older task summaries. Existing source-transition work covers E3-v2. The E5 states give strong within-pipeline reconstruction, but E1–E5 reuse the same300 identities; summaries from other local tasks have different schemas, missing required output distinctions, or unestablished independent generation provenance. They are not independent evidence for the source-bound abstention interaction. The separate new E5 implementation reproduces all60 preserved first-three-readout counts; this verifies computation, not population generality.

Graph evidence was Tier2, project `projectRLMAIN`, generation `2026-10-13T17:28:45Z`. Targeted WEBExtension aggregation lookup returned no graph matches; exact coverage was `not_tracked`. Direct source inspection and bounded file inventory therefore supply the evidence. A clean coverage flag is not a completeness guarantee.

## Public releases inspected

{items}

The actual trees, relevant code and acquisition receipts are retained in {link('sources','sources/')}; {link('results/public_blob_verification.json')} verifies downloaded Git blob hashes against pinned trees. GitHub/Hugging Face metadata retrieval is public HTTP activity; **new paid inference calls=0**. Repository trees were complete for these commits, but linked storage, other branches, private data and all internet releases were not exhaustively audited. No authors were contacted.

The GSM8K selection was committed to a local selection record before downloading outcomes: every regime model directory with `N3_R2_seed1`, rather than whichever model showed a desirable result. Other seeds and the superseded512-token archive were outside this bounded analysis. {link('sources/external_trace_selection.json')} preserves this rule. There are500 unique questions shared by all eight configurations, not4000 independent questions. This exploratory personal research repository is independent public evidence, not a peer-reviewed or independently provider-audited benchmark release.

MuSiQue was acquired from a pinned59.4MB mirror with published SHA256; author-hosted archive byte equality is unverified. The author Drive HEAD probe returned HTTP200 with HTML content, which is not evidence of successful dataset download. Definition/license/evaluator evidence comes from the original author repository and paper; mirror provenance is reported separately. {link('sources/musique_mirror_acquisition.json')}.

No search absence statement here means that suitable traces do not exist anywhere. The supported statement is that none were established in the inspected sources.
''')
    write('02_ABSTENTION_AND_ANSWERABILITY_CONTRACT.md',f'''
# Abstention, answerability and aggregation contracts

**Ignoring abstentions is not a generally safe repair.** On frozen proposals `[abstain, abstain, Paris]`, ordinary plurality abstains, while abstention-aware plurality answers Paris. If the evidence is insufficient and abstention is required, the latter has false-answer rate100% for that constructed case. This is a tested logical counterexample, not an observed benchmark failure rate. On an answerable Paris question, the same change can improve EM. Gold is never used to choose a candidate.

## Separate the objects

- **Question answerability in the world:** a fact may have a real answer independently of this packet.
- **Source sufficiency:** the presented documents support deriving an answer; missing annotated support alone cannot prove insufficiency.
- **Expressed abstention:** a source-defined deliberate action, such as ProjectRL `noanswer` or MuSiQue `predicted_answerable=false`.
- **Provider refusal:** a distinct provider signal. Preserve it even when a readout cannot use it as a vote; do not retry it as though it were a transport failure.
- **Empty/malformed/missing output:** a formatting, extraction or availability problem, not evidence that a model assessed answerability.
- **Substantive wrong answer:** an answer attempt that fails the source evaluator; it is not abstention.

{link('contracts.py')} implements typed proposals and independent gold-free readouts. Raw categories remain available at proposal level; system decisions also retain all-refusal and all-invalid provenance. A system abstention produced by a conservative vote is a readout action, not a newly observed model statement. The old E5 contains only substantive answers, explicit abstentions and flagged refusals; invalid-output handling is not silently retrofitted to those historical results.

## Four frozen readouts

1. Original ProjectRL normalized plurality: count explicit abstentions and refusal-empty category; select the first seat among tied normalized answer strings. This is plurality, not a strict majority requirement.
2. Abstention-aware plurality: ignore explicit abstentions and provider refusals as candidate votes; select a substantive plurality, retaining first-seat ties. With explicit abstentions but no substantive proposal, abstain. All refusals remain refusals; all invalid outputs remain invalid.
3. Abstention-aware lexical ties: the same substantive pool, with lexically smallest normalized tied string. No semantic ranking or label fitting.
4. Conservative strict majority: a substantive answer needs at least two of the original three seats agreeing; otherwise abstain. The denominator includes unavailable seats. This sacrifices some answerable recall; it is not a proven safety guarantee, since two agents may repeat an unsupported answer.

The numeric GSM8K adapter follows its original contract: numeric parsing failures are excluded from votes and grading uses a numeric tolerance of1e-6. Thus its original vote already behaves like an availability-aware vote; no semantic abstention policy is inferred from `None`. A normalization-empty legitimate answer such as `A` remains an answer. Lexical and strict-majority rules are additional offline policies, not purported reconstructions of source originals.

## MuSiQue-Full

The [original author evaluator](https://github.com/stonybrooknlp/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/evaluate_v1.0.py) has distinct `predicted_answer`, `predicted_answerable` and `predicted_support_idxs`. Answer EM/F1 are scored on answerable records; grouped sufficiency gates the answer score using both same-question answerability decisions. Insufficient records retain a world-answer reference: matching it does not excuse predicting answerable. Our ProjectRL-style comparator is not an official MuSiQue aggregator.

The inspected mirror has4834 records,2417 paired IDs, exactly one true/false label per ID and identical question strings within pairs. Source counts range15–20, not uniformly20; all paragraph indices are unique within a record and all texts nonempty. This structural inspection does not independently certify all sufficiency labels. Original definitions and CC BY4.0 licensing are stated by the [authors](https://github.com/stonybrooknlp/musique); seed-dataset overlap cautions also apply.

Report answerable EM/F1, unsupported false-answer rate(answer attempts divided by insufficient records), valid abstention rate, coverage(answer attempts divided by all records), selective risk(errors/answered), and separate provider/invalid rates. Report `null` when a denominator is absent. Refusal or malformed output is not credited as a semantically valid abstention in custom utility. Preserve the official metrics alongside this custom decision utility.

Two independently ordered blinded packets cover48 contexts from24 disjoint question identities. Both annotators' labels are **PENDING**. {link('annotation/README.md')} specifies first-pass source sufficiency without gold, locked labels, reference validation, and adjudication. No completed human annotation or empirical unsupported false-answer estimate is claimed. The prospective selection deduplicates seven repeated question texts across different IDs, retaining2410 question-text units and2366 after the disjoint pilot/annotation reservations. Both original contexts of each retained ID remain together.
''')
    histlines=[]
    for arm in ['P_NoComm','P_SelfUpdate','P_Broadcast','R_NoComm','R_SelfUpdate','R_Broadcast']:
        histlines.append(f"- `{arm}`: "+' / '.join(str(hm[arm,r]['correct']) for r in ['original','aware','lexical','strict_majority'])+' correct out of300 (original / aware / lexical / strict majority).')
    exlines=[]
    for r in ext['models']:
        s={x['rule']:x for x in r['summary']};exlines.append(f"- `{r['model_directory']}`: original {s['original']['initial_correct']}→{s['original']['final_correct']}; aware identical; lexical {s['lexical']['initial_correct']}→{s['lexical']['final_correct']}; strict majority {s['strict_majority']['initial_correct']}→{s['strict_majority']['final_correct']} (out of500).")
    write('03_INDEPENDENT_REPLICATION_RESULTS.md',f'''
# Independent reconstruction and external checks

**The ProjectRL abstention attenuation is computationally reproduced; its generalization to independent Web QA is not established.** External numeric-QA traces show ordinary readout sensitivity, but cannot identify the proposed source-sufficiency mechanism.

## Reproduced historical finding

A new standalone implementation applies all four rules to20 E5 arms on300 identities, producing24000 question/readout records. It imports neither the earlier aggregator nor its scorer. All60 old original/aware/lexical counts match the preserved reassessment, and the original raw results remain unchanged.

{chr(10).join(histlines)}

Partitioned Broadcast–NoComm gains are+8.000pp original, +1.000pp aware,0 lexical, and+7.667pp strict majority. The aware interaction is−7.000pp, paired question-bootstrap95% CI[−10.333,−4.000]. The observed87.5% attenuation describes these two estimated gains; it is not a mediation fraction or a population estimate independently confirmed here. Against the equal-update-count SelfUpdate control, the original gain is+7.333pp and aware gain+1.333pp; the corresponding interaction is−6.000pp CI[−9.667,−2.667]. No new confirmatory p-value claim is made from these post-hoc questions.

The conservative rule yields9 NoComm and32 Broadcast correct. It preserves a large communication difference while reducing answerable accuracy relative to the aware readout. This prevents the claim that every competent readout eliminates communication benefit. This all-answerable Hotpot reference set cannot determine whether that cost buys appropriate abstention on unsupported contexts. Existing learned-policy descriptive trade-offs remain historical, not newly generalized routing superiority.

## Independent public GSM8K states

All36000 agent states across eight configurations and three rounds were inspected. Original majority, saved final numeric grade and saved tie flags reconstruct exactly for4000 model-question records. Re-extracting all raw responses with the pinned source parser gives zero disagreements with saved parsed values. There is one numeric parse failure and no empty raw response. Provider-refusal labels are unavailable. Thirteen raw states mention possible uncertainty phrases; this diagnostic search is not an independent semantic annotation and does not justify reclassifying those numeric decisions as abstentions.

{chr(10).join(exlines)}

All aware-minus-original readout-by-round interactions are exactly0 under this source contract, because it already drops parsing failures and exposes no distinct abstention class. A degenerate paired bootstrap[0,0] describes this algebraic identity; it is not a powered null interval for semantic abstention behavior.

Other rules do affect the observed communication comparison. For Qwen1.5B, strict-majority minus original changes the initial-to-final gain by+4.2pp, exploratory95% CI[+2.2,+6.2]; for Llama1B by+3.8pp[+1.2,+6.4]. These are source-specific numeric-QA contrasts, without a matched three-seat private-revision control or unsupported-question labels. They demonstrate generic aggregation sensitivity, which prior work already motivates. The eight configurations share500 identities; model-wise estimates and bootstrap samples preserve those identities, and these exploratory intervals are not multiplicity-adjusted confirmations.

{link('results/external_gsm_results.json')} includes coverage/selective risk and every model estimate. {link('results/external_gsm_question_readouts.jsonl')} contains48000 decisions. {link('results/historical_four_readouts.json')} and {link('results/historical_question_readouts.jsonl')} preserve the complete internal reanalysis. Numeric accuracy is not silently renamed exact string EM.

{link('figures/readout_sensitivity.png','Figure: historical and independent sensitivity')}

## Evidence not obtained

MuSiQue structure and licensed source availability were inspected, but there are no newly generated MuSiQue agent outcomes. Its answerable/unsupported false-answer comparison remains **NOT RUN**. No independent Web-QA readout×communication×sufficiency interaction, human sufficiency validation, model superiority, or noninferiority has been established. Same-pipeline replication and an independent benchmark data inspection do not substitute for those outcomes.
''')
    literature=[
        ('Debate or Vote','Choi, Zhu and Li','NeurIPS2025','https://arxiv.org/html/2508.17536v2','Initial vote versus communication; theoretical update assumptions; empirical debate controls','Generic debate-versus-vote question already studied'),
        ('When Debate Helps','Zhao et al.','2026preprint','https://arxiv.org/html/2610.04686v1','Proposal supply/readout separation; fixed-proposal verification probes','Readout dependence and minority recovery are not new'),
        ('Beyond Majority Voting','Ai et al.','ICML2026','https://arxiv.org/html/2510.01499v2','Reliability/dependence-aware aggregation beyond vote counts','Ordinary aggregation improvement is insufficient novelty'),
        ("Don\'t Hallucinate, Abstain",'Feng et al.','ACL2024','https://aclanthology.org/2024.acl-long.786/','Cooperative/competitive multi-LLM knowledge-gap abstention; retrieval/multihop analyses','Multi-agent abstention is established'),
        ('Sufficient Context','Joren et al.','ICLR2025','https://arxiv.org/html/2411.06037v3','Context sufficiency and selective-generation trade-offs','Reference presence and missing annotations cannot define sufficiency'),
        ('MADAM-RAG / RAMDocs','Wang et al.','COLM2025','https://arxiv.org/html/2504.13079v2','Conflicting retrieved evidence; aggregator and debate-round ablations','Generic aggregator-by-debate Web/RAG claim overlaps'),
        ('EARS','See primary preprint','2026preprint','https://arxiv.org/html/2606.18668v1','Explanatory sub-agent abstention transmitted to coordinator','Abstention as communication signal already studied'),
        ('MuSiQue-Full','Trivedi et al.','TACL2022','https://aclanthology.org/2022.tacl-1.31/','Paired same-question answerability under source removal; grouped sufficiency evaluation','Using existing contrasts is not a new benchmark'),
        ('Knowing When Not to Answer','Madhusudhan, Yadav and Lacoste','2026preprint','https://arxiv.org/html/2604.14799v1','Multimodal multi-agent abstention and accuracy/abstention trade-offs','Broad multi-agent abstention trade-off is not new'),
        ('Free-MAD','Cui et al.','FindingsACL2026','https://aclanthology.org/2026.findings-acl.1600/','Consensus-free trajectory-based decision; parsed-None filtering in inspected preprint','Do not claim None filtering or alternative debate readout is novel')]
    save('literature/primary_comparisons.json',[{'title':a,'authors':b,'venue_or_status':c,'primary_url':d,'overlap':e,'boundary':f,'reviewed_on':'2026-10-16','empirical_replication_by_us':False} for a,b,c,d,e,f in literature])
    with (ROOT/'literature/comparison_matrix.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['title','authors','venue_or_status','primary_url','overlap','boundary']);w.writeheader();w.writerows(dict(zip(w.fieldnames,r)) for r in literature)
    lits='\n\n'.join(f'- **{a}** ({c}): {e}. {f}. [Primary source]({d}).' for a,b,c,d,e,f in literature)
    write('04_RELATED_WORK_AND_NOVELTY.md',f'''
# Related work and novelty gate

**No new mechanism or general evaluation theory has been independently established.** A bounded primary-source comparison finds close precedents for proposal/readout separation, collaboration-assisted abstention, context sufficiency and aggregator/debate ablations. It does not prove that every possible intersection has already been published.

{lits}

These sources were checked through original papers or official proceedings; discovery-only search snippets were not used as findings. Preprints remain labeled preprints. The [independent novelty review]({ROOT/'reviews/novelty.md'}) gives section-level arguments. {link('literature/comparison_matrix.csv')} preserves the comparison rather than presenting paper titles as empirical replication.

The only residual candidate is narrower: **does a benchmark-compatible abstention contract materially alter the measured peer-versus-private-revision effect, and does that alteration differ between source-sufficient and source-insufficient contexts?** It would need independent data, correct source-bound answer obligations and a consequence that persists against competent prespecified aggregation. Combining familiar words in a title is not a contribution.

A credible **A, findings paper**, requires a consequential independently observed interaction or a well-supported boundary/heterogeneity result. The zero aware interaction in the external numeric contract is informative about compatibility, but does not supply the missing Web evidence. **B, protocol/benchmark**, requires validated reusable labels, ambiguous-case adjudication, independent implementations and evidence that existing protocols miss something important. This phase provides preparation, not that validation. **C, limited implementation audit**, is supported now and does not justify a WWW main-track claim by itself.

The Web connection is static Wikipedia-derived evidence and RAG-style source support. Neither GSM8K arithmetic nor simply renaming the readout creates a Web contribution. A stronger claim would require demonstrable consequences of evidence availability/provenance for source-bound answer obligations. The present material provides no live-Web result.

The independent reviewers agree that a standard aggregator removing the only substantive issue should trigger **STOP THIS NARRATIVE**, unless wider use or a distinct source-sufficiency consequence is independently demonstrated. That stop rule is retained; no further spending is justified merely to obtain a significant or favorable result.
''')
    modeltext='\n'.join(f"- `{m['checkpoint']}` weights and tokenizer revision `{m['revision']}`." for m in protocol['models'])
    write('05_FROZEN_CONFIRMATORY_DESIGN.md',f'''
# Conditional prospective design

**Prepared as a conditional offline specification; not authorized or launch-ready.** Phases1–4 leave a substantive but narrow unresolved contract/sufficiency interaction. They do not justify immediately buying a pilot. This design makes that question reviewable without pretending the current evidence establishes novelty or adequate power.

## Scope and population

Use MuSiQue-Full validation, both original source contexts per unique question. The pinned mirror SHA256 is `{mus['data_sha256']}`. Hash rank `SHA256("external-validation-v1/"+question_id)` first retains one ID per normalized question text, then locks20 pilot IDs, the next24 annotation IDs, then the first1000 confirmation IDs. The4834 records have2417 paired IDs but only2410 distinct normalized question texts. Seven duplicate-ID groups are excluded by identity alone before sampling; no model outcomes exist. The earlier specification is preserved in amendments/001_question_identity_deduplication, with the explicit pre-inference amendment. Both contexts are retained. The remaining1366 confirmation-pool identities are not silently added after looking at effects. {link('results/question_assignment.json')} and {link('frozen/question_manifest.json')} save exact IDs, source row references, packet hashes and execution orders. This is fresh relative to the inspected local Hotpot/GSM evaluation traces, not a claim that pretraining has never seen MuSiQue.

All three agents receive the same complete supplied context, in original paragraph order. This directly respects the benchmark's source-bound answerability contract. It tests transport to this context-controlled regime; it is **not a literal partitioned-source replication** of the old P result. Adding another allocation regime would require a separately costed amendment. Missing source annotations alone never determine inclusion.

{modeltext}

These are frozen proposed model artifacts, not verified deployed endpoints. Historical Azure `gpt-oss-120b` exposed no immutable snapshot and has not been proven equivalent to this pinned checkpoint. Serving runtime/weight identity and tokenizer must be resolved before execution. No model discovery or connectivity inference request was made.

## Architecture and prompts

For each context/backbone, obtain3 independent stochastic initial proposals. Clone the same three states into a3-call private-revision arm and a3-call peer-revision arm. Private sees its own response and original sources. Peer additionally sees the other two agents' parsed answerability, answer and citation-index fields, in seat order. It gets no new documents, rationales, labels or oracle selections. Both revisions start from initials, never from each other. Hash-frozen order alternates which revision arm executes first. This identifies the specified peer-message bundle relative to private revision; equal calls do not imply equal tokens or latency.

Exact system/initial/private/peer prompts and JSON fields are in {link('frozen/protocol.json')}, generated by {link('design.py')}. Temperature0.7, top_p1, max2048 output tokens including reasoning, one sample, nonstreaming, no stop string; gpt-oss reasoning effort low. Three initial draws use separate cache namespaces and no inter-seat reuse. Greedy identical prompts were rejected because they could make independent proposals trivially identical. Input cap32768; any overflow stops launch for an amendment rather than truncating or selecting favorable rows.

The parser requires explicit booleans, strings and valid source-index lists. False means expressed abstention; empty true, true `noanswer`, invalid JSON and invalid indices are failures, not repaired answers. Provider refusal remains distinct. Four offline readouts use the same final proposals: ProjectRL-style plurality comparator, aware plurality, lexical-aware and strict majority. The first comparator is not an official MuSiQue aggregator. Rule/code and prompt hashes are frozen in {link('frozen/freeze_hashes.json')}.

## Primary estimand and safeguards

Let U=1 for a substantive EM/alias-correct answer under sufficient evidence, or a valid system/expressed abstention under insufficient evidence; otherwise0. For each question q and context s,

`D(q,s) = [U(peer,aware)-U(private,aware)] - [U(peer,original)-U(private,original)]`.

Primary model-specific interaction is `E_q[(D(q,sufficient)+D(q,insufficient))/2]`. Minimum meaningful absolute effect=0.05 custom-utility units. This is chosen as a prospective practical threshold, not imported from the old−7pp estimate. Both contexts remain in one question cluster. Test model-specific means using a two-sided paired studentized mean test at Bonferroni alpha0.025 each; report97.5% question-bootstrap intervals and descriptive95% intervals,10000 draws with seed20261016. Never use a binary McNemar test for this four-cell contrast.

The average may cancel a supported-question gain against an unsupported-question loss. Therefore both D components, false-answer change, coverage/selective risk by context, and raw failure/refusal categories are mandatory. No favorable/safe conclusion is allowed from the average alone. `D(sufficient)-D(insufficient)` is a declared exploratory context interaction, not an uncorrected new primary. Report original official MuSiQue answer EM/F1 and grouped sufficiency beside custom U; they are not renamed equivalents.

Selection is independent of correctness, disagreement or beneficial outcome. Blinded semantic annotations do not remove inconvenient primary rows. Preserve benchmark-label ITT and report ambiguity sensitivity. Retain transport/parse failures; an arm with incomplete required outputs has utility0 under the declared failure convention for every readout, with separate adverse bounds for missing paired cells. Refusal is observed, not a free retry. Unknown execution halts. If incomplete-data bounds include0, do not make a confirmatory sign claim from a complete-case subset.

## Why this is not a pilot recommendation

The1000-question specification has adequate modeled power only under some variance assumptions; conservative sensitivities exceed the available pool. Author-archive equivalence, annotation/adjudication, serving/tokenizer identity, real prices and fresh inference authorization remain gates. If the literature and competent-control checks leave only an implementation defect, stop rather than executing this specification. No existing E6 allowance applies.
''')
    p50=[r for r in power['grid'] if r['minimum_meaningful_absolute_interaction']==.05 and r['target_power']==.8]
    powerlines='\n'.join(f"- Assumed SD upper value{r['SD_upper_assumption']}: normal sensitivity N={r['unique_questions']}; {r['successful_completions_two_backbones']:,} successful completions over both backbones; pool-fit={r['fits_pool']}; bounded Monte Carlo power={r.get('bounded_MC',{}).get('simulated_power','not applicable')}." for r in p50)
    budgetlines='\n'.join(f"- {r['purpose']}: {r['unique_questions']} unique questions, {r['successful_completions']:,} successes;10% failed-attempt reserve{r['failure_attempt_reserve_10pct']:,}; hard request quota{r['hard_total_attempt_quota']:,}." for r in budget['budgets'])
    write('06_API_AND_POWER_BUDGET.md',f'''
# Exact interaction power and API budget

**Spend now:0 inference calls.** The conditional experiment is not conservatively powered merely because1000 questions were selected, and no dollar ceiling or deployed version has been approved. The six unused E6 completions remain unauthorized.

## Power for the actual question-cluster contrast

{link('power_budget.py')} uses the actual average of the two context-specific four-cell interactions, with two model-specific alpha0.025 tests. It does not recycle a single paired EM discordance rate or assume the old7pp effect will reproduce. The frozen minimum meaningful effect is5pp in custom utility. With normal sensitivity, `N=ceil(((z(1-alpha/2)+z(power))*SD/delta)^2)`. The grid also varies the threshold2/5/10pp and80/90% target power.

{powerlines}

Available disjoint confirmation pool={mus['confirmation_pool']}unique questions. At SD1 or the universal bounded-contrast SD upper value2, the5pp80% design exceeds that pool. More contexts, agents, models or seeds are not additional independent questions. A1000-question study therefore has conditional, not distribution-free, power. The1000 choice rounds the951-question SD0.5 sensitivity upward, but is not justified as robustly powered without better variance information.

For each feasible sensitivity distribution,20000 Monte Carlo replicates sample the exact question-level contrast from a declared bounded distribution and apply the actual studentized mean test. This validates the contrast/test arithmetic under those distributions, not the distribution's future truth. The full support/probabilities, Monte Carlo standard errors and variance are saved in {link('results/power_sensitivity.json')}. A fixedN forecast cannot be rescued by choosing the favorable pilot effect. No noninferiority/equivalence margin is claimed.

## Architecture accounting

Each context/backbone needs3 initial+3 private+3 peer calls=9. Each unique question has two contexts:18/model. Two backbones require36N successful completions. Initial NoComm is available without an extra call, and all four readouts are offline. Additional seed replication would multiply inference demand; no such replicate is hidden in the budget. The primary already includes the additional backbone. Pilot identities are never reused as confirmation.

{budgetlines}

The20-question pilot is an operational diagnostic, not an effect-size or variance guarantee. One question pair across both models would take36 successes and exercise plumbing only; that is not a scientific pilot. The full declared operational pilot requires720. If both pilot and frozen confirmation were authorized later, they require36720 successes and40392 total attempts under separate10% reserves. A reserve is a hard spending quota, not an assumption that10% overhead guarantees completion. Failed/rejected/uncertain attempts consume the quota and any reported billed usage; undocumented execution halts, and refusals are not retried to purchase a better answer.

## Historical tokens, limits and latency

The prior physical accounting contains real referenced request usage. Deduplicating shared initials for the new9-call architecture gives, per question/context: P input5391.63 and output at least2058.723; R input12568.20 and output at least2484.29. These derive from `(SelfUpdate+Broadcast-NoComm)/300`, not summing overlapping arm totals as exclusive spend. Respectively15 and12 success-output observations are missing; lower bounds are not imputed complete means. {link('results/api_budget.json')} saves source hashes and calculations.

For the720-call pilot, P-reference input is431331 and output at least164698; R-reference input1005456 and output at least198744. These are historical arithmetic references, not MuSiQue forecasts. MuSiQue has longer source banks, a different source-access regime and a second backbone. Input sensitivities multiply these references by1,2,4 while output remains a separate lower-bound reference. Every attempted call has a2048 output-token cap and32768 input-token cap; those hard resource ceilings, including the failed-attempt reserve, are saved separately. Unavailable output/reasoning usage remains unknown.

E6 observed289.4477 summed seconds over144 successes, approximately2.01005s/call;720 calls would correspond arithmetically to1447.24 summed seconds. Neither this nor concurrent division predicts MuSiQue wall time. Measure actual per-call latency, summed latency, end-to-end wall time, prompt/output/reasoning/cache tokens, model identity, dispatches, receipts and all failures.

Current deployment, provider rate tier and immutable snapshot are unresolved. No fabricated dollar estimate is offered: cost is billed input/output tokens times the actual approved per-million rates, plus any other provider charges and required serving costs. Price/latency/token feasibility and conservative power are unresolved, so the budget does not support GO FOR SMALL PILOT today.
''')
    write('07_ADVERSARIAL_REVIEW.md',f'''
# Three independent adversarial perspectives

Three separately delegated automated reviewers produced independent written perspectives. They are not three human peer reviewers, human source annotators, or independent experimental replications. The initial reviews preceded completed external counts and the final conditional protocol, which each explicitly disclosed. Their original files are retained; any follow-up review is recorded separately.

- {link('reviews/importance.md','Importance beyond ProjectRL')}: current evidence supports category C. A broader claim needs external source-bound consequences, not another recomputation of the300 questions. Recommendation REVISE/no paid pilot.
- {link('reviews/design.md','Identification and hidden confounds')}: the averaged utility interaction can cancel opposite sufficiency effects; official MuSiQue uses dual fields and paired grouping; refusal/failure must remain distinct. Recommendation REVISE before pilot, with final-protocol review initially pending.
- {link('reviews/novelty.md','Novelty relative to primary literature')}: familiar aggregation and abstention ingredients overlap strongly with existing work. The exact unresolved interaction is a candidate only if independently consequential. Current category C; REVISE, with an explicit stopping condition.

Repairs incorporated before the final report: retain mandatory context-specific components and false-answer change; label the average as custom utility; retain original official metrics; count all-refusal as refusal rather than credited semantic abstention; add strict typed MuSiQue parsing; freeze source access and peer packet; report the lack of literal partitioned-source replication; separate conditional power from worst-case infeasibility; keep uncertainty and operational failures in the ITT record. Independent semantic annotations remain pending.

The final design reviewer verified closure of three implementation findings: the arm-level failure-utility wrapper and initial-failure branch rule, stale power artifacts, and provider-envelope type/conflict validation. Three targeted tests passed in that separate check. A later pre-inference identity check also removed seven duplicate question IDs from prospective sampling, preserving the superseded draft and adding a gold-free identity regression test. Full-suite results and historical hashes are in COMPLETION_AUDIT.json; that validation is separate from the reviewers' scientific recommendation.

## Conclusions under every required outcome

1. **Broadcast helps with matched aggregation:** report that residual benefit and uncertainty. Current conservative E5 readout already retains a+7.667pp difference. Reject a universal claim that communication gains disappear.
2. **Little incremental communication benefit:** report the interval and minimum meaningful effect. Nonsignificance does not show equivalence; a local null alone does not add novelty beyond prior vote/debate work.
3. **Aware voting helps answerable questions but harms unsupported questions:** reject a universal repair; disclose false answers and risk/coverage. The logical harm is established by a test, but the empirical prevalence and communication interaction remain unmeasured.
4. **Opposite model/dataset effects:** show model-specific signed effects and shared-question clustering. Do not select a preferred backbone or pool away reversal. External numeric initial-to-final gains already differ in sign, but that is not a source-sufficiency result.
5. **A competent standard aggregator removes the issue:** stop the general-failure/main-track narrative unless independently established prevalence or a distinct consequence remains. The external numeric source already handles parse failures, which underlines this stopping risk; it is not a powered semantic null.

All outcomes remain reportable even when they undermine the project. None by itself guarantees a publishable paper. Reviewer recommendations are evidence-limited and do not authorize paid execution.
''')
    write('08_FINAL_RESEARCH_DECISION.md',f'''
# Final research decision

**REVISE. No paid pilot now. Current evidence is category C: a useful implementation/evaluation audit, not an established WWW2027 main-track findings or benchmark contribution.** The conditional protocol is retained for review, not as an automatic experiment recommendation.

## What survives independent verification?

A separately implemented reanalysis reproduces the historical matched-readout observation on the same300 questions: partitioned Broadcast–NoComm changes from+8pp original plurality to+1pp aware plurality, with a−7pp interaction; lexical-aware yields0. The arithmetic attenuation is87.5%. With a prespecified conservative strict-majority rule, NoComm9/300 and Broadcast32/300 retain+7.667pp. Therefore communication's measured benefit depends materially on the decision contract, and ignoring abstentions is not a universally correct repair. The synthetic unsupported-question counterexample establishes a logical safety boundary, not an empirical unsupported error rate.

## Does it extend beyond this implementation?

**The specific abstention mechanism has not been independently replicated.** Eight external model configurations on500 shared GSM8K questions reconstruct exactly under their source numeric contract. Original and aware readouts coincide; alternative tie/consensus readouts produce different communication comparisons. This independently illustrates generic readout sensitivity, already familiar in the literature, but supplies neither explicit abstention labels nor unsupported Web-QA outcomes. It cannot establish the source-sufficiency interaction or a population null for it.

## Is there a meaningful Web-specific contribution, and what is new?

None is established yet. The present Web component is static Wikipedia-derived evidence. MuSiQue-Full is suitable for a source-bound check:4834 labeled records form2417 same-question ID pairs but2410 distinct normalized question-text units, with source documents and an existing sufficiency contract. Its author definitions/license/evaluator were checked; the acquired mirror matches its published hash, but author-archive byte equality remains unverified. Two independent blinded label packages are prepared and PENDING. New inference/semantic validation results are absent.

Proposal/readout separation, collaboration-assisted abstention, context-sufficiency calibration and aggregator/debate ablations all have close precedents. A potentially distinct finding would require an independently consequential peer-versus-private readout interaction that depends on source sufficiency and survives competent controls. The intersection alone is not novelty. Packaging these scripts does not create a benchmark contribution.

## Can the claim be tested with a feasible budget?

The arithmetic is explicit, but conservative feasibility is unresolved. The conditional20-question/two-backbone pilot costs720 successful completions. The locked1000-question confirmation costs36000, excluding the pilot and with separate failure reserves. A5pp interaction needs approximately951 questions at assumed SD0.5 and80% normal sensitivity, but3803 at SD1 and15209 at the universal SD upper value2; both exceed the2366-question disjoint confirmation pool. Historical token references are lower bounds where usage is missing. Actual deployment prices, immutable serving identity, tokenization and long-context latency remain unresolved. No powered or dollar-feasible promise is made.

## Smallest scientifically necessary next step

**Zero additional inference now.** The remaining gates are adjudicating the prepared blinded source-support cases, resolving original data/deployment provenance, and determining whether an independently consequential problem remains beyond competent aggregation and the closest papers. These are conditions for reconsideration, not an instruction to run more experiments. If only the local abstention implementation choice remains, choose **STOP THIS NARRATIVE** and retain the audit rather than buying a favorable result.

## Is this a credible WWW findings-paper direction?

Not on current evidence. It remains a conditional direction requiring substantive independent evidence; do not advertise a general communication failure mode, novel theory, a new benchmark, SparseComm superiority or population noninferiority. Opposite or null results, beneficial communication under competent readouts, and unsupported-question harm must all remain visible.

Offline deliverables are complete. New paid Azure completions=0; paid judges=0; PPO training=0; no historical provider ledgers or artifacts were edited. The six remaining E6 completions were not used. Preservation and test details are recorded in {link('COMPLETION_AUDIT.json')}.
''')
    write('README.md',f'''
# External validation audit

Decision: **REVISE; no paid pilot now; current contribution category C.** Read {link('08_FINAL_RESEARCH_DECISION.md')} first.

This directory contains the eight requested reports, an independent four-readout implementation, frozen-proposal analyses, a bounded public-trace audit, original benchmark contracts, blinded annotation packages, three independent automated reviews, and a conditional prospective specification. Nothing in it enables paid inference.

## Reports

1. {link('01_EXTERNAL_TRACE_AVAILABILITY.md')}
2. {link('02_ABSTENTION_AND_ANSWERABILITY_CONTRACT.md')}
3. {link('03_INDEPENDENT_REPLICATION_RESULTS.md')}
4. {link('04_RELATED_WORK_AND_NOVELTY.md')}
5. {link('05_FROZEN_CONFIRMATORY_DESIGN.md')}
6. {link('06_API_AND_POWER_BUDGET.md')}
7. {link('07_ADVERSARIAL_REVIEW.md')}
8. {link('08_FINAL_RESEARCH_DECISION.md')}

## Offline reproduction

From `/home/saurab/Documents/projectRLMAIN`, run sequentially:

```sh
python3 -m WEBExtension.causal_study.external_validation.reanalyse
python3 -m WEBExtension.causal_study.external_validation.power_budget
python3 -m WEBExtension.causal_study.external_validation.design
python3 -m WEBExtension.causal_study.external_validation.write_reports
python3 -m pytest WEBExtension/causal_study/external_validation/tests -q
```

Public source snapshots are already local. Re-fetching is unnecessary and would require network access; `fetch_public.py` performs public GETs only, never inference. Its original bounded selection and fetch receipts remain evidence. It is not a one-command complete environment installer. Full upstream repositories and prediction archives were not cloned wholesale.

After analyses/tests, `python3 -m WEBExtension.causal_study.external_validation.verify` checks preservation, source and design hashes, labels remaining pending, count alignment, report links, and emits the completion/manifest records. It never rewrites historical outputs. New tests cover meaningful failure boundaries, refusal semantics, benchmark parsing, question clustering and power assumptions.

Do not send sealed annotation/evaluation keys to annotators. Source mirror/data and externally published raw traces are local audit inputs; do not redistribute external traces without resolving their license. The source repo's lack of a visible license is not a permission grant. Generated figures are standard scientific artifacts, not interactive product UI.
''')
    figures(hist,ext)
    print('Wrote8 reports, README, source eligibility, literature matrix and scientific figures.')


def figures(hist,ext):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/projectrl-external-matplotlib')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    rules=['original','aware','lexical','strict_majority'];labels=['Original','Aware','Lexical aware','Strict majority'];x=np.arange(4);barwidth=.35
    data=[]
    for offset,arm,color in [(-.5,'P_NoComm','#46677f'),(.5,'P_Broadcast','#c46b45')]:
        values=[next(r['EM_percent'] for r in hist['metrics'] if r['arm']==arm and r['rule']==rule) for rule in rules]
        axes[0].bar(x+offset*barwidth,values,barwidth,label=arm.removeprefix('P_'),color=color)
        for xx,v in zip(x+offset*barwidth,values):axes[0].text(xx,v+.35,f'{v:.1f}',ha='center',fontsize=9)
        data += [{'scope':'ProjectRL_P','arm':arm,'rule':r,'EM_percent':v} for r,v in zip(rules,values)]
    axes[0].set_xticks(x,labels,rotation=15);axes[0].set_ylim(0,27);axes[0].set_ylabel('EM (%)');axes[0].set_title('Historical: 300 questions, partitioned evidence');axes[0].legend(frameon=False)
    for y,model in enumerate(ext['models']):
        for offset,rule,color in [(-.13,'lexical','#46677f'),(.13,'strict_majority','#c46b45')]:
            row=next(s for s in model['summary'] if s['rule']==rule)['readout_by_round_interaction_vs_original'];m=row['mean_pp'];lo,hi=row['ci95_pp']
            axes[1].errorbar(m,y+offset,xerr=[[m-lo],[hi-m]],fmt='o',color=color,capsize=3,label={'lexical':'Lexical aware','strict_majority':'Strict majority'}[rule] if y==0 else None)
            data.append({'scope':'External_GSM8K','model':model['model_directory'],'rule':rule,**row})
    names={'05b':'Qwen 0.5B','15b':'Qwen 1.5B','3b':'Qwen 3B','7b':'Qwen 7B','gemma1b':'Gemma 1B','gemma4b':'Gemma 4B','llama1b':'Llama 1B','llama3b':'Llama 3B'}
    axes[1].set_yticks(range(8),[names[r['model_directory']] for r in ext['models']]);axes[1].axvline(0,color='#999999',linewidth=1)
    axes[1].set_xlabel('Readout-by-round interaction vs original (pp; 95% CI)');axes[1].set_title('External: 500 shared questions (exploratory)');axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(folder/'readout_sensitivity.png',dpi=180);fig.savefig(folder/'readout_sensitivity.svg');plt.close(fig)
    save('figures/plotted_observations.json',data)
if __name__=='__main__':main()
