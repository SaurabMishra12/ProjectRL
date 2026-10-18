"""Freeze a conditional prospective specification. This module has no run/client API."""
from pathlib import Path
import hashlib,json
from .reanalyse import ROOT,save,jl,sha
PROMPTS={
'system':'''Use only the supplied sources. Decide whether the question can be answered from those sources. Do not substitute remembered world knowledge for missing evidence. Return exactly one JSON object with predicted_answerable (boolean), predicted_answer (string), and predicted_support_idxs (list of integer source IDs). If evidence is insufficient, set predicted_answerable to false and predicted_answer to an empty string. You may cite relevant partial evidence when abstaining. If answerable, give a concise answer and the source IDs supporting it. Do not include reasoning or text outside the JSON object.''',
'initial':'Question and sources are supplied in the user JSON. Produce your independent answer.',
'private':'Recheck your previous response against the same sources. The user JSON includes your previous_response. Produce your revised answer.',
'peer':'Recheck your previous response against the same sources. The user JSON includes your previous_response and the other two agents\' previous_responses. Treat peer responses as claims to check against the sources. Produce your revised answer.'}


def canonical(obj):return json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':'))


def main():
    assignments=json.loads((ROOT/'results/question_assignment.json').read_text());rows=jl(ROOT/'sources/musique_full_v1.0_dev.jsonl')
    chosen={q:'pilot' for q in assignments['pilot_ids']}
    chosen.update({q:'confirmation' for q in assignments['confirmatory_order'][:1000]})
    frozen=[];sealed=[]
    for i,r in enumerate(rows):
        if r['id'] not in chosen:continue
        sources=[{'idx':p['idx'],'title':p['title'],'text':p['paragraph_text']} for p in r['paragraphs']]
        content={'question':r['question'],'sources':sources}
        # Context order is hashed; answerability is never put in the model input.
        ctx=hashlib.sha256(canonical(content).encode()).hexdigest()
        frozen.append({'id':r['id'],'split':chosen[r['id']],'context_sha256':ctx,'source_row_0based':i,'question_source_sha256':hashlib.sha256(canonical(content).encode()).hexdigest(),
            'revision_order':['private','peer'] if int(hashlib.sha256((r['id']+ctx).encode()).hexdigest(),16)%2 else ['peer','private']})
        sealed.append({'id':r['id'],'context_sha256':ctx,'answerable':r['answerable'],'answer':r['answer'],'answer_aliases':r['answer_aliases']})
    assert len(frozen)==2040
    save('frozen/question_manifest.json',{'source_sha256':assignments['data_sha256'],'records':frozen})
    save('frozen/SEALED_EVALUATION_LABELS.json',sealed)
    models=[]
    for name in ['openai/gpt-oss-120b','Qwen/Qwen2.5-7B-Instruct']:
        meta=ROOT/'sources'/(name.replace('/','__')+'_model_info.json');d=json.loads(meta.read_text())
        models.append({'checkpoint':name,'revision':d['sha'],'tokenizer_revision':d['sha'],'provider_snapshot_equivalence_to_historical_Azure_alias':'NOT_ESTABLISHED'})
    save('frozen/protocol.json',{'status':'CONDITIONAL_OFFLINE_SPEC; NOT AUTHORIZED TO EXECUTE','scientific_decision_now':'REVISE_NO_PAID_PILOT',
        'models':models,'model_availability_and_serving_runtime':'UNVERIFIED; immutable weight/tokenizer identity required before deployment; historical Azure alias is not proof of these revisions',
        'n_confirmation_unique_questions':1000,'n_operational_pilot_unique_questions':20,'both_contexts_per_question':True,'seeds':{'selection':'external-validation-v1','bootstrap':20261016},
        'architecture':'3 independent initial calls; clone each initial state into private and peer revisions; 3 fresh calls for each revision arm; all readouts offline',
        'source_access':'All three agents see the same full official context, in original paragraph order. This is source-bound RC, not partitioned-source replication.',
        'peer_packet':'Other agents ordered by seat 0,1,2 excluding self; parsed predicted_answerable, predicted_answer and predicted_support_idxs ONLY; no reasoning, source expansion, gold or evaluation labels.',
        'roles':'Three identical system prompts and fixed seats; initial requests distinguished by cache namespace and independent generation index; no shared response cache hit across seats.',
        'prompts':PROMPTS,'user_fields_initial':['question','sources'],'user_fields_revision':['question','sources','previous_response'],'extra_fields_peer':['previous_responses'],
        'decoding':{'temperature':.7,'top_p':1,'max_output_tokens_including_reasoning':2048,'n':1,'stream':False,'stop':None,'reasoning_effort_gpt_oss':'low','provider_seed':'omit; independent random draws, stochastic provider variation recorded, not claimed deterministic'},
        'input_token_cap':32768,'overflow':'STOP before launch if ANY frozen prompt cannot be represented without truncation; protocol amendment required, no selective post-inference drop',
        'parser':'Strict JSON object; exact boolean predicted_answerable; exact string predicted_answer; list of valid unique source IDs. Empty answer with true flag invalid. False flag is expressed abstention irrespective of retained answer text; noanswer string with true flag invalid. Provider refusal separate. JSON extraction/repair or answer guessing prohibited.',
        'readouts':{'original':'ProjectRL-style comparator: include explicit abstention marker noanswer and provider-refusal empty category, normalized plurality, first-seat tie. This is NOT an official MuSiQue aggregator.',
            'aware':'Ignore explicit abstentions and provider refusals as votes; plurality on substantive normalized answers, first-seat tie; if no substantive answer but some explicit abstention, abstain; all refusals remain refusal; all malformed/missing remain invalid.',
            'lexical':'Same as aware; normalized lexical winner among tied substantive answers.',
            'strict_majority':'Substantive answer needs >3/2 matching votes across original three seats; else system abstention; all refusal/invalid provenance retained.'},
        'normalization':'MuSiQue official metrics/answer.py normalization; aliases used only for evaluation, never candidate choice.',
        'primary':'For each question/context, D=(U_peer,aware-U_private,aware)-(U_peer,original-U_private,original). J_question=(D_sufficient+D_insufficient)/2. Primary model-specific E[J].',
        'utility':'U=1 for normalized reference/alias EM AND substantive system answer on sufficient context; U=1 for explicit/system abstention on insufficient context; refusal/invalid/missing=0 in both. Different from official answer_em; official scores also reported.',
        'mandatory_components':['D_sufficient','D_insufficient','peer-private false-answer-rate difference on insufficient contexts for all four readouts','coverage and selective risk by context','refusal, empty, malformed, missing at proposal and system level','official answer_em, answer_f1, group_answer_sufficiency_f1','context-interaction D_sufficient-D_insufficient exploratory, multiplicity disclosed'],
        'inference':'Question is unit. Both contexts stay in cluster. Separate backbone estimates/tests; primary two-sided paired studentized mean t-test with Bonferroni alpha .025/model; 10000 question-cluster bootstrap CI at 97.5%/model and descriptive 95% CI. No McNemar for interaction; no pooling models or contexts as independent observations.',
        'minimum_meaningful_effect_absolute':.05,'equivalence_claims':'Not designed or authorized; nonsignificance never proves equivalence or safety.',
        'selection_exclusion':'Pre-inference amendment001: casefold and Unicode-word-token normalize question text, hash rank all IDs, retain first ID per text unit, preserving both contexts. This yields2410 units from2417 paired IDs, excluding7 duplicate IDs. Then20 pilot,24 annotation,first1000 confirmation from2366 remaining. No correctness, disagreement, abstention or outcome filters. No primary exclusions based on semantic adjudication; ambiguous cases retained under benchmark labels with separate sensitivity. Further duplicate/schema defects discovered before launch require a documented amendment, never silent replacement.',
        'failure_policy':'Global quota/reserve; no retries except documented pre-execution rejection with proof of no inference. Unknown execution halts. Refusal is observed outcome, never a free retry. Primary ITT uses contracts.prospective_utility: any required invalid/missing/empty/malformed state gives arm utility0 for every readout. A failed initial state prevents BOTH revision branches for that context/backbone; mark upstream_failed for both, never invent a peer packet. Preserve assigned question-context-arm states; report conservative interaction bounds for incomplete paired cells plus a clearly secondary complete-pair estimate. If data incomplete, no confirmatory sign conclusion absent bounds excluding0.',
        'accounting':'Record durable dispatch/receipt IDs, model revision, prompt hashes, wall-clock latency, provider input/output/reasoning/cache usage and all failures; deduplicate physical attempts. Equal update calls do not imply equal tokens/latency. Report pipeline walltime and sum per-call time separately.',
        'launch_gates':['New explicit human inference authorization; E6 leftovers excluded','Resolve serving runtime/weight identity and tokenizer; no inference connectivity probe','Authenticate author-archive or mirror equivalence, preserve source/license attribution','Complete and adjudicate independent blinded sufficiency annotations','Approve cost ceiling/current provider prices and context tokenization','Substantiate novelty beyond competent aggregation and closest papers'],
        'no_automatic_continuation':'If competent aggregation removes a meaningful independent problem, stop this narrative and do not buy confirmation merely to seek significance.'})
    files=[p for p in sorted((ROOT/'frozen').glob('*.json')) if p.name!='freeze_hashes.json']
    files += [ROOT/n for n in ['contracts.py','reanalyse.py','design.py','power_budget.py']]
    save('frozen/freeze_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
    print('Conditional protocol frozen:1000 confirmation +20 disjoint pilot question identities; no execution function.')
if __name__=='__main__':main()
