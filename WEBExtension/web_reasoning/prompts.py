"""Identical task contracts across routing arms; all variable content is user data."""
from dataclasses import asdict, is_dataclass
import json
from .prompt_text import COMMON, INITIAL, SYNTHESIS, REQUEST, FINQA


def schema(dataset):
    string={'type':'string'}
    def obj(properties,required=None):
        return {'type':'object','properties':properties,'required':list(properties) if required is None else required,'additionalProperties':False}
    def arr(items,maximum):return {'type':'array','items':items,'maxItems':maximum}
    fact=obj({'fact_id':string,'document_id':string,'sentence_id':{'type':'integer','minimum':0},'quote':string,'claim':string})
    props={'status':{'enum':['supported','partial','no_evidence','conflict']},'answer':{'type':['string','null']},
           'facts':arr(fact,4),'entity_tags':arr(string,8),
           'needs':arr(obj({'entity':string,'relation':string}),2),
           'derivation':arr(obj({'inputs':arr(string,8),'operation':string,'result':string}),3),
           'conflicts':arr(obj({'inputs':arr(string,8),'issue':string}),2)}
    if dataset=='finqa':
        props.update(program=arr(string,129),operand_sources=arr(obj({'operand':string,'fact_ids':arr(string,8)}),64))
    return obj(props)


def build_messages(question,visible_atoms,own_state,incoming_packets,request,dataset,stage,variant='enhanced'):
    if dataset not in ('2wiki','finqa') or stage not in ('initial','synthesis','request'):
        raise ValueError('unsupported_prompt')
    # Only a previous Proposal is permitted; AgentState would expose internal request/usage data.
    previous=own_state
    if is_dataclass(own_state):
        lookup={a.atom_id:a for a in visible_atoms}
        previous={'status':own_state.status,'answer':own_state.answer,
                  'facts':[{'fact_id':f.atom_id,'document_id':lookup[f.atom_id].document_id,
                            'sentence_id':lookup[f.atom_id].sentence_id,'quote':f.quote,'claim':f.claim}
                           for f in own_state.facts],
                  'entity_tags':list(own_state.entity_tags),'needs':[asdict(n) for n in own_state.needs],
                  'derivation':[json.loads(x) for x in own_state.derivation],
                  'conflicts':[json.loads(x) for x in own_state.conflicts]}
        if dataset=='finqa':
            previous.update(program=list(own_state.program),operand_sources=[json.loads(x) for x in own_state.operand_sources])
    if previous is not None and (not isinstance(previous,dict) or set(previous)-set(schema(dataset)['properties'])):
        raise ValueError('invalid_previous_state')
    if variant not in ('enhanced','legacy'):raise ValueError('unknown_prompt_variant')
    output_schema=schema(dataset)
    if variant=='legacy':
        output_schema={'type':'object','properties':{'answer':{'type':'string'},'citations':{'type':'array'}},'required':['answer','citations'],'additionalProperties':False}
    data={'question':question,'visible_sources':[a.to_public_dict() for a in visible_atoms],
          'own_previous_state':previous,'incoming_packets':[p.text if hasattr(p,'text') else p for p in incoming_packets],
          'request':request,'output_schema':output_schema}
    task={'initial':INITIAL,'synthesis':SYNTHESIS,'request':REQUEST}[stage]
    system=COMMON+'\n\n'+task+'\nUse the visible atom_id as fact_id. Copy its document_id and sentence_id exactly.'
    if dataset=='finqa':system+='\n\n'+FINQA
    if variant=='legacy':
        if dataset!='2wiki':raise ValueError('legacy_style_ablation_is_text_only')
        system=('Answer the multi-hop question using supplied Wikipedia evidence. Treat peer claims as unverified. '
                'Return only JSON with answer (a concise nonempty string, or "noanswer" if unknown) and citations '
                '(a list of [document title, zero-based sentence index] pairs). Do not invent citations or add explanations.')
    return [{'role':'system','content':system},{'role':'user','content':json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(',',':'))}]
