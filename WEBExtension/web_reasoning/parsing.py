"""Strict source-visible output parsing; unsuccessful outputs never become predictions."""
import hashlib
import json
from .contracts import Proposal,Outcome,Fact,Need
from .prompts import schema


def strict_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate_json_key')
        result[key]=value
    return result


def validate_schema(value,spec):
    if 'enum' in spec and value not in spec['enum']:raise ValueError('unknown_enum')
    types=spec.get('type');types=[types] if isinstance(types,str) else types
    allowed={'object':dict,'array':list,'string':str,'integer':int,'null':type(None)}
    if types and not any(type(value) is allowed[t] for t in types):raise ValueError('incorrect_field_type')
    if isinstance(value,dict):
        props=spec['properties']
        if set(value)!=set(spec['required']):raise ValueError('unexpected_or_missing_fields')
        for key,v in value.items():validate_schema(v,props[key])
    if isinstance(value,list):
        if len(value)>spec['maxItems']:raise ValueError('list_limit')
        for v in value:validate_schema(v,spec['items'])
    if type(value) is int and value<spec.get('minimum',value):raise ValueError('invalid_index')


def parse_outcome(raw,visible_atoms,dataset,variant='enhanced'):
    digest=hashlib.sha256(json.dumps(raw,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    usage=raw.get('usage') if isinstance(raw,dict) else None
    try:
        choice=raw['choices'][0];message=choice['message']
        if choice.get('finish_reason')=='content_filter' or message.get('refusal'):
            return Outcome('refusal',raw_sha256=digest,usage=usage,reason='explicit_provider_refusal')
        if choice.get('finish_reason') not in ('stop',None):raise ValueError('unfinished_output')
        obj=json.loads(message['content'],object_pairs_hook=strict_object)
        if variant=='legacy':
            if dataset!='2wiki' or not isinstance(obj,dict) or set(obj)!={'answer','citations'} or not isinstance(obj['answer'],str) or not obj['answer'].strip() or not isinstance(obj['citations'],list):
                raise ValueError('invalid_legacy_output')
            lookup={(a.title,a.sentence_id):a for a in visible_atoms};facts=[]
            for c in obj['citations']:
                if not isinstance(c,list) or len(c)!=2 or not isinstance(c[0],str) or type(c[1]) is not int or tuple(c) not in lookup:raise ValueError('invalid_legacy_citation')
                a=lookup[tuple(c)];facts.append({'fact_id':a.atom_id,'document_id':a.document_id,'sentence_id':a.sentence_id,'quote':a.text,'claim':''})
            answer=None if obj['answer'].strip().casefold()=='noanswer' else obj['answer'].strip()
            obj={'status':'supported' if answer else ('partial' if facts else 'no_evidence'),'answer':answer,'facts':facts,
                 'entity_tags':[],'needs':[],'derivation':[],'conflicts':[]}
        elif variant!='enhanced':raise ValueError('unknown_prompt_variant')
        validate_schema(obj,schema(dataset))
        lookup={a.atom_id:a for a in visible_atoms};facts=[];seen=set()
        for entry in obj['facts']:
            atom=lookup.get(entry['fact_id'])
            if atom is None or entry['document_id']!=atom.document_id or entry['sentence_id']!=atom.sentence_id or entry['quote']!=atom.text:
                raise ValueError('invalid_or_invisible_source')
            if atom.atom_id in seen:raise ValueError('duplicate_fact_id')
            if len(entry['claim'].split())>24:raise ValueError('claim_limit')
            seen.add(atom.atom_id);facts.append(Fact(atom.atom_id,atom.text,entry['claim']))
        for need in obj['needs']:
            if any(len(v.split())>12 for v in need.values()):raise ValueError('need_limit')
        for step in obj['derivation']+obj['conflicts']:
            if any(i not in seen for i in step['inputs']):raise ValueError('unknown_derivation_fact')
            if any(len(v.split())>24 for k,v in step.items() if k!='inputs'):raise ValueError('derivation_limit')
        if dataset=='finqa':
            for mapping in obj['operand_sources']:
                if not mapping['fact_ids'] or any(i not in seen for i in mapping['fact_ids']):raise ValueError('hidden_operand_source')
        proposal=Proposal(obj['status'],obj['answer'],tuple(facts),tuple(obj['entity_tags']),
                          tuple(Need(**n) for n in obj['needs']),
                          tuple(json.dumps(s,sort_keys=True) for s in obj['derivation']),
                          tuple(json.dumps(s,sort_keys=True) for s in obj['conflicts']),
                          tuple(obj.get('program',())),tuple(json.dumps(s,sort_keys=True) for s in obj.get('operand_sources',())))
        return Outcome('valid',proposal,digest,usage)
    except (ValueError,TypeError,KeyError,IndexError) as error:
        return Outcome('invalid',raw_sha256=digest,usage=usage,reason=str(error))
