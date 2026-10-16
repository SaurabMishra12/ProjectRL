"""Complete source atoms on the wire; receipt objects cannot expand hidden sources."""
from dataclasses import dataclass, replace
import json
import re
from .contracts import Atom,Fact
from .evidence import merge_ledger


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def tokens(text):return set(re.findall(r'\w+',str(text).casefold()))


@dataclass(frozen=True)
class Packet:
    sender:int
    recipient:int
    text:str
    tokens:int
    facts:tuple[Fact,...]=()
    omitted_atom_ids:tuple[str,...]=()
    unavailable:bool=False
    kind:str='evidence'


def pack_evidence(state,recipient,request,cap,tokenizer,include_answer=True,question='',kind='evidence',mode='atomic'):
    visible={a.atom_id:a for a in state.visible_atoms}
    for f in state.ledger:
        if f.atom_id not in visible or visible[f.atom_id].text!=f.quote:raise ValueError('sender_does_not_own_fact')
    need=tokens(request or '');query=tokens(question)
    order=sorted(enumerate(state.ledger),key=lambda pair:(-len(tokens(pair[1].quote)&need),-len(tokens(pair[1].quote)&query),pair[0]))
    proposal=state.outcome.proposal
    envelope={'s':state.agent_id,'r':recipient,'status':proposal.status if proposal else state.outcome.kind,
              'answer':proposal.answer if proposal and include_answer else None,'facts':[]}
    if mode=='prefix':
        envelope['facts']=[{'source':visible[f.atom_id].to_public_dict(),'claim':f.claim} for _,f in order]
        text=tokenizer.prefix(canonical(envelope),cap)
        entries=complete_prefix_entries(text)
        facts=tuple(Fact(x['source']['atom_id'],x['source']['text'],x['claim']) for x in entries)
        return Packet(state.agent_id,recipient,text,tokenizer.count(text),facts,
                      tuple(f.atom_id for _,f in order if f.atom_id not in {x.atom_id for x in facts}),not bool(facts),'prefix')
    if mode!='atomic':raise ValueError('unknown_packet_mode')
    if tokenizer.count(canonical(envelope))>cap:
        envelope['answer']=None
    if tokenizer.count(canonical(envelope))>cap:raise ValueError('packet_cap_below_minimum_envelope')
    included=[];omitted=[]
    for _,fact in order:
        atom=visible[fact.atom_id];entry={'source':atom.to_public_dict(),'claim':fact.claim}
        candidate={**envelope,'facts':envelope['facts']+[entry]}
        if tokenizer.count(canonical(candidate))>cap and envelope['answer'] is not None:
            candidate['answer']=None;envelope['answer']=None
        if tokenizer.count(canonical(candidate))<=cap:
            envelope=candidate;included.append(fact)
        else:omitted.append(fact.atom_id)
    text=canonical(envelope)
    return Packet(state.agent_id,recipient,text,tokenizer.count(text),tuple(included),tuple(omitted),not bool(included),kind)


def apply_delivery(receiver,packet):
    if packet.recipient!=receiver.agent_id:raise ValueError('wrong_recipient')
    if packet.kind=='prefix':body={'r':packet.recipient,'s':packet.sender,'facts':complete_prefix_entries(packet.text)}
    else:body=json.loads(packet.text)
    if body['r']!=packet.recipient or body['s']!=packet.sender:raise ValueError('packet_identity_mismatch')
    atoms=[];facts=[]
    for entry in body['facts']:
        source=entry['source'];source={**source,'context':tuple(source['context'].items())}
        atom=Atom(**source);atoms.append(atom);facts.append(Fact(atom.atom_id,atom.text,entry['claim']))
    if tuple(facts)!=packet.facts:raise ValueError('packet_receipt_mismatch')
    visible={a.atom_id:a for a in receiver.visible_atoms}
    for atom in atoms:
        if atom.atom_id in visible and atom!=visible[atom.atom_id]:raise ValueError('source_identity_changed')
        visible[atom.atom_id]=atom
    return replace(receiver,visible_atoms=tuple(visible.values()),ledger=merge_ledger(receiver.ledger,facts))


def complete_prefix_entries(text):
    marker='"facts":[';start=text.find(marker)
    if start<0:return []
    offset=start+len(marker);decoder=json.JSONDecoder();entries=[]
    while offset<len(text):
        try:entry,end=decoder.raw_decode(text,offset)
        except ValueError:break
        if not isinstance(entry,dict) or set(entry)!={'source','claim'}:break
        entries.append(entry);offset=end
        if offset>=len(text) or text[offset]!=',':break
        offset+=1
    return entries


def pack_control(agent_id,payload,cap,tokenizer):
    value={'agent':agent_id,**payload}
    while tokenizer.count(canonical(value))>cap:
        candidates=[k for k,v in value.items() if isinstance(v,list) and v]
        if not candidates:raise ValueError('control_cap_below_minimum_envelope')
        key=candidates[-1];value[key]=value[key][:-1]
    return canonical(value)
