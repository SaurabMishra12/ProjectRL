"""Bounded consultation. Observed support assertions are not semantic guarantees."""
from collections import Counter
from dataclasses import replace
import json
from .packets import Packet,canonical,pack_control
from .policies import one_round_schedule,Schedule,select_senders
from .runner import control_packets,update,ArmResult


def stop_fields(status,answer,valid_fact_count,needs_count,conflicts_count):
    return (status=='supported' and isinstance(answer,str) and bool(answer.strip()) and
            valid_fact_count>0 and needs_count==0 and conflicts_count==0)


def should_stop(state):
    p=state.outcome.proposal
    return state.outcome.kind=='valid' and stop_fields(p.status,p.answer,len(p.facts),len(p.needs),len(p.conflicts))


def validate_transmission_budget(transmissions,n_agents,packet_cap,metadata_cap,max_control_frames=6):
    data=[p for p in transmissions if p.kind!='control'];counts=Counter()
    if len(data)>3*(n_agents-1):raise ValueError('linear_delivery_ceiling_exceeded')
    for packet in transmissions:
        if packet.tokens<0 or packet.tokens>(metadata_cap if packet.kind=='control' else packet_cap):raise ValueError('packet_ceiling_exceeded')
        if packet.kind=='control':
            node=packet.sender if packet.sender>=0 else packet.recipient
            if not 0<=node<n_agents:raise ValueError('invalid_control_owner')
            counts[node]+=1
    if any(x>max_control_frames for x in counts.values()):raise ValueError('control_frame_ceiling_exceeded')


def run_adaptive(public,initial,sink,backend,config,policy='selected',always_run=False):
    always_run=always_run or not config.stopping
    n=len(initial);metadata,controls=control_packets(initial,backend.tokenizer,config.metadata_cap)
    start=len(backend.references);transmissions=list(controls);states=initial
    def check(extra=()):
        validate_transmission_budget(tuple(transmissions)+tuple(extra),n,config.packet_cap,config.metadata_cap,config.max_control_frames)
    def result(reason):
        check();return ArmResult('adaptive_'+policy,states,sink,states[sink].outcome,
                                 tuple(backend.references[start:]),tuple(transmissions),reason)
    def notices(recipients,phase,dense=False):
        frames=[]
        for recipient in recipients:
            text=pack_control(recipient,{'phase':phase},config.metadata_cap,backend.tokenizer)
            frames.append(Packet(-1,recipient,text,backend.tokenizer.count(text),kind='control'))
        if not dense:check(frames)
        transmissions.extend(frames)
    if not always_run and should_stop(states[sink]):return result('supported_stop')
    if policy=='private':
        # Fixed maximum private schedule; its outcomes cannot change the schedule.
        leaves=tuple(i for i in range(n) if i!=sink)[:max(1,n//2)]
        for recipients in ((sink,),leaves,(sink,))[:config.max_phases]:
            notices(recipients,'private')
            states,_,_=update(public,states,Schedule((),recipients),backend,config,'adaptive_private')
        return result('phase_limit')
    if policy=='dense':
        # Dense is a separate comparator; its quadratic schedule is not passed to the sparse-bound validator.
        for _ in range(config.max_phases):
            notices(tuple(range(n)),'dense',dense=True)
            states,packets,_=update(public,states,one_round_schedule('dense',n,sink,()),backend,config,'adaptive_dense')
            transmissions.extend(packets)
        return ArmResult('adaptive_dense',states,sink,states[sink].outcome,tuple(backend.references[start:]),tuple(transmissions),'phase_limit')
    senders=tuple(i for i in range(n) if i!=sink) if policy=='instar' else select_senders(policy,metadata,sink,max(1,n//2),public.question,config.seed,set())
    schedule=one_round_schedule('instar' if policy=='instar' else 'selected',n,sink,senders)
    check([Packet(i,sink,'',config.packet_cap) for i in senders])
    notices((sink,),'gather')
    states,packets,_=update(public,states,schedule,backend,config,'adaptive_gather')
    transmissions.extend(packets)
    if not always_run and should_stop(states[sink]):return result('supported_stop')
    if config.max_phases<3:return result('phase_limit')
    p=states[sink].outcome.proposal
    needs=[{'entity':x.entity,'relation':x.relation} for x in p.needs] if p else []
    if not needs and not always_run:return result('unresolved_no_request')
    # The always-run diagnostic has a declared fixed verification request when no model request exists.
    request=canonical({'needs':needs}) if needs else canonical({'verify_question':public.question})
    request_packets=[]
    for leaf in senders:
        wire=canonical({'s':sink,'r':leaf,'request':json.loads(request)})
        request_packets.append(Packet(sink,leaf,wire,backend.tokenizer.count(wire),kind='request'))
    try:check(request_packets+[Packet(leaf,sink,'',config.packet_cap,kind='reply') for leaf in senders])
    except ValueError:return result('budget_stop')
    transmissions.extend(request_packets)
    notices(senders,'request')
    states,_,_=update(public,states,Schedule((),senders),backend,config,'adaptive_request',request=request,stage='request')
    notices((sink,),'reply')
    states,packets,_=update(public,states,schedule,backend,config,'adaptive_reply',request=request,kind='reply')
    transmissions.extend(packets)
    return result('supported_stop' if should_stop(states[sink]) else 'unresolved')
