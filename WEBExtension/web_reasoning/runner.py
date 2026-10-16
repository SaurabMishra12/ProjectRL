"""Frozen-state comparisons. Targets cannot enter this module's call signatures."""
from dataclasses import dataclass,replace,field
import hashlib
import json
from pathlib import Path
import time
import tracemalloc
from .contracts import AgentState
from .backend import Request
from .prompts import build_messages
from .evidence import merge_ledger
from .packets import Packet,apply_delivery,pack_evidence,canonical,pack_control
from .policies import metadata_frame,select_senders,one_round_schedule
from .storage import digest


@dataclass(frozen=True)
class ArmResult:
    name:str
    states:tuple[AgentState,...]
    sink:int
    outcome:object
    references:tuple=()
    transmissions:tuple=()
    stop_reason:str='fixed_depth'


@dataclass(frozen=True)
class QuestionRun:
    question_key:str
    dataset:str
    n_agents:int
    initial:tuple[AgentState,...]
    arms:dict
    logical_refs:tuple
    transmissions:tuple
    mode:str
    contract_hash:str=''
    timings:dict=field(default_factory=dict)
    policy_metadata:dict=field(default_factory=dict)
    cohort:str=''


def call(public,visible,previous,incoming,request,stage,backend,config,slot):
    messages=build_messages(public.question,visible,previous,incoming,request,public.dataset,stage,variant=config.prompt_variant)
    req=Request(messages,{'model':config.model,'revision':config.deployment_revision,'base_url':config.provider_base_url},backend.tokenizer.identity,
                {'dataset':public.dataset,'max_tokens':config.max_output_per_call,'temperature':0,
                 'prompt_version':config.prompt_version,'schema_version':config.schema_version,'replicate_id':config.replicate_id,
                 'prompt_variant':config.prompt_variant})
    return backend.complete(req,{'question':public.key,**slot})


def initialize(public,assignments,backend,config):
    result=[]
    for i,atoms in enumerate(assignments):
        outcome=call(public,atoms,None,(),None,'initial',backend,config,{'arm':'initial','stage':'initial','node':i})
        result.append(AgentState(i,tuple(atoms),outcome.proposal.facts if outcome.proposal else (),outcome))
    return tuple(result)


def incoming_summary(packet):
    if packet.kind=='prefix':return {'raw_prefix':packet.text,'facts':[]}
    body=json.loads(packet.text)
    return {**body,'facts':[{'fact_id':entry['source']['atom_id'],'claim':entry['claim']} for entry in body['facts']]}


def update(public,initial,schedule,backend,config,name,request=None,stage='synthesis',kind='evidence'):
    states=list(initial);packets=[];start=len(backend.references)
    for recipient in schedule.recipients:
        receiver=initial[recipient];incoming=[]
        for sender,target in schedule.edges:
            if target!=recipient:continue
            packet=pack_evidence(initial[sender],recipient,request,config.packet_cap,backend.tokenizer,
                                 question=public.question,kind=kind,include_answer=config.include_answer,mode=config.packet_mode)
            receiver=apply_delivery(receiver,packet);incoming.append(incoming_summary(packet));packets.append(packet)
        outcome=call(public,receiver.visible_atoms,receiver.outcome.proposal,tuple(incoming),request,stage,backend,config,
                     {'arm':name,'stage':stage,'node':recipient})
        ledger=merge_ledger(receiver.ledger,outcome.proposal.facts if outcome.proposal else ())
        states[recipient]=replace(receiver,ledger=ledger,outcome=outcome)
    return tuple(states),tuple(packets),tuple(backend.references[start:])


def control_packets(initial,tokenizer,cap):
    metadata=[];packets=[]
    for state in initial:
        item,text=metadata_frame(state,tokenizer,cap);metadata.append(item)
        packets.append(Packet(state.agent_id,-1,text,tokenizer.count(text),kind='control'))
    return tuple(metadata),tuple(packets)


def run_w1(public,assignments,sink,backend,config):
    started=time.perf_counter();cpu_started=time.process_time();routing_wall=routing_cpu=0.;routing_peak=0
    n=len(assignments)
    if n not in (2,3,4,5) or not 0<=sink<n:raise ValueError('invalid_team')
    expected={a.atom_id for source in public.sources for a in source.atoms}
    if {a.atom_id for shard in assignments for a in shard}!=expected:raise ValueError('assignment_union_mismatch')
    start=len(backend.references);initial=initialize(public,assignments,backend,config);before=digest(initial)
    metadata,controls=control_packets(initial,backend.tokenizer,config.metadata_cap)
    arms={'nocomm':ArmResult('nocomm',initial,sink,initial[sink].outcome,(),controls)}
    for name in ('private','static','random','selected','instar','dense'):
        senders=()
        if name in ('static','random','selected'):
            owned_trace=not tracemalloc.is_tracing()
            if owned_trace:tracemalloc.start()
            wall=time.perf_counter();cpu=time.process_time()
            try:
                senders=route_senders(name,metadata,sink,max(1,n//2),public.question,config,set())
                routing_wall+=time.perf_counter()-wall;routing_cpu+=time.process_time()-cpu
                routing_peak=max(routing_peak,tracemalloc.get_traced_memory()[1])
            finally:
                if owned_trace:tracemalloc.stop()
        schedule=one_round_schedule(name,n,sink,senders)
        states,packets,refs=update(public,initial,schedule,backend,config,name)
        # Routing notices are explicit charged control frames, not model prompts.
        notices=[]
        for recipient in schedule.recipients:
            text=pack_control(recipient,{'senders':[s for s,t in schedule.edges if t==recipient]},config.metadata_cap,backend.tokenizer)
            notices.append(Packet(-1,recipient,text,backend.tokenizer.count(text),kind='control'))
        arms[name]=ArmResult(name,states,sink,states[sink].outcome,refs,controls+tuple(notices)+packets)
    all_atoms=tuple(a for source in public.sources for a in source.atoms)
    ref_start=len(backend.references)
    full=call(public,all_atoms,None,(),None,'initial',backend,config,{'arm':'full_union','stage':'initial','node':sink})
    fullstate=AgentState(sink,all_atoms,full.proposal.facts if full.proposal else (),full)
    arms['full_union']=ArmResult('full_union',(fullstate,),sink,full,tuple(backend.references[ref_start:]))
    if digest(initial)!=before:raise ValueError('initial_state_mutated')
    return QuestionRun(public.key,public.dataset,n,initial,arms,tuple(backend.references[start:]),
                       tuple(p for arm in arms.values() for p in arm.transmissions),config.mode,
                       contract_hash(public,assignments,sink,config,backend.tokenizer.identity),
                       {'client_wall_seconds':time.perf_counter()-started,'client_cpu_seconds':time.process_time()-cpu_started,
                        'routing_wall_seconds':routing_wall,'routing_cpu_seconds':routing_cpu,'routing_python_peak_bytes':routing_peak,
                        'memory_scope':'Python-traced peak; excludes native tensor and GPU allocations; preexisting tracing may include other allocations'},
                       policy_metadata(config))


def policy_metadata(config):
    if not config.router_checkpoint:
        return {'kind':'deterministic_metadata_heuristic','training_costs':None}
    from .learned import FEATURE_VERSION
    from .training import load_checkpoint
    _,meta=load_checkpoint(config.router_checkpoint,FEATURE_VERSION,config.prompt_version,mode=config.mode)
    return {'kind':'learned_utility_regression','checkpoint_sha256':hashlib.sha256(Path(config.router_checkpoint).read_bytes()).hexdigest(),
            'training_costs':meta['costs'],'training_cost_scope':'training and validation outcome collection; offline optimizer compute excluded',
            'record_sha256':meta['record_sha256'],'settings_sha256':meta['settings_sha256'],'mode':meta['mode']}


def contract_hash(public,assignments,sink,config,tokenizer_identity):
    checkpoint_hash=hashlib.sha256(Path(config.router_checkpoint).read_bytes()).hexdigest() if config.router_checkpoint else None
    return digest({'public':public,'assignments':assignments,'sink':sink,'config':config,'tokenizer':tokenizer_identity,'router_sha256':checkpoint_hash})


def route_senders(kind,metadata,sink,k,question,config,delivered):
    if kind=='selected' and config.router_checkpoint:
        from .learned import FEATURE_VERSION,rank_senders
        from .training import load_checkpoint
        model,_=load_checkpoint(config.router_checkpoint,FEATURE_VERSION,config.prompt_version,mode=config.mode)
        return rank_senders(model,metadata,sink,k,question,delivered,config.seed)
    return select_senders(kind,metadata,sink,k,question,config.seed,delivered)
