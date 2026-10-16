"""Explicit reconstruction of frozen public inputs and saved engineering/results records."""
from .contracts import Atom,Source,PublicExample,PrivateTargets,Proposal,Fact,Need,Outcome,AgentState
from .runner import ArmResult,QuestionRun
from .packets import Packet


def public_from_dict(obj):
    if set(obj)!={'key','dataset','question','sources'}:raise ValueError('unexpected_public_fields')
    sources=[]
    for s in obj['sources']:
        sources.append(Source(s['document_id'],s['title'],tuple(Atom(**a) for a in s['atoms'])))
    return PublicExample(obj['key'],obj['dataset'],obj['question'],tuple(sources))


def targets_from_dict(obj):
    return PrivateTargets(obj['answer'],tuple(obj['aliases']),tuple(tuple(x) for x in obj['supporting_locators']),obj['program'],tuple(tuple(x) for x in obj['strata']))


def outcome_from_dict(obj):
    p=obj['proposal']
    if p is not None:
        p=Proposal(**{**p,'facts':tuple(Fact(**x) for x in p['facts']),'needs':tuple(Need(**x) for x in p['needs'])})
    return Outcome(**{**obj,'proposal':p})


def state_from_dict(obj):
    return AgentState(obj['agent_id'],tuple(Atom(**x) for x in obj['visible_atoms']),tuple(Fact(**x) for x in obj['ledger']),outcome_from_dict(obj['outcome']))


def packet_from_dict(obj):
    return Packet(**{**obj,'facts':tuple(Fact(**x) for x in obj['facts']),'omitted_atom_ids':tuple(obj['omitted_atom_ids'])})


def run_from_dict(obj):
    arms={}
    for name,a in obj['arms'].items():
        arms[name]=ArmResult(a['name'],tuple(state_from_dict(x) for x in a['states']),a['sink'],outcome_from_dict(a['outcome']),
                             tuple(a['references']),tuple(packet_from_dict(x) for x in a['transmissions']),a['stop_reason'])
    return QuestionRun(obj['question_key'],obj['dataset'],obj['n_agents'],tuple(state_from_dict(s) for s in obj['initial']),
                       arms,tuple(obj['logical_refs']),tuple(packet_from_dict(x) for x in obj['transmissions']),obj['mode'],obj['contract_hash'],obj.get('timings',{}),obj.get('policy_metadata',{}))
