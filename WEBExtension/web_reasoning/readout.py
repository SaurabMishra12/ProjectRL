"""Readouts never silently drop invalid or unknown service outcomes."""
from collections import Counter
import re
import string


def normalize(answer):
    value=''.join(c for c in answer.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',value).split())


def abstention_vote(states,node_order):
    return vote(states,node_order,exclude_abstentions=True)


def original_vote(states,node_order):
    return vote(states,node_order,exclude_abstentions=False)


def vote(states,node_order,exclude_abstentions):
    mapping={s.agent_id:s for s in states}
    if set(node_order)!=set(mapping) or len(node_order)!=len(mapping):raise ValueError('invalid_node_order')
    pool=[]
    for node in node_order:
        outcome=mapping[node].outcome
        if outcome.kind in ('invalid','unresolved'):raise ValueError('incomplete_readout_cell')
        if outcome.proposal and outcome.proposal.answer is not None:pool.append(outcome.proposal.answer)
        elif not exclude_abstentions:pool.append('noanswer')
    if not pool:return 'noanswer'
    counts=Counter(normalize(a) for a in pool)
    return max(pool,key=lambda a:counts[normalize(a)])


def sink_answer(arm):
    if arm.outcome.kind in ('invalid','unresolved'):return None
    return arm.outcome.proposal.answer if arm.outcome.proposal and arm.outcome.proposal.answer else 'noanswer'
