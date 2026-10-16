"""Bounded observable metadata and explicit matched update schedules."""
from dataclasses import dataclass
import hashlib
import json
from .data import seeded
from .packets import pack_control,tokens


@dataclass(frozen=True)
class Metadata:
    agent_id:int
    status:str
    valid_fact_count:int
    entity_tags:tuple[str,...]
    needs:tuple[str,...]
    fact_hashes:tuple[str,...]


@dataclass(frozen=True)
class Schedule:
    edges:tuple[tuple[int,int],...]
    recipients:tuple[int,...]
    control_frames:tuple=()


def metadata_frame(state,tokenizer,cap=96):
    p=state.outcome.proposal
    payload={'status':p.status if p else state.outcome.kind,'count':len(state.ledger),
             'tags':list(p.entity_tags) if p else [],'needs':[n.entity+' '+n.relation for n in p.needs] if p else [],
             'hashes':[hashlib.sha256(f.atom_id.encode()).hexdigest() for f in state.ledger]}
    text=pack_control(state.agent_id,payload,cap,tokenizer);wire=json.loads(text)
    return Metadata(state.agent_id,wire['status'],wire['count'],tuple(wire['tags']),tuple(wire['needs']),tuple(wire['hashes'])),text


def jaccard(a,b):
    a,b=set(a),set(b);return len(a&b)/len(a|b) if a|b else 0.


def select_senders(kind,metadata,sink,k,question,seed,already_delivered):
    by_id={m.agent_id:m for m in metadata}
    if len(by_id)!=len(metadata) or sink not in by_id:raise ValueError('invalid_metadata_population')
    candidates=[m.agent_id for m in metadata if m.agent_id!=sink]
    if type(k) is not int or not 0<=k<=len(candidates):raise ValueError('invalid_sender_count')
    tie=list(sorted(candidates));seeded(seed,'sender_tie',sink).shuffle(tie);rank={node:i for i,node in enumerate(tie)}
    if kind=='static':return tuple(sorted(tie[:k]))
    if kind=='random':
        seeded(seed,'random_senders',sink).shuffle(tie);return tuple(sorted(tie[:k]))
    if kind not in ('heuristic','selected'):raise ValueError('unknown_policy')
    needs=tokens(' '.join(by_id[sink].needs));query=tokens(question)
    def score(node):
        m=by_id[node];tags=tokens(' '.join(m.entity_tags))
        delivered=bool(m.fact_hashes) and set(m.fact_hashes)<=set(already_delivered)
        return 2*jaccard(tags,needs)+jaccard(tags,query)+.5*(m.valid_fact_count>0)-.5*delivered
    # Sorting is O(N log N); the feature/scoring pass is O(N). No linear-selection claim.
    return tuple(sorted(sorted(candidates,key=lambda node:(-score(node),rank[node]))[:k]))


def one_round_schedule(kind,n_agents,sink,senders):
    if n_agents not in (2,3,4,5) or type(sink) is not int or not 0<=sink<n_agents:raise ValueError('invalid_team')
    if kind=='dense':edges=tuple((s,t) for t in range(n_agents) for s in range(n_agents) if s!=t)
    elif kind=='instar':edges=tuple((s,sink) for s in range(n_agents) if s!=sink)
    elif kind=='private':return Schedule((),(sink,))
    elif kind=='nocomm':return Schedule((),())
    elif kind in ('static','random','heuristic','selected'):
        if len(senders)!=len(set(senders)) or any(type(s) is not int or not 0<=s<n_agents or s==sink for s in senders):
            raise ValueError('invalid_sender')
        edges=tuple((s,sink) for s in sorted(senders))
    else:raise ValueError('unknown_policy')
    return Schedule(edges,tuple(sorted({t for s,t in edges})))
