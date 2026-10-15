"""Independent, gold-free frozen-proposal readouts; no provider dependencies.

Original is Hotpot-style plurality with first-seat ties. Benchmark adapters must
explicitly describe missing-value handling; GSM8K parser failures are not votes.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import re
import string

KINDS={'answer','abstain','refusal','empty','missing','malformed','invalid'}
RULES=('original','aware','lexical','strict_majority')

@dataclass(frozen=True)
class Proposal:
    text:str
    kind:str='answer'
    def __post_init__(self):
        if not isinstance(self.text,str) or self.kind not in KINDS:raise ValueError('Typed string proposal required')


def hotpot_normalize(text):
    text=text.lower().translate(str.maketrans('','',string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def hotpot_proposal(response):
    refusal=response.get('provider_refusal',False)
    if type(refusal) is not bool:raise ValueError('Boolean provider refusal flag required')
    if 'answer' not in response:return Proposal('','missing')
    value=response['answer']
    if not isinstance(value,str):return Proposal('','malformed')
    if refusal and value.strip():raise ValueError('Refusal conflicts with answer')
    if refusal:return Proposal(value,'refusal')
    if not value.strip():return Proposal(value,'empty')
    if hotpot_normalize(value)=='noanswer':return Proposal(value,'abstain')
    return Proposal(value)


def gsm_proposal(parsed_answer):
    # Source parser's None is missing numeric extraction, not a semantic decision.
    return Proposal('','malformed') if parsed_answer is None else Proposal(str(parsed_answer))


def readout(proposals,rule,normalize=hotpot_normalize):
    if not proposals or rule not in RULES:raise ValueError('Known rule and nonempty original seats required')
    valid=[p for p in proposals if p.kind in {'answer','abstain','refusal'}]
    if not valid:return Proposal('','invalid')
    pool=valid if rule=='original' else [p for p in valid if p.kind=='answer']
    if not pool:return Proposal('noanswer','abstain')
    counts=Counter(normalize(p.text) for p in pool)
    top=max(counts.values())
    if rule=='strict_majority' and top<=len(proposals)/2:return Proposal('noanswer','abstain')
    tied={k for k,v in counts.items() if v==top}
    key=min(tied) if rule=='lexical' else next(normalize(p.text) for p in pool if normalize(p.text) in tied)
    return next(p for p in pool if normalize(p.text)==key)


def metrics(rows):
    if not rows:raise ValueError('Nonempty evaluation required')
    if any(type(r['answerable']) is not bool or type(r['correct']) is not bool for r in rows):
        raise ValueError('Explicit source-answerability and correctness labels required')
    answered=[r for r in rows if r['decision'].kind=='answer']
    supported=[r for r in rows if r['answerable']];unsupported=[r for r in rows if not r['answerable']]
    safe=lambda a,b:a/b if b else None
    return {'n':len(rows),'answer_accuracy_supported':safe(sum(r['correct'] and r['decision'].kind=='answer' for r in supported),len(supported)),
        'coverage':len(answered)/len(rows),'selective_risk':safe(sum(not r['answerable'] or not r['correct'] for r in answered),len(answered)),
        'false_answer_rate':safe(sum(r['decision'].kind=='answer' for r in unsupported),len(unsupported)),
        'valid_abstention_rate_unsupported':safe(sum(r['decision'].kind=='abstain' for r in unsupported),len(unsupported)),
        'provider_refusal_rate':sum(r['decision'].kind=='refusal' for r in rows)/len(rows),
        'invalid_output_rate':sum(r['decision'].kind in {'empty','missing','malformed','invalid'} for r in rows)/len(rows)}


def cluster_interactions(rows):
    groups=defaultdict(dict)
    for r in rows:
        if r['context'] in groups[r['id']]:raise ValueError('Duplicate question-context')
        if any(r[k] not in (0,1) for k in ['CA','PA','CO','PO']):raise ValueError('Binary outcomes required')
        groups[r['id']][r['context']]=(r['CA']-r['PA'])-(r['CO']-r['PO'])
    if any(set(g)!={'sufficient','insufficient'} for g in groups.values()):raise ValueError('Both paired contexts required')
    return {q:sum(g.values())/2 for q,g in groups.items()}
