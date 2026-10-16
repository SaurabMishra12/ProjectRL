"""Deterministic grouped splits; infeasible exact cohorts fail rather than leak."""
import hashlib
import re
from collections import Counter
from .data import seeded


def normalize(text):return ' '.join(re.findall(r'\w+',text.casefold()))


def balanced_sinks(keys,n_agents,seed):
    if n_agents not in (2,3,4,5) or len(keys)!=len(set(keys)):raise ValueError('invalid_sink_cohort')
    order=sorted(keys,key=lambda k:hashlib.sha256(f'{seed}:{k}'.encode()).hexdigest())
    nodes=list(range(n_agents));seeded(seed,'sink').shuffle(nodes)
    return {key:nodes[i%n_agents] for i,key in enumerate(order)}


def freeze_splits(examples,sizes,seed):
    rows=sorted(examples,key=lambda e:e.public.key)
    if len({e.public.key for e in rows})!=len(rows) or any(type(n) is not int or n<0 for n in sizes.values()):
        raise ValueError('invalid_split_population')
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,e in enumerate(rows):
        markers=[('question',normalize(e.public.question))]
        if e.public.dataset=='finqa':markers.append(('report',dict(e.targets.strata)['report']))
        else:markers.extend(('title',normalize(s.title)) for s in e.public.sources)
        for marker in markers:
            if marker in seen:parent[root(i)]=root(seen[marker])
            else:seen[marker]=i
    groups={}
    for i,e in enumerate(rows):groups.setdefault(root(i),[]).append(e)
    remaining=list(groups.values());seeded(seed,'groups').shuffle(remaining)
    population=Counter(dict(e.targets.strata).get('type','unknown') for e in rows)
    result={}
    for name,n in sizes.items():
        chosen=[]; counts=Counter()
        while len(chosen)<n:
            eligible=[g for g in remaining if len(g)<=n-len(chosen)]
            if not eligible:raise ValueError('source_group_constraints_prevent_exact_cohort')
            def deficit(g):
                return sum(population[dict(e.targets.strata).get('type','unknown')]*n/max(1,len(rows))
                           -counts[dict(e.targets.strata).get('type','unknown')] for e in g)/len(g)
            group=max(eligible,key=deficit)
            remaining.remove(group);chosen.extend(group)
            counts.update(dict(e.targets.strata).get('type','unknown') for e in group)
        result[name]=tuple(chosen)
    return result
