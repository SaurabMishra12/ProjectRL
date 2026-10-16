"""Paired whole-question inference and explicit unresolved-cell bounds."""
import math
import numpy as np


def exact_mcnemar(wins,losses):
    if type(wins) is not int or type(losses) is not int or min(wins,losses)<0:raise ValueError('invalid_discordances')
    n=wins+losses
    return min(1.,2*sum(math.comb(n,i) for i in range(min(wins,losses)+1))/(2**n)) if n else 1.


def paired_summary(question_rows,contrast,draws=10000,seed=20261016,metric='em'):
    if not question_rows or type(draws) is not int or draws<=0:raise ValueError('nonempty_paired_cohort_required')
    groups={};left,right=contrast
    for row in question_rows:
        if row.get('mode')!='REAL':raise ValueError('mock_is_not_scientific_evidence')
        a=row.get('arms',{}).get(left,{}).get(metric);b=row.get('arms',{}).get(right,{}).get(metric)
        if any(x is not None and (type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=100) for x in (a,b)):
            raise ValueError('invalid_score_scale')
        low=(a if a is not None else 0)-(b if b is not None else 100)
        high=(a if a is not None else 100)-(b if b is not None else 0)
        groups.setdefault(row['question_key'],[]).append((a-b if a is not None and b is not None else None,low,high))
    bounds=[float(np.mean([np.mean([x[k] for x in rows]) for rows in groups.values()])) for k in (1,2)]
    base={'n_questions':len(groups),'bounds':bounds,'resampling_unit':'question','effect':None,'ci95':None,'p_value':None}
    if any(x[0] is None for rows in groups.values() for x in rows):return {**base,'status':'INCOMPLETE'}
    values=np.array([np.mean([x[0] for x in rows]) for rows in groups.values()])
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(draws)])
    wins=int(sum(values>0));losses=int(sum(values<0))
    return {**base,'status':'COMPLETE','effect':float(values.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),
            'wins':wins,'losses':losses,'ties':int(sum(values==0)),
            'p_value':exact_mcnemar(wins,losses) if metric=='em' and all(x in (-100,0,100) for x in values) else None,
            'noninferiority_established':False}


def identification_bounds(question_rows,contrast=('selected','random'),metric='em'):
    return paired_summary(question_rows,contrast,draws=1,metric=metric)['bounds']
