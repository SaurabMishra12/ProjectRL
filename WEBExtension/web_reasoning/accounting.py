"""Observed physical union and nonadditive logical request references."""
def valid_usage(value):return type(value) is int and value>=0


def cost_summary(events,logical_refs):
    starts=[e for e in events if e['event']=='started'];success=[e for e in events if e['event']=='success']
    closed={e['attempt_id'] for e in events if e['event'] in ('success','rejected')}
    result={'physical_attempts':len(starts),'physical_completions':len(success),'logical_references':len(logical_refs),
            'rejected_attempts':sum(e['event']=='rejected' for e in events),
            'uncertain_attempts':sum(e['attempt_id'] not in closed for e in starts)}
    for field,label in [('prompt_tokens','input'),('completion_tokens','output')]:
        values=[(e.get('usage') or {}).get(field) for e in success]
        result[f'known_{label}_tokens_lower_bound']=sum(v for v in values if valid_usage(v))
        result[f'{label}_usage_missing']=sum(not valid_usage(v) for v in values)
    result['currency_cost']=None
    return result
