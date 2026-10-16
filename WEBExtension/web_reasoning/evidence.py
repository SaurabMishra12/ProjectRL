"""Immutable source identity wins over paraphrase similarity."""
def merge_ledger(previous,incoming):
    result=list(previous);seen={f.atom_id:f for f in previous}
    for fact in incoming:
        if fact.atom_id in seen:
            if seen[fact.atom_id].quote!=fact.quote:raise ValueError('fact_source_changed')
        else:result.append(fact);seen[fact.atom_id]=fact
    return tuple(result)
