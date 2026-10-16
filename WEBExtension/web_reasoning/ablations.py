"""Explicit development variants. Nothing launches the factorial automatically."""
from dataclasses import replace
from .config import validate_config
from .data import allocate


def ablation_variants(base_config):
    base=validate_config(base_config)
    return tuple(validate_config(replace(base,experiment='W4',**change)) for change in (
        {},{'prompt_variant':'legacy'},{'packet_mode':'prefix'},
        {'include_answer':False},{'stopping':False},{'regime':'R'}))


def compile_variant(base_state,variant,config,n_agents=3,placement='natural',token_count=len):
    variant=validate_config(variant);example=base_state
    public=example.public
    assignment=allocate(public,n_agents,variant.regime,variant.seed,token_count)
    if placement in ('split_support','co_located'):
        if public.dataset!='2wiki':raise ValueError('oracle_text_placement_only')
        titles={t for t,_ in example.targets.supporting_locators};shards=[[] for _ in range(n_agents)];support_index=0
        for i,source in enumerate(public.sources):
            if source.title in titles:
                node=0 if placement=='co_located' else support_index%n_agents;support_index+=1
            else:node=i%n_agents
            shards[node].extend(source.atoms)
        assignment=tuple(tuple(s) for s in shards)
    elif placement!='natural':raise ValueError('unknown_placement')
    return {'public':public,'assignments':assignment,'config':variant,'oracle':placement!='natural',
            'interpretation':'annotated-support placement, not semantic necessity' if placement!='natural' else 'natural'}
