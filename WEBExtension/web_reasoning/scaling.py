"""Common-cohort scaling only; legacy estimates cannot stand in for new outcomes."""
from .data import allocate
from .splits import balanced_sinks
from .runner import run_w1,contract_hash


def theoretical_reference(n,rounds):
    if n not in (2,3,4,5) or rounds not in (1,2,3):raise ValueError('unsupported_team_or_depth')
    return {'dense_deliveries':rounds*n*(n-1),'sparse_three_phase_ceiling':3*(n-1),'reference_only':True}


def run_scaling(public_examples,nested_keys,backend_factory,config,existing_runs=None):
    population={p.key:p for p in public_examples};nested=set(nested_keys)
    if len(population)!=len(public_examples) or len(nested)!=len(nested_keys) or not nested<=set(population):raise ValueError('invalid_nested_cohort')
    result=[];existing_runs=existing_runs or {}
    for n in config.team_sizes:
        backend=backend_factory(n,config);sinks=balanced_sinks(list(population),n,config.seed)
        for key in sorted(nested):
            public=population[key];assignment=allocate(public,n,config.regime,config.seed,backend.tokenizer.count)
            expected=contract_hash(public,assignment,sinks[key],config,backend.tokenizer.identity)
            if (key,n) in existing_runs:
                run=existing_runs[key,n]
                if run.contract_hash!=expected or run.mode!=config.mode or run.question_key!=key or run.n_agents!=n:
                    raise ValueError('incompatible_nested_reuse')
            else:run=run_w1(public,assignment,sinks[key],backend,config)
            result.append(run)
    return tuple(result)
