"""Offline gold-location diagnostics. Never imported by inference or training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import ROOT, retrieve, assign_evidence, file_hash, write_json, paired_effect

STRATA = ['Disjoint-Gold', 'Co-located-Gold', 'Retrieval-Missing-Gold', 'Undefined-Gold']

def classify_gold(row, assignment):
    titles = sorted({fact[0] for fact in row['supporting_facts']})
    owners = {title: [i for i, docs in enumerate(assignment)
                      if any(d['title'] == title for d in docs)] for title in titles}
    if len(titles) != 2:
        label = 'Undefined-Gold'
    elif any(not owners[t] for t in titles):
        label = 'Retrieval-Missing-Gold'
    elif any(len(owners[t]) != 1 for t in titles):
        raise ValueError('P-stratum requires exclusive document shards')
    else:
        label = 'Co-located-Gold' if owners[titles[0]] == owners[titles[1]] else 'Disjoint-Gold'
    return {'id': row['id'], 'type': row['type'], 'stratum': label,
            'gold_titles': titles, 'gold_owners': owners,
            'shard_sizes': [len(docs) for docs in assignment]}

def classify_rows(rows, cfg):
    return [classify_gold(row, assign_evidence(retrieve(row['question'], row['documents'], cfg['top_k']),
                'P', row['id'], cfg['partition_seed'])) for row in rows]

def heterogeneity(disjoint, colocated, draws=10000, seed=43):
    """Independent question bootstrap of difference in mean paired F1 changes."""
    if not disjoint or not colocated:
        return {'status': 'UNESTIMABLE_EMPTY_STRATUM', 'n_disjoint': len(disjoint), 'n_colocated': len(colocated)}
    a, b = np.array(list(disjoint.values())), np.array(list(colocated.values()))
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
                     for _ in range(draws)])
    return {'status': 'ESTIMATED', 'n_disjoint': len(a), 'n_colocated': len(b),
            'effect': float(a.mean() - b.mean()), 'ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
            'interpretation': 'Observational effect heterogeneity; stratum membership was not randomized'}

def contrasts(rows, experiment, settings=None):
    by = {label: {r['id']: r['conditions'][label]['f1'] for r in rows} for label in rows[0]['conditions']}
    result = []
    if experiment == 'E5':
        seeds = settings['training_seeds']
        for regime in settings['regimes']:
            a = {r['id']: float(np.mean([r['conditions'][regime+f'_Learned_seed{s}']['f1'] for s in seeds])) for r in rows}
            for baseline in ['NoComm', 'SelfUpdate', 'Static', 'Broadcast', 'MatchedRandom']:
                b = ({r['id']: float(np.mean([r['conditions'][regime+f'_MatchedRandom_seed{s}']['f1'] for s in seeds])) for r in rows}
                     if baseline == 'MatchedRandom' else by[regime+'_'+baseline])
                result.append((regime+'_seed_mean_Learned - '+baseline, a, b))
    elif experiment == 'E1':
        for regime in ['R', 'P']:
            for method in ['CW', 'CCW']:
                for base in ['NoComm', 'SelfUpdate']:
                    if regime+'_'+base in by:
                        result.append((regime+'_'+method+' - '+base, by[regime+'_'+method], by[regime+'_'+base]))
        a = {q: by['P_CW'][q]-by['P_NoComm'][q] for q in by['P_CW']}
        b = {q: by['R_CW'][q]-by['R_NoComm'][q] for q in by['R_CW']}
        result.append(('(CW-NoComm)_P - (CW-NoComm)_R', a, b))
    else:
        a,b = {'E2': ('R_CW','R_CCW'), 'E3': ('R_prefix','R_crop'), 'E4': ('P_stopped','P_final')}[experiment]
        result.append((a+' - '+b, by[a], by[b]))
    return result

def analyze(directory, experiment):
    directory = Path(directory)
    if experiment == 'E5':
        # Reuse full checkpoint, sample, arm and result authentication gates.
        from controller_reporting import report_controller
        report_controller(directory)
        plan = json.loads((directory/'manifest.json').read_text())
        gate = json.loads((directory/'evaluation_gate.json').read_text())
        split = next(p for p in directory.parent.glob('controller_splits_*')
                     if file_hash(p/'manifest.json') == gate['split_manifest_sha256'])
        from controller_splits import load_partition
        gold = load_partition(split, 'heldout', gold=True)
        cfg,settings = plan['base'],plan['settings']
        source_hashes = plan['sources']
    else:
        from reporting import load_completed
        load_completed(directory, experiment)
        plan = json.loads((directory/'frozen_manifest.json').read_text())['plan']
        gold = json.loads((directory/'sample_gold.json').read_text())
        cfg,settings = plan['config'],None
        source_hashes = plan['sources']
        if cfg.get('heldout_split_dir'):
            from controller_splits import load_partition
            split = ROOT/cfg['heldout_split_dir']
            if file_hash(split/'manifest.json') != plan['heldout_split_manifest_sha256']:
                raise ValueError('Held-out split manifest changed')
            authenticated = load_partition(split,'heldout',gold=True)
        else:
            from protocol import load_official
            authenticated = [r for r in load_official() if r['id'] in set(plan['question_ids'])]
        if {r['id']:r for r in gold} != {r['id']:r for r in authenticated}:
            raise ValueError('Diagnostic annotations differ from authenticated source')
    if any(file_hash(ROOT/name) != source_hashes[name] for name in ['protocol.py','experiments.py']):
        raise ValueError('Assignment implementation differs from frozen inference source')
    rows = [json.loads(l) for l in (directory/(experiment+'.jsonl')).read_text().splitlines()]
    if len(gold) != len(rows) or {r['id'] for r in gold} != {r['id'] for r in rows}:
        raise ValueError('Gold diagnostic sample differs from completed sample')
    classes = classify_rows(gold,cfg)
    mapping = {r['id']:r['stratum'] for r in classes}
    out = directory/'reports'/experiment/'gold_strata';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'question_assignments.json',classes)
    records = [{'id':r['id'],'type':r['type'],'stratum':mapping[r['id']], 'condition':label,
                'f1':c['f1'],'em':c['em'],'payload_tokens':c['payload_tokens'],
                'provider_refusal_count':c.get('provider_refusal_count',0),
                'logical_calls':c['cost']['logical_calls']} for r in rows for label,c in r['conditions'].items()]
    df = pd.DataFrame(records);df.to_csv(out/'question_level.csv',index=False)
    summary = df.groupby(['stratum','condition']).agg(n=('id','size'),F1=('f1','mean'),EM=('em','mean'),
                     payload=('payload_tokens','mean'),calls=('logical_calls','mean'),
                     refused_terminal_responses=('provider_refusal_count','sum')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(classes).groupby(['stratum','type']).size().rename('n').to_csv(out/'stratum_type_counts.csv')
    effects,interactions = [],[]
    for label,a,b in contrasts(rows,experiment,settings):
        deltas = {}
        for stratum in STRATA:
            ids = [q for q in a if mapping[q] == stratum]
            if not ids:
                effects.append({'contrast':label,'stratum':stratum,'n':0,'status':'UNESTIMABLE_EMPTY_STRATUM'})
                continue
            aa,bb = {q:a[q] for q in ids},{q:b[q] for q in ids}
            effects.append({'contrast':label,'stratum':stratum,'status':'ESTIMATED',**paired_effect(aa,bb)})
            deltas[stratum] = {q:a[q]-b[q] for q in ids}
        interactions.append({'contrast':label,'heterogeneity':'Disjoint-Gold minus Co-located-Gold',
                             **heterogeneity(deltas.get('Disjoint-Gold',{}),deltas.get('Co-located-Gold',{}))})
    estimated = sorted([e for e in effects if e['status']=='ESTIMATED'],key=lambda e:e['randomization_p'])
    previous = 0.
    for i,e in enumerate(estimated):
        previous = max(previous,min(1.,(len(estimated)-i)*e['randomization_p']));e['holm_p']=previous
    stats = {'status':'COMPLETE','experiment':experiment,'n':len(rows),
             'counts':{s:sum(r['stratum']==s for r in classes) for s in STRATA},
             'effects':effects,'heterogeneity':interactions,
             'analysis_source_sha256':file_hash(Path(__file__)),
             'result_sha256':file_hash(directory/(experiment+'.jsonl')),
             'limitations':['Post-hoc diagnostic stratification; pooled primary estimand is retained',
               'Gold labels never select documents, partitions, prompts, actions or checkpoints',
               'Strata describe P initial evidence; R results use these same question groups for paired interaction',
               'Retrieval-Missing-Gold remains in pooled analysis and is reported separately',
               'Seed means use questions as independent units; no guaranteed positive or massive effect',
               'Bootstrap heterogeneity intervals are exploratory and unadjusted; effect p-values Holm-adjusted within this report']}
    write_json(out/'statistics.json',stats)
    return summary,stats
