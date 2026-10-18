"""Independent saved-output analysis. No project runner imports, model SDKs or network."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, itertools, json, math, re, string
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / 'results'
INPUTS = {}

def load(path):
    p = REPO / path
    INPUTS[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    if p.suffix == '.jsonl':
        return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    return json.loads(p.read_text())

def normalize(text):
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())

def score(pred, gold):
    p, g = normalize(pred), normalize(gold)
    em = float(p == g)
    if (p in ('yes', 'no', 'noanswer') or g in ('yes', 'no', 'noanswer')) and p != g:
        return em, 0.
    pt, gt = p.split(), g.split()
    common = sum((Counter(pt) & Counter(gt)).values())
    f1 = 2 * common / (len(pt) + len(gt)) if common else 0.
    return em, f1

def vote(answers, rule='original'):
    # Ties use stable node order, matching the historical original/aware rule.
    nonempty = list(answers) if rule == 'original' else [a for a in answers if normalize(a) and normalize(a) != 'noanswer']
    if not nonempty:
        return 'noanswer'
    counts = Counter(normalize(a) for a in nonempty)
    if rule == 'strict' and max(counts.values()) <= len(answers) / 2:
        return 'noanswer'
    winners = [a for a in nonempty if counts[normalize(a)] == max(counts.values())]
    if rule == 'lex':
        return min(winners, key=lambda a: normalize(a))
    return winners[0]

def effect(values, seed=20261016):
    x = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), (10000, len(x)))].mean(axis=1)
    return dict(n=len(x), mean=float(x.mean()), ci95=list(map(float, np.quantile(boot,[.025,.975]))),
                sd=float(x.std(ddof=1)), positive=int((x>1e-12).sum()), negative=int((x<-1e-12).sum()))

def graph_support():
    edges = [(i,j) for i in range(3) for j in range(3) if i != j]
    groups = defaultdict(list)
    for mask in range(64):
        e = [edges[k] for k in range(6) if mask & (1<<k)]
        active = tuple(sorted({j for i,j in e}))
        groups[len(e),active].append(e)
    return {'total_graphs':64, 'groups':[{'edge_count':k[0],'recipients':k[1], 'graphs':v,
             'alternative_count':len(v)-1} for k,v in sorted(groups.items())],
            'two_edge_one_recipient_class_sizes':[len(v) for (m,r),v in groups.items() if m==2 and len(r)==1],
            'cw_ccw_unlabeled_isomorphic':True}

def main():
    gold = {q['id']:q['answer'] for q in load('WEBExtension/data/hotpot_dev_fullwiki_v1.json')}
    base = 'WEBExtension/artifacts/run_f6aa61bac2c3dfae/'
    sets = {n:load(base+n+'.jsonl') for n in ('E1','E2','E3','E4')}
    sets['E3v2'] = load('WEBExtension/artifacts/e3_corrected_b5c4ff9db9b9b2bc/E3.jsonl')
    sets['E5'] = load('WEBExtension/artifacts/controller_study_a0b3768357484b45/E5.jsonl')
    ids = {k:{r['id'] for r in rows} for k,rows in sets.items()}
    assert all(len(v)==300 for v in ids.values())
    assert len(set.union(*ids.values()))==300
    mismatches=[]; conditions=defaultdict(list); evidence=[]
    for exp,rows in sets.items():
        for row in rows:
            for name,c in row['conditions'].items():
                em,f1=score(c['answer'], gold[row['id']])
                if abs(c['em']/100-em)>1e-9 or abs(c['f1']/100-f1)>1e-9:
                    mismatches.append([exp,row['id'],name])
                conditions[exp+':'+name].append((em,f1))
                evidence.append({'experiment':exp,'id':row['id'],'arm':name,'em':em,'f1':f1})
    byid={exp:{r['id']:r for r in rows} for exp,rows in sets.items()}
    e1=sets['E1'];e2=byid['E2']
    reuse=sum(r['conditions'][a]['cost']['call_keys']==e2[r['id']]['conditions'][a]['cost']['call_keys']
              for r in e1 for a in ('R_NoComm','R_CW','R_CCW'))
    inter=[((r['conditions']['R_CW']['f1']-r['conditions']['R_NoComm']['f1'])-
           (r['conditions']['P_CW']['f1']-r['conditions']['P_NoComm']['f1']))/100 for r in e1]
    oldident=sum(r['crop_equals_prefix'] for r in sets['E3'])
    eligible=[r for r in sets['E3v2'] if r['eligible']]
    distinct=sum(r['conditions']['R_prefix']['deliveries'][0]['text'] !=
                 r['conditions']['R_crop']['deliveries'][0]['text'] for r in eligible)
    lengths=Counter(c['deliveries'][0]['tokens'] for r in eligible for a,c in r['conditions'].items()
                    if a in ('R_crop','R_prefix'))
    e4=sets['E4']
    activity=[];readout_counts=defaultdict(Counter); e5rows=sets['E5']
    for row in e5rows:
        for arm,c in row['conditions'].items():
            answers=[s['answer'] for s in c['responses']]
            for rule in ('original','aware','lex','strict'):
                ans=vote(answers,rule)
                readout_counts[arm][rule]+=int(score(ans,gold[row['id']])[0])
            if 'Learned_seed' in arm:
                other=arm.replace('Learned_seed','MatchedRandom_seed')
                a,b=c['action'],row['conditions'][other]['action']
                rec=lambda x: sorted({e[1] for e in x['edges']})
                activity.append({'id':row['id'],'arm':arm,'edges':a['edges'],'recipients':rec(a),
                    'calls':c['cost']['logical_calls'],'matched_random_recipients':rec(b),
                    'same_recipients':rec(a)==rec(b),
                    'same_edge_count':len(a['edges'])==len(b['edges']), 'same_quota':a['quota']==b['quota']})
    four={}; p_interaction=[]
    for rule in ('original','aware','lex','strict'):
        contrasts=[]
        for row in e5rows:
            vals={a:score(vote([s['answer'] for s in c['responses']],rule),gold[row['id']])[0]
                  for a,c in row['conditions'].items()}
            contrasts.append((vals['R_Broadcast']-vals['R_NoComm'])-(vals['P_Broadcast']-vals['P_NoComm']))
        four[rule]=effect(contrasts)
    for row in e5rows:
        outcomes={(a,r):score(vote([s['answer'] for s in row['conditions'][a]['responses']],r),gold[row['id']])[0]
                  for a in ('P_Broadcast','P_NoComm') for r in ('original','aware')}
        p_interaction.append(outcomes['P_Broadcast','aware']-outcomes['P_NoComm','aware']-
                             outcomes['P_Broadcast','original']+outcomes['P_NoComm','original'])
    e6=[]; e6mis=[]
    for p in sorted((REPO/'WEBExtension/causal_study/paid_pilot/questions').glob('*.json')):
        r=load(str(p.relative_to(REPO)));cs={}
        for a,c in r['cells'].items():
            em,f1=score(c['answer'],gold[r['question_id']]);cs[a]=f1
            if abs(em*100-c['em'])>1e-8 or abs(f1*100-c['f1'])>1e-8:e6mis.append([r['question_id'],a])
        e6.append({'id':r['question_id'],**cs,'interaction':cs['distributed_comm']-cs['distributed_nocomm']-
                   cs['colocated_comm']+cs['colocated_nocomm']})
    ledger=load('WEBExtension/causal_study/paid_pilot/ledger.jsonl')
    success=[r for r in ledger if r['event']=='success']
    status=load('WEBExtension/causal_study/paid_pilot/status.json')
    latency=sum(r['latency_seconds'] for r in success)
    # Pure mathematical check: exact PPO clipped objective is flat in the clipped positive-advantage region.
    eps=.2;ratio=1.5;adv=1.;mu=math.log(ratio)
    objective=lambda m:min(math.exp(m)*adv,min(max(math.exp(m),1-eps),1+eps)*adv)
    h=1e-6; numerical=(objective(mu+h)-objective(mu-h))/(2*h)
    historical_records=load('results/honest_fixed_budget/full_150q_results.json')
    legacy={k:{'rows':len(v),'unique_question_texts':len({r['question'] for r in v}),
                'correct_field_total':sum(bool(r.get('is_correct')) for r in v),
                'first_question':v[0]['question']} for k,v in historical_records.items()}
    result={'units':'All effects fractions, multiply by 100 for percentage points. Bootstrap is exploratory.',
      'independent_ids':{k:len(v) for k,v in ids.items()},'union_ids':300,'score_mismatches':mismatches,
      'condition_means':{k:{'em':float(np.mean([x[0] for x in v])), 'f1':float(np.mean([x[1] for x in v]))} for k,v in conditions.items()},
      'E1_interaction_R_minus_P_CW_vs_NoComm':effect(inter), 'E2_E1_reused_key_lists':reuse,
      'E2_CW_minus_CCW':effect([(r['conditions']['R_CW']['f1']-r['conditions']['R_CCW']['f1'])/100 for r in sets['E2']]),
      'E3_original_identical_pairs':oldident,'E3v2_eligible':len(eligible),'E3v2_distinct_payloads':distinct,
      'E3v2_recorded_token_lengths':dict(lengths),'E3v2_prefix_minus_crop':effect([(r['conditions']['R_prefix']['f1']-r['conditions']['R_crop']['f1'])/100 for r in eligible]),
      'E4_stopped_minus_final':effect([(r['conditions']['P_stopped']['f1']-r['conditions']['P_final']['f1'])/100 for r in e4]),
      'E4_hypothetical_calls_avoided':sum(r['hypothetical_calls_avoided'] for r in e4),
      'E4_wrong_unanimity':sum(r['wrong_unanimity'] for r in e4),
      'E4_continuation_corrected_wrong_stop':sum(r['continuation_corrected_wrong_stop'] for r in e4),
      'E5_activity_pairs':len(activity),'E5_recipient_mismatches':sum(not a['same_recipients'] for a in activity),
      'E5_recipient_count_mismatches':sum(len(a['recipients'])!=len(a['matched_random_recipients']) for a in activity),
      'E5_edge_quota_mismatches':sum(not(a['same_edge_count'] and a['same_quota']) for a in activity),
      'E5_learned_recipient_counts':dict(Counter(len(a['recipients']) for a in activity)),
      'E5_readout_exact_counts':dict(readout_counts),'E5_readout_interactions':four,
      'E5_P_readout_by_communication_interaction':effect(p_interaction),
      'E6_interaction':effect([r['interaction'] for r in e6]),'E6_score_mismatches':e6mis,
      'E6_timing':{'success':len(success),'http_latency_sum_seconds':latency,'request_latency_mean_seconds':latency/len(success),
                  'session_wall_seconds':status['runtime_this_session_seconds'],
                  'amortized_wall_seconds_per_success':status['runtime_this_session_seconds']/len(success), 'usage':status['usage']},
      'graph_support':graph_support(), 'legacy_saved_records':legacy,
      'ppo_plateau_check':{'ratio':ratio,'advantage':adv,'clip_epsilon':eps,'objective_numerical_derivative':numerical,
                           'current_legacy_code_multiplier':min(max(ratio,1-eps),1+eps)*adv,
                           'scope':'Current NumPy code only; score-function factor omitted. No checkpoint provenance or training claim.'}}
    for name,rows in [('question_scores',evidence),('activity_pairs',activity),('e6_question_contrasts',e6)]:
        (OUT/(name+'.jsonl')).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    (OUT/'offline_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'analysis_input_hashes.json').write_text(json.dumps(INPUTS,indent=2)+'\n')
    print(json.dumps({k:(len(result[k]) if isinstance(result[k],list) else result[k]) for k in ('union_ids','score_mismatches','E2_E1_reused_key_lists','E3_original_identical_pairs','E3v2_eligible','E5_recipient_mismatches','E6_score_mismatches','ppo_plateau_check')},indent=2))

if __name__=='__main__':
    main()
