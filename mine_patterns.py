"""
Deep Hidden Pattern Mining for Sparse Semantic Communication
============================================================
Mines publishable trends beyond basic accuracy/efficiency:

 1. Per-round accuracy trajectory (does communication help each round?)
 2. Initial agent diversity → final accuracy predictor
 3. Compression ratio on correct vs incorrect samples
 4. Weight asymmetry between agent pairs (who trusts whom?)
 5. Question complexity (length) vs compression vs accuracy
 6. Echo chamber depth — does consensus form early or late?
 7. Token savings distribution: where does the system save most?
 8. Communication rescue conditions (what makes a sample rescuable?)
 9. Mode switching patterns across rounds
10. Agent dominance analysis (which agent id wins most?)

Output: resultsall/hidden_patterns.json
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json, re, math, statistics, pathlib
from collections import defaultdict, Counter

RESULTS_DIR = pathlib.Path(r"c:\Users\msaur\OneDrive\Desktop\projectRL\resultsall")
OUTPUT = RESULTS_DIR / "hidden_patterns.json"

# ─── Load all valid samples ──────────────────────────────────────────────────
def load_all():
    merged = {}
    for jf in sorted(RESULTS_DIR.glob("*.json")):
        if jf.name in ("analysis.json", "full_analysis.json", "hidden_patterns.json"):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except:
            continue
        for method_key, samples in data.get("detailed_samples", {}).items():
            if not isinstance(samples, list):
                continue
            mk = method_key.lower()
            ds = "aqua_math" if "math" in mk or "aqua" in mk else "gsm8k"
            for s in samples:
                q = s.get("question", "")
                if q and q not in merged:
                    merged[q] = (s, ds)
    return [(s, ds) for s, ds in merged.values()]

all_samples_ds = load_all()
all_samples = [s for s, _ in all_samples_ds]
ds_map      = {s["question"]: ds for s, ds in all_samples_ds}
n = len(all_samples)
print(f"Loaded {n} unique samples")

avg = lambda x: sum(x)/len(x) if x else 0
std = lambda x: statistics.stdev(x) if len(x) > 1 else 0

report = {}

# ─── 1. PER-ROUND ACCURACY TRAJECTORY ───────────────────────────────────────
print("\n[1] Per-round accuracy trajectory...")

def get_round_correct(s, round_num):
    """Get per-agent correctness at a given round."""
    rc = s.get("raw_communications", [])
    gold = s.get("gold_answer", "")
    if round_num == 0:
        ia = s.get("initial_answers", {})
        ic = s.get("initial_correct", {})
        return [bool(ic.get(k, False)) for k in ia]
    # For rounds 1-3: look at final_answers (we only have initial + final, not intermediate)
    # Proxy: use final answers for last round
    if round_num >= 3:
        fc = s.get("final_answers", {})
        gold_lower = str(gold).strip().lower()
        correctness = []
        for k, v in fc.items():
            nums_v = re.findall(r"-?\d+\.?\d*", str(v).lower())
            nums_g = re.findall(r"-?\d+\.?\d*", gold_lower)
            if nums_v and nums_g:
                try:
                    correct = abs(float(nums_v[-1]) - float(nums_g[-1])) < 1e-3
                except:
                    correct = str(v).strip().lower() == gold_lower
            else:
                correct = str(v).strip().lower() == gold_lower
            correctness.append(correct)
        return correctness
    return []

round0_acc = []
round3_acc = []
for s in all_samples:
    r0 = get_round_correct(s, 0)
    r3 = get_round_correct(s, 3)
    if r0: round0_acc.append(avg([1 if c else 0 for c in r0]))
    if r3: round3_acc.append(avg([1 if c else 0 for c in r3]))

# Initial majority vote
init_mv = []
final_mv = []
for s in all_samples:
    ic = s.get("initial_correct", {})
    vals = [bool(v) for v in ic.values()]
    if vals:
        init_mv.append(1 if sum(vals) > len(vals)/2 else 0)
    final_mv.append(1 if s.get("is_correct") else 0)

trajectory = {
    "round0_per_agent_accuracy": round(avg(round0_acc), 4),
    "round3_per_agent_accuracy": round(avg(round3_acc), 4),
    "round0_majority_vote_accuracy": round(avg(init_mv), 4),
    "final_majority_vote_accuracy": round(avg(final_mv), 4),
    "per_agent_improvement": round(avg(round3_acc) - avg(round0_acc), 4),
    "majority_vote_improvement": round(avg(final_mv) - avg(init_mv), 4),
}
report["1_round_trajectory"] = trajectory
print(f"  Round-0 agent avg: {trajectory['round0_per_agent_accuracy']*100:.2f}%")
print(f"  Round-3 agent avg: {trajectory['round3_per_agent_accuracy']*100:.2f}%")
print(f"  MV improvement:    {trajectory['majority_vote_improvement']*100:+.2f}%")


# ─── 2. INITIAL DIVERSITY → FINAL ACCURACY ──────────────────────────────────
print("\n[2] Initial agent diversity vs final accuracy...")

diversity_vs_correct = defaultdict(list)
for s in all_samples:
    ic = s.get("initial_correct", {})
    vals = [bool(v) for v in ic.values()]
    n_agents = len(vals)
    if n_agents == 0:
        continue
    n_correct = sum(vals)
    # diversity bucket: how many agents initially correct
    bucket = f"{n_correct}/{n_agents}_correct_at_round0"
    diversity_vs_correct[bucket].append(1 if s.get("is_correct") else 0)

diversity_result = {}
for bucket in sorted(diversity_vs_correct.keys()):
    vals = diversity_vs_correct[bucket]
    diversity_result[bucket] = {
        "count": len(vals),
        "final_accuracy": round(avg(vals), 4),
        "pct_of_total": round(len(vals)/n, 4),
    }
    print(f"  {bucket}: n={len(vals)}, final_acc={avg(vals)*100:.1f}%")

report["2_initial_diversity_vs_final_accuracy"] = diversity_result


# ─── 3. COMPRESSION ON CORRECT vs INCORRECT ─────────────────────────────────
print("\n[3] Compression ratio on correct vs incorrect samples...")

correct_comp   = [s.get("compression_ratio", 0) for s in all_samples if s.get("is_correct")]
incorrect_comp = [s.get("compression_ratio", 0) for s in all_samples if not s.get("is_correct")]
correct_tx   = [s.get("total_tokens_transmitted", 0) for s in all_samples if s.get("is_correct")]
incorrect_tx = [s.get("total_tokens_transmitted", 0) for s in all_samples if not s.get("is_correct")]

comp_correct_vs_incorrect = {
    "correct_n": len(correct_comp),
    "incorrect_n": len(incorrect_comp),
    "correct_avg_compression": round(avg(correct_comp), 4),
    "incorrect_avg_compression": round(avg(incorrect_comp), 4),
    "compression_diff": round(avg(correct_comp) - avg(incorrect_comp), 4),
    "correct_avg_tokens_tx": round(avg(correct_tx), 1),
    "incorrect_avg_tokens_tx": round(avg(incorrect_tx), 1),
    "token_diff": round(avg(correct_tx) - avg(incorrect_tx), 1),
}

try:
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(correct_comp, incorrect_comp, alternative='two-sided')
    comp_correct_vs_incorrect["mannwhitney_u"] = round(float(u), 1)
    comp_correct_vs_incorrect["mannwhitney_p"] = round(float(p), 6)
    comp_correct_vs_incorrect["significant_p05"] = p < 0.05
except:
    pass

report["3_compression_correct_vs_incorrect"] = comp_correct_vs_incorrect
print(f"  Correct   samples: comp={avg(correct_comp):.4f}, tokens={avg(correct_tx):.0f}")
print(f"  Incorrect samples: comp={avg(incorrect_comp):.4f}, tokens={avg(incorrect_tx):.0f}")
print(f"  Diff: {comp_correct_vs_incorrect['compression_diff']:+.4f}")


# ─── 4. WEIGHT ASYMMETRY (who trusts whom?) ──────────────────────────────────
print("\n[4] Agent-pair weight asymmetry analysis...")

pair_weights = defaultdict(list)  # (i,j) -> list of weights
for s in all_samples:
    wms = s.get("weight_matrices", [])
    if not wms:
        continue
    wm = wms[-1]  # last round weight matrix
    for i, row in enumerate(wm):
        for j, w in enumerate(row):
            if i != j and w > 0:
                pair_weights[(i, j)].append(w)

pair_stats = {}
for (i, j), ws in sorted(pair_weights.items()):
    reverse = pair_weights.get((j, i), [])
    asym = avg(ws) - avg(reverse) if reverse else 0
    pair_stats[f"agent{i}->agent{j}"] = {
        "avg_weight": round(avg(ws), 4),
        "std_weight": round(std(ws), 4),
        "reverse_avg": round(avg(reverse), 4) if reverse else None,
        "asymmetry_delta": round(asym, 4),
    }
    print(f"  Agent{i}->Agent{j}: w={avg(ws):.4f}  | reverse={avg(reverse):.4f} | asym={asym:+.4f}")

report["4_weight_asymmetry"] = pair_stats

# Overall asymmetry score
asym_vals = [abs(v["asymmetry_delta"]) for v in pair_stats.values() if v["asymmetry_delta"] is not None]
report["4_weight_asymmetry"]["_summary"] = {
    "mean_absolute_asymmetry": round(avg(asym_vals), 4),
    "max_asymmetry": round(max(asym_vals), 4) if asym_vals else 0,
}
print(f"  Mean |asymmetry|: {avg(asym_vals):.4f}")


# ─── 5. QUESTION LENGTH vs COMPRESSION vs ACCURACY ──────────────────────────
print("\n[5] Question complexity (length) vs compression vs accuracy...")

# Split into 3 length terciles
lengths = [len(s.get("question", "").split()) for s in all_samples]
l_sorted = sorted(lengths)
t1 = l_sorted[n//3]
t2 = l_sorted[2*n//3]

short_comp = avg_comp = long_comp = []
short_acc = avg_acc = long_acc = []
buckets = {"short": [], "medium": [], "long": []}
for s in all_samples:
    q_len = len(s.get("question", "").split())
    bucket = "short" if q_len <= t1 else ("medium" if q_len <= t2 else "long")
    buckets[bucket].append({
        "comp": s.get("compression_ratio", 0),
        "correct": 1 if s.get("is_correct") else 0,
        "tokens": s.get("total_tokens_transmitted", 0),
        "len": q_len,
    })

complexity_result = {}
for bname, items in buckets.items():
    comps = [x["comp"] for x in items]
    accs  = [x["correct"] for x in items]
    toks  = [x["tokens"] for x in items]
    lens  = [x["len"] for x in items]
    complexity_result[bname] = {
        "n": len(items),
        "avg_question_words": round(avg(lens), 1),
        "avg_compression_ratio": round(avg(comps), 4),
        "avg_accuracy": round(avg(accs), 4),
        "avg_tokens_tx": round(avg(toks), 1),
    }
    print(f"  {bname}: n={len(items)}, q_len={avg(lens):.0f}w, comp={avg(comps):.4f}, acc={avg(accs)*100:.1f}%")

report["5_complexity_vs_compression_vs_accuracy"] = complexity_result


# ─── 6. ECHO CHAMBER DEPTH (when does consensus form?) ──────────────────────
print("\n[6] Echo chamber depth — early vs late consensus...")

early_consensus_correct  = []  # unanimous in initial
late_consensus_correct   = []  # unanimous only in final
no_consensus_correct     = []

for s in all_samples:
    ic = s.get("initial_correct", {})
    fa = s.get("final_answers", {})
    init_vals = [bool(v) for v in ic.values()]
    final_vals = list(fa.values())
    is_correct = s.get("is_correct", False)

    # Early consensus: all agree in Round 0
    ic_answers = s.get("initial_answers", {})
    init_answers = list(ic_answers.values())
    c_init = Counter(str(a).strip().lower() for a in init_answers)
    init_unanimous = (c_init.most_common(1)[0][1] == len(init_answers)) if init_answers else False

    c_final = Counter(str(a).strip().lower() for a in final_vals)
    final_unanimous = (c_final.most_common(1)[0][1] == len(final_vals)) if final_vals else False

    if init_unanimous:
        early_consensus_correct.append(1 if is_correct else 0)
    elif final_unanimous:
        late_consensus_correct.append(1 if is_correct else 0)
    else:
        no_consensus_correct.append(1 if is_correct else 0)

consensus_depth = {
    "early_consensus_n":     len(early_consensus_correct),
    "early_consensus_acc":   round(avg(early_consensus_correct), 4),
    "late_consensus_n":      len(late_consensus_correct),
    "late_consensus_acc":    round(avg(late_consensus_correct), 4),
    "no_consensus_n":        len(no_consensus_correct),
    "no_consensus_acc":      round(avg(no_consensus_correct), 4),
    "late_vs_early_delta":   round(avg(late_consensus_correct) - avg(early_consensus_correct), 4),
}
report["6_consensus_depth"] = consensus_depth
for k, v in consensus_depth.items():
    print(f"  {k}: {v}")


# ─── 7. TOKEN SAVINGS DISTRIBUTION ──────────────────────────────────────────
print("\n[7] Token savings distribution...")

savings_pct = [(1 - s.get("compression_ratio", 0)) * 100 for s in all_samples]
savings_pct_sorted = sorted(savings_pct)
N = len(savings_pct_sorted)

token_dist = {
    "p10": round(savings_pct_sorted[N//10], 2),
    "p25": round(savings_pct_sorted[N//4], 2),
    "p50": round(statistics.median(savings_pct_sorted), 2),
    "p75": round(savings_pct_sorted[3*N//4], 2),
    "p90": round(savings_pct_sorted[9*N//10], 2),
    "mean": round(avg(savings_pct), 2),
    "std":  round(std(savings_pct), 2),
    "frac_saves_over_40pct": round(sum(1 for x in savings_pct if x > 40)/N, 4),
    "frac_saves_over_50pct": round(sum(1 for x in savings_pct if x > 50)/N, 4),
}
report["7_token_savings_distribution"] = token_dist
print(f"  P10={token_dist['p10']}%  P25={token_dist['p25']}%  P50={token_dist['p50']}%  P75={token_dist['p75']}%  P90={token_dist['p90']}%")
print(f"  Saves >40%: {token_dist['frac_saves_over_40pct']*100:.1f}% of samples")
print(f"  Saves >50%: {token_dist['frac_saves_over_50pct']*100:.1f}% of samples")


# ─── 8. RESCUE CONDITIONS ────────────────────────────────────────────────────
print("\n[8] Rescue condition analysis...")

rescued_comps = []
rescued_density = []
rescued_q_lens  = []
non_rescued_comps = []
non_rescued_density = []
non_rescued_q_lens  = []

for s in all_samples:
    ic = s.get("initial_correct", {})
    vals = [bool(v) for v in ic.values()]
    init_mv_correct = sum(vals) > len(vals)/2 if vals else True
    is_correct = s.get("is_correct", False)
    rescued = not init_mv_correct and is_correct

    comp = s.get("compression_ratio", 0)
    density = s.get("communication_density", 0)
    q_len = len(s.get("question", "").split())

    if rescued:
        rescued_comps.append(comp)
        rescued_density.append(density)
        rescued_q_lens.append(q_len)
    else:
        non_rescued_comps.append(comp)
        non_rescued_density.append(density)
        non_rescued_q_lens.append(q_len)

rescue_analysis = {
    "total_rescued": len(rescued_comps),
    "rescue_rate": round(len(rescued_comps)/n, 4),
    "rescued_avg_compression": round(avg(rescued_comps), 4),
    "non_rescued_avg_compression": round(avg(non_rescued_comps), 4),
    "rescued_avg_density": round(avg(rescued_density), 4),
    "non_rescued_avg_density": round(avg(non_rescued_density), 4),
    "rescued_avg_question_len_words": round(avg(rescued_q_lens), 1),
    "non_rescued_avg_question_len_words": round(avg(non_rescued_q_lens), 1),
}
report["8_rescue_conditions"] = rescue_analysis
print(f"  Rescued samples: {len(rescued_comps)}/{n} ({len(rescued_comps)/n*100:.1f}%)")
print(f"  Rescued avg comp: {avg(rescued_comps):.4f} vs non-rescued: {avg(non_rescued_comps):.4f}")
print(f"  Rescued avg density: {avg(rescued_density):.4f} vs non-rescued: {avg(non_rescued_density):.4f}")
print(f"  Rescued avg q_len: {avg(rescued_q_lens):.1f} vs non-rescued: {avg(non_rescued_q_lens):.1f}")


# ─── 9. MODE SWITCHING ACROSS ROUNDS ────────────────────────────────────────
print("\n[9] Mode stability: do modes switch across rounds?")

round_modes = defaultdict(lambda: defaultdict(int))
for s in all_samples:
    rcs = s.get("raw_communications", [])
    for rc in rcs:
        rnum = rc.get("round", 0)
        wm = rc.get("weights", [])
        for row in wm:
            for w in row:
                if w == 0:
                    continue
                if w < 0.25:
                    mode = "BLOCK"
                elif w < 0.5:
                    mode = "SUMMARIZE"
                elif w < 0.75:
                    mode = "FILTER"
                else:
                    mode = "FULL"
                round_modes[rnum][mode] += 1

mode_by_round = {}
for rnum in sorted(round_modes.keys()):
    total = sum(round_modes[rnum].values())
    mode_by_round[f"round_{rnum}"] = {mode: round(cnt/total, 4) for mode, cnt in round_modes[rnum].items()}
    print(f"  Round {rnum}: {mode_by_round[f'round_{rnum}']}")

report["9_mode_distribution_by_round"] = mode_by_round


# ─── 10. AGENT DOMINANCE ────────────────────────────────────────────────────
print("\n[10] Agent dominance — which agent is most often correct at end?")

agent_final_correct = defaultdict(list)
for s in all_samples:
    fa = s.get("final_answers", {})
    gold = s.get("gold_answer", "")
    gold_lower = str(gold).strip().lower()
    for ag_id, ans in fa.items():
        ans_lower = str(ans).strip().lower()
        nums_a = re.findall(r"-?\d+\.?\d*", ans_lower)
        nums_g = re.findall(r"-?\d+\.?\d*", gold_lower)
        if nums_a and nums_g:
            try:
                correct = abs(float(nums_a[-1]) - float(nums_g[-1])) < 1e-3
            except:
                correct = ans_lower == gold_lower
        else:
            correct = ans_lower == gold_lower or gold_lower in ans_lower
        agent_final_correct[ag_id].append(1 if correct else 0)

dominance = {}
for ag_id, vals in sorted(agent_final_correct.items()):
    dominance[f"agent_{ag_id}"] = {
        "final_accuracy": round(avg(vals), 4),
        "n_samples": len(vals),
    }
    print(f"  Agent {ag_id}: final accuracy = {avg(vals)*100:.2f}%  (n={len(vals)})")

report["10_agent_dominance"] = dominance

# Identify dominant agent
best_agent = max(dominance.items(), key=lambda x: x[1]["final_accuracy"])
worst_agent = min(dominance.items(), key=lambda x: x[1]["final_accuracy"])
report["10_agent_dominance"]["_summary"] = {
    "dominant_agent": best_agent[0],
    "dominant_accuracy": best_agent[1]["final_accuracy"],
    "weakest_agent": worst_agent[0],
    "weakest_accuracy": worst_agent[1]["final_accuracy"],
    "dominance_gap": round(best_agent[1]["final_accuracy"] - worst_agent[1]["final_accuracy"], 4),
}
print(f"  Dominance gap: {report['10_agent_dominance']['_summary']['dominance_gap']*100:.2f}%")


# ─── 11. INFORMATION BOTTLENECK TIGHTNESS ───────────────────────────────────
print("\n[11] IB tightness: weight variance as a proxy for controller confidence...")

per_sample_weight_var = []
per_sample_weight_mean = []
for s in all_samples:
    wms = s.get("weight_matrices", [])
    if not wms:
        continue
    wm = wms[-1]
    flat = [w for row in wm for w in row if w != 0.0]
    if flat:
        per_sample_weight_var.append(statistics.variance(flat) if len(flat) > 1 else 0)
        per_sample_weight_mean.append(avg(flat))

# Correct samples vs incorrect sample weight variance
correct_vars = [per_sample_weight_var[i] for i, s in enumerate(all_samples) if s.get("is_correct") and i < len(per_sample_weight_var)]
incorrect_vars = [per_sample_weight_var[i] for i, s in enumerate(all_samples) if not s.get("is_correct") and i < len(per_sample_weight_var)]

ib_tightness = {
    "overall_avg_weight_variance": round(avg(per_sample_weight_var), 6),
    "overall_avg_weight_mean": round(avg(per_sample_weight_mean), 4),
    "correct_avg_weight_variance": round(avg(correct_vars), 6),
    "incorrect_avg_weight_variance": round(avg(incorrect_vars), 6),
    "variance_diff_correct_vs_incorrect": round(avg(correct_vars) - avg(incorrect_vars), 6),
}
report["11_ib_tightness"] = ib_tightness
print(f"  Correct   weight var: {avg(correct_vars):.6f}")
print(f"  Incorrect weight var: {avg(incorrect_vars):.6f}")
print(f"  Diff: {ib_tightness['variance_diff_correct_vs_incorrect']:+.6f}")


# ─── 12. FIRST-TOKEN ANSWER CHANGE RATE ─────────────────────────────────────
print("\n[12] Answer change rate: how often does communication change the answer?")

changed = unchanged = 0
changed_to_correct = changed_to_wrong = unchanged_correct = unchanged_wrong = 0
for s in all_samples:
    ia = s.get("initial_answers", {})
    fa = s.get("final_answers", {})
    ic = s.get("initial_correct", {})
    is_correct = s.get("is_correct", False)
    # Sample-level: did the majority answer change?
    init_mv = Counter(str(v).strip().lower() for v in ia.values()).most_common(1)[0][0] if ia else ""
    final_mv = Counter(str(v).strip().lower() for v in fa.values()).most_common(1)[0][0] if fa else ""
    init_mv_correct = any(bool(v) for v in ic.values())

    if init_mv != final_mv:
        changed += 1
        if is_correct:    changed_to_correct += 1
        else:              changed_to_wrong += 1
    else:
        unchanged += 1
        if is_correct:    unchanged_correct += 1
        else:              unchanged_wrong += 1

answer_change = {
    "changed_n": changed,
    "unchanged_n": unchanged,
    "change_rate": round(changed/n, 4),
    "when_changed_acc": round(changed_to_correct/changed, 4) if changed else 0,
    "when_unchanged_acc": round(unchanged_correct/unchanged, 4) if unchanged else 0,
    "accuracy_gain_from_change": round(changed_to_correct/changed - unchanged_correct/unchanged, 4) if changed and unchanged else 0,
}
report["12_answer_change_rate"] = answer_change
print(f"  Changed:   {changed}/{n} ({changed/n*100:.1f}%) | acc when changed: {answer_change['when_changed_acc']*100:.1f}%")
print(f"  Unchanged: {unchanged}/{n} ({unchanged/n*100:.1f}%) | acc when unchanged: {answer_change['when_unchanged_acc']*100:.1f}%")
print(f"  Accuracy gain from changing: {answer_change['accuracy_gain_from_change']*100:+.2f}%")


# ─── SAVE ────────────────────────────────────────────────────────────────────
class Enc(json.JSONEncoder):
    def default(self, o):
        try: return super().default(o)
        except: return str(o)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, cls=Enc)

print(f"\n\nFull hidden patterns saved to: {OUTPUT}")
