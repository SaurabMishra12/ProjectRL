"""
EMNLP-Ready Comparative Analysis: With vs Without Sparse Communication
=======================================================================
Run:  python analyze_full.py
Output: resultsall/full_analysis.json  +  console report
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json, re, math, statistics, pathlib
from collections import defaultdict, Counter

try:
    from scipy import stats as sc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not found — p-values will be approximated.")

RESULTS_DIR = pathlib.Path(r"c:\Users\msaur\OneDrive\Desktop\projectRL\resultsall")
OUTPUT      = RESULTS_DIR / "full_analysis.json"

# ─────────────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────────────

def classify_dataset(method_key: str, questions: list) -> str:
    mk = method_key.lower()
    if "hotpot" in mk:
        return "hotpotqa"
    if "math" in mk or "aqua" in mk:
        return "aqua_math"
    # Heuristic: AQuA questions start with "aqua math problem"
    if questions:
        q0 = questions[0].get("question", "").lower()
        if "aqua" in q0 or "options:" in q0:
            return "aqua_math"
        # HotpotQA questions are typically short factual queries without numbers
        nums = re.findall(r"\d+", q0)
        if len(nums) == 0 and len(q0) < 200:
            return "hotpotqa"
    return "gsm8k"


def load_all():
    merged = defaultdict(dict)   # merged[ds][question_text] = sample
    file_inventory = []

    for jf in sorted(RESULTS_DIR.glob("*.json")):
        if jf.name in ("analysis.json", "full_analysis.json"):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [SKIP-PARSE] {jf.name}: {e}")
            continue

        detailed = data.get("detailed_samples", {})
        if not detailed:
            print(f"  [SKIP-EMPTY] {jf.name}")
            continue

        file_counts = {}
        for method_key, samples in detailed.items():
            if not isinstance(samples, list) or not samples:
                continue
            ds = classify_dataset(method_key, samples)
            added = 0
            for s in samples:
                q = s.get("question", "")
                if q and q not in merged[ds]:
                    merged[ds][q] = s
                    added += 1
            file_counts[f"{method_key}→{ds}"] = (len(samples), added)

        file_inventory.append({"file": jf.name, "size_kb": round(jf.stat().st_size/1024, 1), "counts": file_counts})
        print(f"  {jf.name:45s} {file_counts}")

    return {ds: list(qmap.values()) for ds, qmap in merged.items()}, file_inventory


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def check_answer(pred: str, gold: str, ds: str) -> bool:
    pred = str(pred).strip().lower().rstrip(".,")
    gold = str(gold).strip().lower().rstrip(".,")
    if ds == "gsm8k":
        nums_p = re.findall(r"-?\d+\.?\d*", pred)
        nums_g = re.findall(r"-?\d+\.?\d*", gold)
        if nums_p and nums_g:
            try:
                return abs(float(nums_p[-1]) - float(nums_g[-1])) < 1e-3
            except:
                pass
    elif ds == "aqua_math":
        # AQuA: letter answer A-E
        m_p = re.search(r"\\?boxed\{([a-eA-E])\}", pred)
        m_g = re.search(r"\\?boxed\{([a-eA-E])\}", gold)
        letter_p = m_p.group(1).upper() if m_p else re.search(r"\b([A-E])\b", pred.upper())
        letter_g = m_g.group(1).upper() if m_g else gold.strip().upper()
        if hasattr(letter_p, 'group'):
            letter_p = letter_p.group(1)
        return str(letter_p) == str(letter_g) if letter_p else (gold in pred or pred == gold)
    else:
        # HotpotQA: substring / overlap
        return pred == gold or gold in pred or pred in gold
    return pred == gold


def majority_vote(answers: list) -> str:
    """Simple majority vote over a list of string answers."""
    if not answers:
        return ""
    c = Counter(str(a).strip().lower() for a in answers)
    return c.most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE (no comm) — Round 0 majority vote of initial independent answers
# ─────────────────────────────────────────────────────────────────────────────

def compute_baseline_accuracy(samples: list, ds: str) -> dict:
    """Compute accuracy using stored initial_correct field (pipeline-consistent).
    majority_vote_accuracy = majority of agents correct in Round 0.
    """
    correct_majority = []
    correct_any      = []
    correct_all      = []

    for s in samples:
        ic = s.get("initial_correct", {})  # {"0": bool, "1": bool, ...}
        vals = [bool(v) for v in ic.values()]
        if not vals:
            continue
        # majority vote
        mv_correct = sum(vals) > len(vals) / 2
        correct_majority.append(1 if mv_correct else 0)
        correct_any.append(1 if any(vals) else 0)
        correct_all.append(1 if all(vals) else 0)

    n = len(correct_majority)
    avg = lambda x: sum(x)/len(x) if x else 0
    return {
        "n": n,
        "majority_vote_accuracy": round(avg(correct_majority), 4),
        "any_agent_correct":      round(avg(correct_any), 4),
        "all_agents_correct":     round(avg(correct_all), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# WITH SPARSE COMM metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_sparse_comm_metrics(samples: list, ds: str) -> dict:
    n = len(samples)
    if n == 0:
        return {}

    correct       = [1 if s.get("is_correct") else 0 for s in samples]
    tokens_tx     = [s.get("total_tokens_transmitted", 0) for s in samples]
    tokens_full   = [s.get("total_tokens_if_full", 0) for s in samples]
    comp_ratios   = [s.get("compression_ratio", 0) for s in samples]
    gen_tokens    = [s.get("generation_tokens", 0) for s in samples]
    density       = [s.get("communication_density", 0) for s in samples]

    avg = lambda x: sum(x)/len(x) if x else 0
    std = lambda x: statistics.stdev(x) if len(x) > 1 else 0
    med = lambda x: statistics.median(x) if x else 0

    return {
        "n": n,
        "final_accuracy":          round(avg(correct), 4),
        "std_accuracy":            round(std(correct), 4),
        "ci95":                    round(1.96 * std(correct) / math.sqrt(n), 4),
        "avg_tokens_transmitted":  round(avg(tokens_tx), 1),
        "std_tokens_transmitted":  round(std(tokens_tx), 1),
        "avg_tokens_if_full":      round(avg(tokens_full), 1),
        "avg_compression_ratio":   round(avg(comp_ratios), 4),
        "std_compression_ratio":   round(std(comp_ratios), 4),
        "median_compression_ratio":round(med(comp_ratios), 4),
        "avg_tokens_saved_pct":    round((1 - avg(comp_ratios)) * 100, 2),
        "avg_generation_tokens":   round(avg(gen_tokens), 1),
        "avg_communication_density": round(avg(density), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EFFICIENCY RATIO
# ─────────────────────────────────────────────────────────────────────────────

def compute_efficiency(baseline: dict, sparse: dict) -> dict:
    """How many tokens were spent per percentage point of accuracy gain?"""
    acc_gain_pct = (sparse["final_accuracy"] - baseline["majority_vote_accuracy"]) * 100
    tokens_saved = sparse["avg_tokens_if_full"] - sparse["avg_tokens_transmitted"]
    tokens_spent = sparse["avg_tokens_transmitted"]

    # Tokens per +1% accuracy gain (lower = better)
    toks_per_acc_pt = tokens_spent / acc_gain_pct if acc_gain_pct > 0 else float("inf")
    # Token savings per +1% accuracy gain
    savings_per_acc_pt = tokens_saved / acc_gain_pct if acc_gain_pct > 0 else float("inf")

    return {
        "accuracy_gain_pct":         round(acc_gain_pct, 2),
        "tokens_per_acc_point":      round(toks_per_acc_pt, 1) if math.isfinite(toks_per_acc_pt) else "N/A",
        "token_savings_per_acc_pt":  round(savings_per_acc_pt, 1) if math.isfinite(savings_per_acc_pt) else "N/A",
        "net_token_saving_pct":      round((1 - sparse["avg_compression_ratio"]) * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL TESTS (with vs without comm on same samples)
# ─────────────────────────────────────────────────────────────────────────────

def stats_with_vs_without(samples: list, ds: str) -> dict:
    """McNemar + paired t-test comparing baseline vs post-comm correctness."""
    results = {}
    baseline_correct = []
    final_correct    = []

    for s in samples:
        # Use stored initial_correct majority vote (pipeline-consistent)
        ic = s.get("initial_correct", {})
        vals = [bool(v) for v in ic.values()]
        b = 1 if (sum(vals) > len(vals) / 2) else 0
        f = 1 if s.get("is_correct") else 0
        baseline_correct.append(b)
        final_correct.append(f)

    # McNemar: b=correct→wrong, c=wrong→correct
    b_count = sum(1 for b, f in zip(baseline_correct, final_correct) if b == 1 and f == 0)
    c_count = sum(1 for b, f in zip(baseline_correct, final_correct) if b == 0 and f == 1)
    results["mcnemar_b_count"] = b_count  # degraded
    results["mcnemar_c_count"] = c_count  # improved

    if b_count + c_count > 0:
        chi2 = ((abs(b_count - c_count) - 1) ** 2) / (b_count + c_count)
        if HAS_SCIPY:
            p = float(sc.chi2.sf(chi2, df=1))
        else:
            # Rough normal approx
            z = (abs(b_count - c_count) - 1) / math.sqrt(b_count + c_count)
            p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
        results["mcnemar_chi2"] = round(chi2, 4)
        results["mcnemar_p"]    = round(p, 8)
        results["mcnemar_sig"]  = p < 0.05
    else:
        results["mcnemar_chi2"] = 0.0
        results["mcnemar_p"]    = 1.0
        results["mcnemar_sig"]  = False

    # Paired t-test
    n = len(baseline_correct)
    if n >= 5:
        if HAS_SCIPY:
            t, tp = sc.ttest_rel(final_correct, baseline_correct)
            results["paired_ttest_t"] = round(float(t), 4)
            results["paired_ttest_p"] = round(float(tp), 8)
            results["paired_ttest_sig"] = tp < 0.05
        else:
            diffs = [f - b for f, b in zip(final_correct, baseline_correct)]
            mean_d = sum(diffs) / n
            std_d  = statistics.stdev(diffs) if n > 1 else 1
            t_approx = mean_d / (std_d / math.sqrt(n)) if std_d > 0 else 0
            results["paired_ttest_t"] = round(t_approx, 4)
            results["paired_ttest_p"] = "scipy_required"
            results["paired_ttest_sig"] = None

    # Wilcoxon on compression ratios vs 0.5 (random baseline)
    comp_ratios = [s.get("compression_ratio", 0.5) for s in samples]
    if HAS_SCIPY and len(comp_ratios) >= 5:
        try:
            w, wp = sc.wilcoxon([c - 0.5 for c in comp_ratios])
            results["wilcoxon_comp_vs_half_p"] = round(float(wp), 8)
            results["wilcoxon_comp_sig"]       = wp < 0.05
        except:
            pass

    # Effect size (Cohen's d for accuracy gain)
    diff_mean = sum(f - b for f, b in zip(final_correct, baseline_correct)) / n
    diff_std  = statistics.stdev([f - b for f, b in zip(final_correct, baseline_correct)]) if n > 1 else 0
    results["cohens_d"] = round(diff_mean / diff_std, 4) if diff_std > 0 else 0.0

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ROUND-BY-ROUND AGENT DRIFT
# ─────────────────────────────────────────────────────────────────────────────

def agent_drift(samples: list, ds: str) -> dict:
    improved = degraded = stable_correct = stable_wrong = total = 0
    rescued_samples = degraded_samples = echo_wrong = 0

    for s in samples:
        ic = s.get("initial_correct", {})
        fa = s.get("final_answers", {})
        is_correct = s.get("is_correct", False)

        # Sample-level: initial majority vote
        vals = [bool(v) for v in ic.values()]
        init_mv_correct = sum(vals) > len(vals) / 2 if vals else False
        if not init_mv_correct and is_correct:
            rescued_samples += 1
        if init_mv_correct and not is_correct:
            degraded_samples += 1

        # Echo chamber: wrong + unanimous final
        final_answers = list(fa.values())
        if not is_correct and final_answers:
            c = Counter(str(a).strip().lower() for a in final_answers)
            if c.most_common(1)[0][1] == len(final_answers):
                echo_wrong += 1

        # Per-agent: use stored initial_correct + compare final answers with gold
        gold = s.get("gold_answer", "")
        for ag_id, init_ok in ic.items():
            fin_ans = fa.get(ag_id, "")
            fin_ok  = check_answer(fin_ans, gold, ds) if gold and fin_ans else bool(init_ok)
            total += 1
            if init_ok and fin_ok:       stable_correct += 1
            elif init_ok and not fin_ok: degraded += 1
            elif not init_ok and fin_ok: improved += 1
            else:                        stable_wrong += 1

    n_samp = len(samples)
    n_ag   = total if total > 0 else 1
    return {
        "per_agent_stayed_correct":   round(stable_correct/n_ag, 4),
        "per_agent_rescued":          round(improved/n_ag, 4),
        "per_agent_degraded":         round(degraded/n_ag, 4),
        "per_agent_stayed_wrong":     round(stable_wrong/n_ag, 4),
        "per_sample_rescued":         round(rescued_samples/n_samp, 4),
        "per_sample_degraded":        round(degraded_samples/n_samp, 4),
        "echo_chamber_rate":          round(echo_wrong/n_samp, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def mode_dist(samples: list) -> dict:
    total = defaultdict(int)
    for s in samples:
        for k, v in s.get("mode_distribution", {}).items():
            total[k] += v
    grand = sum(total.values()) or 1
    return {k: round(v/grand, 4) for k, v in total.items()}


# ─────────────────────────────────────────────────────────────────────────────
# PRINTING
# ─────────────────────────────────────────────────────────────────────────────

SEP = "=" * 74

def pr(title=""):
    print("\n" + SEP)
    if title: print(f"  {title}")
    print(SEP)


def pct(v): return f"{v*100:.2f}%"
def pm(v, e): return f"{v*100:.2f}% ± {e*100:.2f}%"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

DS_LABEL = {
    "gsm8k":     "GSM8K (Math Word Problems)",
    "aqua_math": "DeepMind AQuA (Algebra MCQ)",
    "hotpotqa":  "HotpotQA (Multi-hop Factual)",
}

if __name__ == "__main__":
    pr("SPARSE SEMANTIC COMMUNICATION — FULL COMPARATIVE ANALYSIS")
    print("  Loading result files …")
    all_data, inventory = load_all()

    if not all_data:
        print("[ERROR] No data loaded. Check resultsall/ directory.")
        exit(1)

    total_samples = sum(len(v) for v in all_data.values())
    print(f"\n  Datasets: {list(all_data.keys())}")
    print(f"  Total unique samples: {total_samples}")

    full_report = {"inventory": inventory, "datasets": {}}

    # ── Per-dataset analysis ──────────────────────────────────────────────────
    for ds, samples in all_data.items():
        label = DS_LABEL.get(ds, ds)
        pr(f"DATASET: {label}  (n={len(samples)})")

        base  = compute_baseline_accuracy(samples, ds)
        sparse = compute_sparse_comm_metrics(samples, ds)
        eff   = compute_efficiency(base, sparse)
        stat  = stats_with_vs_without(samples, ds)
        drift = agent_drift(samples, ds)
        modes = mode_dist(samples)

        print(f"\n  ┌─ ACCURACY: Without vs With Sparse Communication ─────────────────┐")
        print(f"  │  WITHOUT (Round-0 majority vote):  {pct(base['majority_vote_accuracy']):>8}  (any agent: {pct(base['any_agent_correct'])})")
        print(f"  │  WITH Sparse Communication:        {pm(sparse['final_accuracy'], sparse['ci95']):>22}")
        print(f"  │  Accuracy Gain (Δ):                {eff['accuracy_gain_pct']:>+7.2f}%")
        print(f"  └────────────────────────────────────────────────────────────────────┘")

        print(f"\n  ┌─ TOKEN EFFICIENCY ────────────────────────────────────────────────┐")
        print(f"  │  Avg Tokens Sent (with comm):      {sparse['avg_tokens_transmitted']:>8.0f}  ± {sparse['std_tokens_transmitted']:.0f}")
        print(f"  │  Avg Tokens If Full Broadcast:     {sparse['avg_tokens_if_full']:>8.0f}")
        print(f"  │  Token Savings vs Full Broadcast:  {sparse['avg_tokens_saved_pct']:>7.2f}%")
        print(f"  │  Compression Ratio:                {sparse['avg_compression_ratio']:.4f}  ± {sparse['std_compression_ratio']:.4f}  (median {sparse['median_compression_ratio']:.4f})")
        print(f"  │  Tokens per +1% Accuracy Gained:  {str(eff['tokens_per_acc_point']):>8}")
        print(f"  │  Token Savings per +1% Accuracy:  {str(eff['token_savings_per_acc_pt']):>8}")
        print(f"  └────────────────────────────────────────────────────────────────────┘")

        print(f"\n  ┌─ STATISTICAL TESTS ────────────────────────────────────────────────┐")
        mc_sig = "✅ p<0.05 SIGNIFICANT" if stat.get("mcnemar_sig") else "✗ Not significant"
        tt_sig = "✅ p<0.05 SIGNIFICANT" if stat.get("paired_ttest_sig") else "✗ Not significant"
        wil_sig = "✅ p<0.05 SIGNIFICANT" if stat.get("wilcoxon_comp_sig") else "✗ Not significant"
        print(f"  │  McNemar χ²={stat.get('mcnemar_chi2','n/a'):<8}  p={stat.get('mcnemar_p','n/a'):<12}  → {mc_sig}")
        print(f"  │  Paired t-test t={stat.get('paired_ttest_t','n/a'):<7}  p={stat.get('paired_ttest_p','n/a'):<12}  → {tt_sig}")
        print(f"  │  Wilcoxon (compression vs 0.5):  p={stat.get('wilcoxon_comp_vs_half_p','n/a'):<12}  → {wil_sig}")
        print(f"  │  Cohen's d (effect size):         {stat.get('cohens_d', 'n/a')}")
        print(f"  │  McNemar: improved={stat.get('mcnemar_c_count',0)}  degraded={stat.get('mcnemar_b_count',0)}")
        print(f"  └────────────────────────────────────────────────────────────────────┘")

        print(f"\n  ┌─ AGENT DRIFT (per-agent across rounds) ───────────────────────────┐")
        print(f"  │  Stayed Correct:   {drift['per_agent_stayed_correct']*100:5.1f}%  │  Rescued (W→C): {drift['per_agent_rescued']*100:5.1f}%")
        print(f"  │  Degraded (C→W):   {drift['per_agent_degraded']*100:5.1f}%  │  Stayed Wrong:  {drift['per_agent_stayed_wrong']*100:5.1f}%")
        print(f"  │  Sample-level Rescues:   {drift['per_sample_rescued']*100:5.1f}%  │  Degradations: {drift['per_sample_degraded']*100:5.1f}%")
        print(f"  │  Echo Chamber Rate (wrong+unanimous): {drift['echo_chamber_rate']*100:.1f}%")
        print(f"  └────────────────────────────────────────────────────────────────────┘")

        print(f"\n  ┌─ MODE DISTRIBUTION ───────────────────────────────────────────────┐")
        for mode, pct_val in modes.items():
            bar = "█" * int(pct_val * 30)
            print(f"  │  {mode:<12}  {bar:<30}  {pct_val*100:.1f}%")
        print(f"  └────────────────────────────────────────────────────────────────────┘")

        full_report["datasets"][ds] = {
            "label": label, "n": len(samples),
            "baseline": base, "sparse_comm": sparse,
            "efficiency": eff, "stats": stat,
            "drift": drift, "modes": modes,
        }

    # ── Master comparison table ───────────────────────────────────────────────
    pr("MASTER COMPARISON TABLE (EMNLP READY)")
    hdr = (f"{'Dataset':<14} {'N':>5} "
           f"{'No-Comm Acc':>12} {'Sparse Acc':>12} {'Δ Acc':>8} "
           f"{'Token Save':>11} {'Comp Ratio':>11} "
           f"{'McNemar p':>12} {'Cohen d':>9}")
    print(hdr)
    print("─" * len(hdr))
    for ds, d in full_report["datasets"].items():
        b, s, e, st = d["baseline"], d["sparse_comm"], d["efficiency"], d["stats"]
        p_val = st.get("mcnemar_p", "n/a")
        p_str = f"{p_val:.2e}" if isinstance(p_val, float) else str(p_val)
        print(
            f"{ds:<14} {d['n']:>5} "
            f"{b['majority_vote_accuracy']*100:>11.2f}% "
            f"{s['final_accuracy']*100:>11.2f}% "
            f"{e['accuracy_gain_pct']:>+7.2f}% "
            f"{s['avg_tokens_saved_pct']:>10.2f}% "
            f"{s['avg_compression_ratio']:>11.4f} "
            f"{p_str:>12} "
            f"{st.get('cohens_d', 0):>9.4f}"
        )

    # ── Efficiency highlight ──────────────────────────────────────────────────
    pr("EFFICIENCY ANALYSIS — Tokens Spent per Accuracy Point")
    for ds, d in full_report["datasets"].items():
        e = d["efficiency"]
        s = d["sparse_comm"]
        print(f"\n  {DS_LABEL.get(ds, ds)}")
        print(f"    Accuracy gain:           {e['accuracy_gain_pct']:+.2f}%")
        print(f"    Token savings:           {e['net_token_saving_pct']:.2f}% vs full broadcast")
        print(f"    Cost (tokens/+1% acc):   {e['tokens_per_acc_point']}")
        print(f"    Savings (tokens/+1% acc):{e['token_savings_per_acc_pt']}")
        print(f"    Avg generation tokens:   {s['avg_generation_tokens']:.0f}")

    # ── Save output ───────────────────────────────────────────────────────────
    class Enc(json.JSONEncoder):
        def default(self, o):
            try: return super().default(o)
            except: return str(o)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, cls=Enc)

    print(f"\n\n✅ Full analysis saved → {OUTPUT}")
    print("\n📌 Quick-cite numbers for Table 1:")
    for ds, d in full_report["datasets"].items():
        b, s, e = d["baseline"], d["sparse_comm"], d["efficiency"]
        print(f"  {ds:<12}: No-Comm={b['majority_vote_accuracy']*100:.2f}%  "
              f"Sparse={s['final_accuracy']*100:.2f}%  "
              f"Δ={e['accuracy_gain_pct']:+.2f}%  "
              f"TokenSave={s['avg_tokens_saved_pct']:.2f}%  "
              f"CompRatio={s['avg_compression_ratio']:.4f}")
