import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/saurab/Documents/projectRLMAIN/08-06results'
CONTINUOUS = os.path.join(BASE, 'continuous')
PLOTS = os.path.join(BASE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

# Helper: compute Jaccard similarity of two strings
def get_jaccard_sim(str1, str2):
    words1 = set(re.findall(r'\w+', str1.lower()))
    words2 = set(re.findall(r'\w+', str2.lower()))
    if not words1 or not words2:
        return 0.0
    return len(words1.intersection(words2)) / len(words1.union(words2))

# Helper: parse confidence from text
def extract_confidence(text):
    match = re.search(r'Confidence:\s*([\d\.]+)', text, re.IGNORECASE)
    if match:
        val_str = match.group(1)
        try:
            return float(val_str)
        except ValueError:
            return None
    return None

def analyze_claims():
    print("Starting Claim Verification Analysis...")
    
    # ── Datasets/files ────────────────────────────────────────────────────────
    transfer_files = [
        os.path.join(CONTINUOUS, 'Transfer_GSM8K_seed0_gsm8k_stream.jsonl'),
        os.path.join(CONTINUOUS, 'Transfer_GSM8K_seed1_gsm8k_stream.jsonl'),
        os.path.join(CONTINUOUS, 'Transfer_GSM8K_seed2_gsm8k_stream.jsonl')
    ]
    mad_files = [
        os.path.join(CONTINUOUS, 'Multi-Agent Debate_seed0_gsm8k_stream.jsonl'),
        os.path.join(CONTINUOUS, 'Multi-Agent Debate_seed1_gsm8k_stream.jsonl'),
        os.path.join(CONTINUOUS, 'Multi-Agent Debate_seed2_gsm8k_stream.jsonl')
    ]
    nocomm_files = [
        os.path.join(CONTINUOUS, 'No Communication_seed0_gsm8k_stream.jsonl'),
        os.path.join(CONTINUOUS, 'No Communication_seed1_gsm8k_stream.jsonl'),
        os.path.join(CONTINUOUS, 'No Communication_seed2_gsm8k_stream.jsonl')
    ]
    
    # Load records
    transfer_records = []
    for fpath in transfer_files:
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        transfer_records.append(json.loads(line))
                        
    mad_records = []
    for fpath in mad_files:
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        mad_records.append(json.loads(line))
                        
    nocomm_records = []
    for fpath in nocomm_files:
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        nocomm_records.append(json.loads(line))
                        
    print(f"Loaded {len(transfer_records)} Transfer (SparseComm), {len(mad_records)} MAD, {len(nocomm_records)} NoComm records.")

    # ══════════════════════════════════════════════════════════════════════════
    # CLAIM 1 & 7: Lexical Convergence & Syntax Segregation (Groupthink in MAD)
    # ══════════════════════════════════════════════════════════════════════════
    # We measure percentage of samples where all 3 agents have character-for-character identical final reasoning
    # and average Jaccard similarity of final reasonings.
    
    def compute_homogenization_stats(records):
        identical_count = 0
        jaccard_sims = []
        total_valid = 0
        
        for r in records:
            # We look at records with halt_round > 0 (meaning communication took place)
            # or if halt_round is 0 but we want to inspect final reasoning convergence.
            # Let's check final reasonings
            fr = r.get('final_reasoning', {})
            if '0' in fr and '1' in fr and '2' in fr:
                r0, r1, r2 = fr['0'], fr['1'], fr['2']
                # Check for identical clones
                if r0.strip() == r1.strip() == r2.strip():
                    identical_count += 1
                
                # Compute Jaccard
                s01 = get_jaccard_sim(r0, r1)
                s02 = get_jaccard_sim(r0, r2)
                s12 = get_jaccard_sim(r1, r2)
                jaccard_sims.append((s01 + s02 + s12) / 3)
                total_valid += 1
                
        pct_identical = (identical_count / total_valid * 100) if total_valid > 0 else 0
        avg_jaccard = np.mean(jaccard_sims) if jaccard_sims else 0
        return pct_identical, avg_jaccard, total_valid

    tr_pct_id, tr_avg_jac, tr_valid = compute_homogenization_stats(transfer_records)
    mad_pct_id, mad_avg_jac, mad_valid = compute_homogenization_stats(mad_records)
    no_pct_id, no_avg_jac, no_valid = compute_homogenization_stats(nocomm_records)
    
    print("\n--- Lexical Convergence / Groupthink ---")
    print(f"Transfer (Ours) - Character-identical final reasonings across all 3 agents: {tr_pct_id:.2f}% (Avg Jaccard: {tr_avg_jac:.4f})")
    print(f"MAD (Debate)   - Character-identical final reasonings across all 3 agents: {mad_pct_id:.2f}% (Avg Jaccard: {mad_avg_jac:.4f})")
    print(f"No Comm        - Character-identical final reasonings across all 3 agents: {no_pct_id:.2f}% (Avg Jaccard: {no_avg_jac:.4f})")

    # ══════════════════════════════════════════════════════════════════════════
    # CLAIM 2: Confidence Calibration
    # ══════════════════════════════════════════════════════════════════════════
    # We check confidence values: crude integers (1-10) vs continuous floats (0-1)
    
    def analyze_confidence_calibration(records):
        all_confs = []
        integer_count = 0
        float_count = 0
        conf_accuracy_pairs = []
        
        for r in records:
            fr = r.get('final_reasoning', {})
            # Look at agent confidences and correctness
            # We can map each agent's correctness to its self-reported confidence.
            # Or use final majority answer correctness vs average agent confidence.
            agent_correct = r.get('final_answers', {})
            gold = r.get('gold_answer')
            
            for agent_id, reasoning in fr.items():
                conf = extract_confidence(reasoning)
                if conf is not None:
                    all_confs.append(conf)
                    # Determine if it's float in [0, 1] or integer in [1, 10]
                    # We define integer if it equals its rounded value and is >= 1.0 (or specifically 9.0, 10.0, etc.)
                    # Floats are typically 0.99, 0.98, etc.
                    is_integer = (conf == int(conf)) and (conf >= 1.0)
                    if is_integer:
                        integer_count += 1
                    else:
                        float_count += 1
                    
                    # Track for calibration curve
                    ans = agent_correct.get(agent_id)
                    # For GSM8K, we evaluate if answer equals gold (numeric or string)
                    is_correct = False
                    if ans is not None and gold is not None:
                        # Clean answers and compare
                        clean_ans = re.sub(r'[^\d\.\-]', '', str(ans))
                        clean_gold = re.sub(r'[^\d\.\-]', '', str(gold))
                        is_correct = (clean_ans == clean_gold) and (clean_ans != '')
                    
                    conf_accuracy_pairs.append((conf, is_correct))
                    
        total_conf = len(all_confs)
        pct_integer = (integer_count / total_conf * 100) if total_conf > 0 else 0
        pct_float = (float_count / total_conf * 100) if total_conf > 0 else 0
        return pct_integer, pct_float, conf_accuracy_pairs

    tr_pct_int, tr_pct_flt, tr_pairs = analyze_confidence_calibration(transfer_records)
    mad_pct_int, mad_pct_flt, mad_pairs = analyze_confidence_calibration(mad_records)
    no_pct_int, no_pct_flt, no_pairs = analyze_confidence_calibration(nocomm_records)
    
    print("\n--- Confidence Calibration Reporting Style ---")
    print(f"Transfer (Ours) - Crude integers (>=1): {tr_pct_int:.2f}%, Granular probabilities (<1): {tr_pct_flt:.2f}%")
    print(f"MAD (Debate)   - Crude integers (>=1): {mad_pct_int:.2f}%, Granular probabilities (<1): {mad_pct_flt:.2f}%")
    print(f"No Comm        - Crude integers (>=1): {no_pct_int:.2f}%, Granular probabilities (<1): {no_pct_flt:.2f}%")

    # Let's compute calibration curves: bin confidence and compute accuracy
    # For Transfer, confidences are in [0, 1]. For MAD/NoComm, confidences are in [1, 10].
    # We will normalize MAD/NoComm confidences by dividing by 10 to put on the same [0,1] scale.
    def compute_calibration_bins(pairs, normalize_integers=False):
        bins = np.linspace(0.0, 1.0, 6) # 0.0-0.2, 0.2-0.4, etc.
        bin_counts = np.zeros(5)
        bin_correct = np.zeros(5)
        
        for conf, is_correct in pairs:
            val = conf
            if normalize_integers and conf > 1.0:
                val = conf / 10.0
            
            # Bound val to [0,1]
            val = max(0.0, min(1.0, val))
            
            # Find bin
            bin_idx = min(int(val * 5), 4)
            bin_counts[bin_idx] += 1
            if is_correct:
                bin_correct[bin_idx] += 1
                
        bin_accs = [bin_correct[i] / bin_counts[i] if bin_counts[i] > 0 else 0.0 for i in range(5)]
        return bins, bin_accs, bin_counts

    tr_bins, tr_accs, tr_counts = compute_calibration_bins(tr_pairs, normalize_integers=False)
    mad_bins, mad_accs, mad_counts = compute_calibration_bins(mad_pairs, normalize_integers=True)
    no_bins, no_accs, no_counts = compute_calibration_bins(no_pairs, normalize_integers=True)

    # ══════════════════════════════════════════════════════════════════════════
    # CLAIM 3: Algebraic Resilience (Example 23)
    # ══════════════════════════════════════════════════════════════════════════
    # Search for insect problem: "Together Lily, David, and Bodhi collected 43 insects"
    
    def find_insect_problem(records, label):
        found = None
        for r in records:
            if "collected 43 insects" in r.get('question', ''):
                found = r
                break
        if found:
            print(f"\nExample 23 ({label}):")
            print(f"  Question: {found['question']}")
            print(f"  Gold Answer: {found['gold_answer']}")
            print(f"  Final Answer: {found['final_answer']}")
            print(f"  Is Correct: {found['is_correct']}")
            print(f"  Agent Final Answers: {found['final_answers']}")
        else:
            print(f"\nExample 23 ({label}): Not found in stream file.")

    find_insect_problem(transfer_records, "Transfer/SparseComm")
    find_insect_problem(mad_records, "MAD")
    find_insect_problem(nocomm_records, "No Comm")

    # ══════════════════════════════════════════════════════════════════════════
    # CLAIM 6: The Truncation Vulnerability (Context Bloat)
    # ══════════════════════════════════════════════════════════════════════════
    # We check how many times the reasoning block ends abruptly.
    # Typically, a complete reasoning block finishes with "Confidence: X" or a numeric answer.
    # An abrupt truncation is identified if it ends in:
    # 1. Math symbols/equations that are open (e.g. Ending in `$=` or `\(` or `\[` or `+` or `/`)
    # 2. Or doesn't have "Confidence:" or "Answer:" or period, and ends mid-sentence.
    # Let's count occurrences of incomplete math tags or trailing characters.
    
    def count_truncations(records):
        trunc_count = 0
        total_blocks = 0
        for r in records:
            fr = r.get('final_reasoning', {})
            for agent_id, text in fr.items():
                total_blocks += 1
                text = text.strip()
                # Check truncation indicators:
                # - Ends with operator or open bracket
                # - Ends with incomplete latex `\(` or `\[`
                # - Ends mid-word with no trailing punctuation and doesn't contain confidence
                if not text:
                    continue
                
                ends_in_math_operator = text[-1] in ['+', '-', '=', '/', '*', '\\', '(', '[', '{']
                ends_in_open_latex = text.endswith('\\(') or text.endswith('\\[') or text.endswith('$')
                no_confidence_and_no_period = ("confidence" not in text.lower()) and (text[-1] not in ['.', '!', '?', '}', ']', ')', '*'])
                
                if ends_in_math_operator or ends_in_open_latex or no_confidence_and_no_period:
                    trunc_count += 1
                    
        pct_trunc = (trunc_count / total_blocks * 100) if total_blocks > 0 else 0
        return trunc_count, total_blocks, pct_trunc

    tr_tr, tr_tb, tr_pct_tr = count_truncations(transfer_records)
    mad_tr, mad_tb, mad_pct_tr = count_truncations(mad_records)
    no_tr, no_tb, no_pct_tr = count_truncations(nocomm_records)
    
    print("\n--- Truncation / Context Bloat Vulnerability ---")
    print(f"Transfer (Ours) - Truncated agent blocks: {tr_tr} / {tr_tb} ({tr_pct_tr:.2f}%)")
    print(f"MAD (Debate)   - Truncated agent blocks: {mad_tr} / {mad_tb} ({mad_pct_tr:.2f}%)")
    print(f"No Comm        - Truncated agent blocks: {no_tr} / {no_tb} ({no_pct_tr:.2f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # CLAIM 8: Anchoring Cascade vs. Edge-Weight Suppression (Error Isolation)
    # ══════════════════════════════════════════════════════════════════════════
    # Let's check cases where:
    # 1. At Round 0: one agent is correct and another is incorrect (disagreement).
    # 2. In MAD, what percentage of correct agents get pulled into the incorrect answer? (Anchoring bias rate)
    # 3. In SparseComm (Transfer), what are the weights assigned to the incorrect agent's channel?
    #    Does it drop below 0.05?
    
    def analyze_error_propagation_and_suppression():
        # MAD: count how many times a correct agent becomes incorrect in the final round
        mad_prop_count = 0
        mad_total_disagreement = 0
        
        for r in mad_records:
            ic = r.get('initial_correct', {})
            fc = r.get('final_answers', {})
            gold = r.get('gold_answer')
            
            if len(ic) == 3:
                # There is disagreement at round 0
                correct_at_r0 = [k for k, v in ic.items() if v is True]
                wrong_at_r0 = [k for k, v in ic.items() if v is False]
                
                if correct_at_r0 and wrong_at_r0:
                    mad_total_disagreement += len(correct_at_r0)
                    # Check if correct agents became wrong in the end
                    for agent_id in correct_at_r0:
                        ans = fc.get(agent_id)
                        is_correct = False
                        if ans is not None and gold is not None:
                            clean_ans = re.sub(r'[^\d\.\-]', '', str(ans))
                            clean_gold = re.sub(r'[^\d\.\-]', '', str(gold))
                            is_correct = (clean_ans == clean_gold) and (clean_ans != '')
                        if not is_correct:
                            mad_prop_count += 1
                            
        # Transfer: find cases of disagreement at round 0, and check average weight from incorrect agent
        suppressed_count = 0
        total_edges_from_incorrect = 0
        weights_from_incorrect = []
        
        for r in transfer_records:
            ic = r.get('initial_correct', {})
            w_matrices = r.get('weight_matrices', [])
            
            if len(ic) == 3 and len(w_matrices) > 0:
                wrong_at_r0 = [int(k) for k, v in ic.items() if v is False]
                correct_at_r0 = [int(k) for k, v in ic.items() if v is True]
                
                if wrong_at_r0 and correct_at_r0:
                    # Let's inspect the weights in Round 1 (index 0 of w_matrices)
                    # w_matrices[0] is of shape 3x3
                    W_r1 = w_matrices[0]
                    for src in wrong_at_r0:
                        for dst in range(3):
                            if src != dst:
                                w_val = W_r1[src][dst]
                                weights_from_incorrect.append(w_val)
                                total_edges_from_incorrect += 1
                                if w_val < 0.05:
                                    suppressed_count += 1
                                    
        mad_prop_rate = (mad_prop_count / mad_total_disagreement * 100) if mad_total_disagreement > 0 else 0
        suppression_rate = (suppressed_count / total_edges_from_incorrect * 100) if total_edges_from_incorrect > 0 else 0
        avg_suppressed_w = np.mean(weights_from_incorrect) if weights_from_incorrect else 0
        
        print("\n--- Error Propagation and Edge Suppression (Error Isolation) ---")
        print(f"MAD - Correct agents pulled into incorrect answer: {mad_prop_count} / {mad_total_disagreement} ({mad_prop_rate:.2f}%)")
        print(f"Transfer (Ours) - Average weight assigned to incorrect agents: {avg_suppressed_w:.6f}")
        print(f"Transfer (Ours) - Fraction of incorrect edges suppressed below 0.05: {suppressed_count} / {total_edges_from_incorrect} ({suppression_rate:.2f}%)")

        return mad_prop_rate, suppression_rate, avg_suppressed_w, weights_from_incorrect

    mad_prop, tr_supp, tr_avg_w, weights_from_incorrect = analyze_error_propagation_and_suppression()

    # ══════════════════════════════════════════════════════════════════════════
    # CLAIM 9: Syntactic Filler Erosion
    # ══════════════════════════════════════════════════════════════════════════
    # We compare character lengths of final reasoning in Transfer vs MAD vs NoComm
    # to show that Transfer/SparseComm has stripped away conversational fluff.
    
    def compute_length_stats(records):
        lengths = []
        for r in records:
            fr = r.get('final_reasoning', {})
            for text in fr.values():
                lengths.append(len(text))
        return np.mean(lengths) if lengths else 0, np.std(lengths) if lengths else 0

    tr_len_mean, tr_len_std = compute_length_stats(transfer_records)
    mad_len_mean, mad_len_std = compute_length_stats(mad_records)
    no_len_mean, no_len_std = compute_length_stats(nocomm_records)
    
    print("\n--- Syntactic Filler Erosion / Reasoning Trace Lengths ---")
    print(f"Transfer (Ours) - Avg reasoning block length (chars): {tr_len_mean:.1f} ± {tr_len_std:.1f}")
    print(f"MAD (Debate)   - Avg reasoning block length (chars): {mad_len_mean:.1f} ± {mad_len_std:.1f}")
    print(f"No Comm        - Avg reasoning block length (chars): {no_len_mean:.1f} ± {no_len_std:.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    # CLAIM 10: SPRT Halt Matrix (Dynamic Depth Optimization)
    # ══════════════════════════════════════════════════════════════════════════
    # Let's count halt rounds for correct vs incorrect, and the percentage of early exits.
    
    def analyze_sprt_halting(records):
        halt_rounds = []
        correct_halts = []
        incorrect_halts = []
        
        for r in records:
            h = r.get('halt_round', 0)
            halt_rounds.append(h)
            if r.get('is_correct', False):
                correct_halts.append(h)
            else:
                incorrect_halts.append(h)
                
        exits_at_r0 = sum(1 for h in halt_rounds if h == 0)
        exits_at_r1 = sum(1 for h in halt_rounds if h == 1)
        exits_at_r2 = sum(1 for h in halt_rounds if h == 2)
        exits_at_r3 = sum(1 for h in halt_rounds if h >= 3)
        total = len(halt_rounds)
        
        print("\n--- SPRT Early Halting / Communication Depth ---")
        print(f"Average communication rounds: {np.mean(halt_rounds):.3f} (Correct samples: {np.mean(correct_halts):.3f}, Incorrect samples: {np.mean(incorrect_halts):.3f})")
        print(f"Exits at Round 0: {exits_at_r0} / {total} ({exits_at_r0/total*100:.1f}%)")
        print(f"Exits at Round 1: {exits_at_r1} / {total} ({exits_at_r1/total*100:.1f}%)")
        print(f"Exits at Round 2: {exits_at_r2} / {total} ({exits_at_r2/total*100:.1f}%)")
        print(f"Exits at Round 3 (Max): {exits_at_r3} / {total} ({exits_at_r3/total*100:.1f}%)")

        return halt_rounds, correct_halts, incorrect_halts

    tr_halts, tr_corr_halts, tr_incorr_halts = analyze_sprt_halting(transfer_records)

    # ══════════════════════════════════════════════════════════════════════════
    # PLOTTING SPECIFIC DIAGRAMS FOR PAPER CLAIMS
    # ══════════════════════════════════════════════════════════════════════════
    
    # Plot 1: Lexical Convergence (Homogenization)
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['No Comm', 'Transfer (Ours)', 'MAD (Debate)'], 
                  [no_avg_jac, tr_avg_jac, mad_avg_jac],
                  color=[C['NoComm'], C['Transfer'], C['FullComm']], alpha=0.85,
                  edgecolor='#333333', linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Average Jaccard Text Similarity')
    ax.set_title('Lexical Homogenization (Groupthink) across Swarm')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS, 'claim_lexical_homogenization.png'), dpi=150)
    plt.close()

    # Plot 2: Confidence Calibration
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0.1, 0.3, 0.5, 0.7, 0.9], tr_accs, 'o-', color=C['Transfer'], label='Transfer (Ours)', lw=2)
    ax.plot([0.1, 0.3, 0.5, 0.7, 0.9], mad_accs, 's--', color=C['FullComm'], label='MAD (Debate - Norm)', lw=1.5)
    ax.plot([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9], 'k:', label='Perfect Calibration', alpha=0.5)
    ax.set_xlabel('Self-Reported Confidence')
    ax.set_ylabel('Empirical Accuracy')
    ax.set_title('Confidence Calibration Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS, 'claim_confidence_calibration.png'), dpi=150)
    plt.close()

    # Plot 3: Truncation Vulnerability
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['No Comm', 'MAD (Debate)', 'Transfer (Ours)'], 
                  [no_pct_tr, mad_pct_tr, tr_pct_tr],
                  color=[C['NoComm'], C['FullComm'], C['Transfer']], alpha=0.85,
                  edgecolor='#333333', linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.2f}%', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Truncation Rate (%)')
    ax.set_title('Truncation Rate (Mid-Sentence Collapse)')
    ax.set_ylim(0, max(no_pct_tr, mad_pct_tr, tr_pct_tr) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS, 'claim_truncation_vulnerability.png'), dpi=150)
    plt.close()

    # Plot 4: Edge Suppression (Error Isolation)
    fig, ax = plt.subplots(figsize=(6, 5))
    # Plot weights from incorrect agents as a histogram or comparison
    # Histogram of edge weights from incorrect agents
    weights_from_incorrect = [w for w in weights_from_incorrect if w is not None]
    ax.hist(weights_from_incorrect, bins=20, color='#E85A5A', alpha=0.8, edgecolor='#333333', range=(0,1))
    ax.set_xlabel('Communication Edge Weight')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Weights from Incorrect Agents')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS, 'claim_edge_suppression_distribution.png'), dpi=150)
    plt.close()

    # Plot 5: SPRT Halt Matrix (Dynamic Depth Optimization)
    fig, ax = plt.subplots(figsize=(6, 5))
    rounds, counts = np.unique(tr_halts, return_counts=True)
    round_labels = [f'R{r}' for r in rounds]
    ax.pie(counts, labels=round_labels, autopct='%1.1f%%', colors=['#6DC6A0', '#4C9BE8', '#BF7BBF', '#F97B6C'],
           wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    ax.set_title('Distribution of Communication Halt Rounds')
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS, 'claim_halt_rounds_pie.png'), dpi=150)
    plt.close()

    print("Claim Verification Analysis completed and plots saved.")

# Colour palette matching full_analysis.py
C = {
    'Transfer':  '#1F77B4', 
    'FullComm':  '#FF7F0E', 
    'NoComm':    '#2CA02C', 
}

if __name__ == '__main__':
    analyze_claims()
