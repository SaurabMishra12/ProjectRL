import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import json
import numpy as np
from collections import Counter
from dataset_loaders import hotpotqa_f1_score, evaluate_answer

w3_records = [json.loads(line) for line in open("results/W3_corrected/raw_results.jsonl")]
w4_records = [json.loads(line) for line in open("results/W4_corrected/raw_results.jsonl")]

print("=" * 80)
print("1. W3: PRECISION OF EARLY EXIT (SPRT AS AN UNCERTAINTY FILTER)")
print("=" * 80)
r1_clean = [r for r in w3_records if r["Clean_Adaptive"]["rounds"] == 1]
r2_clean = [r for r in w3_records if r["Clean_Adaptive"]["rounds"] == 2]
r3_clean = [r for r in w3_records if r["Clean_Adaptive"]["rounds"] == 3]

print(f"Clean Adaptive R1 Stop (N={len(r1_clean)}): EM={np.mean([r['Clean_Adaptive']['is_em'] for r in r1_clean])*100:.1f}%, F1={np.mean([r['Clean_Adaptive']['f1'] for r in r1_clean])*100:.2f}%")
print(f"Clean Adaptive R2 Stop (N={len(r2_clean)}): EM={np.mean([r['Clean_Adaptive']['is_em'] for r in r2_clean])*100:.1f}%, F1={np.mean([r['Clean_Adaptive']['f1'] for r in r2_clean])*100:.2f}%")
print(f"Clean Adaptive R3 Stop (N={len(r3_clean)}): EM={np.mean([r['Clean_Adaptive']['is_em'] for r in r3_clean])*100:.1f}%, F1={np.mean([r['Clean_Adaptive']['f1'] for r in r3_clean])*100:.2f}%")

r1_noisy = [r for r in w3_records if r["Noisy_Adaptive"]["rounds"] == 1]
r2_noisy = [r for r in w3_records if r["Noisy_Adaptive"]["rounds"] == 2]
r3_noisy = [r for r in w3_records if r["Noisy_Adaptive"]["rounds"] == 3]

print(f"Noisy Adaptive R1 Stop (N={len(r1_noisy)}): EM={np.mean([r['Noisy_Adaptive']['is_em'] for r in r1_noisy])*100:.1f}%, F1={np.mean([r['Noisy_Adaptive']['f1'] for r in r1_noisy])*100:.2f}%")
print(f"Noisy Adaptive R2 Stop (N={len(r2_noisy)}): EM={np.mean([r['Noisy_Adaptive']['is_em'] for r in r2_noisy])*100:.1f}%, F1={np.mean([r['Noisy_Adaptive']['f1'] for r in r2_noisy])*100:.2f}%")
print(f"Noisy Adaptive R3 Stop (N={len(r3_noisy)}): EM={np.mean([r['Noisy_Adaptive']['is_em'] for r in r3_noisy])*100:.1f}%, F1={np.mean([r['Noisy_Adaptive']['f1'] for r in r3_noisy])*100:.2f}%")

print("\n" + "=" * 80)
print("2. DELIBERATION DRIFT: DOES FORCED DELIBERATION (ROUND 3) CAUSE DRIFT?")
print("=" * 80)
# Look at questions where Adaptive stopped at Round 1, comparing Round 1 answer vs Fixed Round 3 answer
r1_drift_worse = []
r1_drift_better = []
r1_unchanged = []
for r in r1_clean:
    f1_adapt = r["Clean_Adaptive"]["f1"]
    f1_fixed = r["Clean_Fixed"]["f1"]
    if f1_adapt > f1_fixed:
        r1_drift_worse.append(r)
    elif f1_adapt < f1_fixed:
        r1_drift_better.append(r)
    else:
        r1_unchanged.append(r)

print(f"In Clean Evidence when agents agreed at Round 1 (N={len(r1_clean)}):")
print(f" - Forcing 2 extra rounds degraded answer (Fixed < Adapt): {len(r1_drift_worse)} questions ({len(r1_drift_worse)/len(r1_clean)*100:.1f}%)")
print(f" - Forcing 2 extra rounds improved answer (Fixed > Adapt): {len(r1_drift_better)} questions ({len(r1_drift_better)/len(r1_clean)*100:.1f}%)")
print(f" - Answer remained identical: {len(r1_unchanged)} questions ({len(r1_unchanged)/len(r1_clean)*100:.1f}%)")

print("\n" + "=" * 80)
print("3. W4: TOPOLOGY ANSWER OVERLAP & CONSISTENCY DYNAMICS")
print("=" * 80)
same_all_3 = 0
same_static_learned = 0
same_static_random = 0
same_learned_random = 0

for r in w4_records:
    ans_s = r["Static_Role"]["answer"].strip().lower()
    ans_l = r["Learned_SparseComm"]["answer"].strip().lower()
    ans_r = r["Random_Routing"]["answer"].strip().lower()
    
    if ans_s == ans_l == ans_r:
        same_all_3 += 1
    if ans_s == ans_l:
        same_static_learned += 1
    if ans_s == ans_r:
        same_static_random += 1
    if ans_l == ans_r:
        same_learned_random += 1

print(f"Questions where Static, Learned, AND Random gave identical answer: {same_all_3}/100 ({same_all_3}%)")
print(f"Static vs Learned answer agreement: {same_static_learned}/100 ({same_static_learned}%)")
print(f"Static vs Random answer agreement: {same_static_random}/100 ({same_static_random}%)")
print(f"Learned vs Random answer agreement: {same_learned_random}/100 ({same_learned_random}%)")

print("\n" + "=" * 80)
print("4. W4: COMPLETE BROADCAST BANDWIDTH FRAGMENTATION PENALTY")
print("=" * 80)
# Compare Complete Matched vs 2-edge topologies
cm_f1 = [r["Complete_Broadcast_Matched"]["f1"] for r in w4_records]
cf_f1 = [r["Complete_Broadcast_Full"]["f1"] for r in w4_records]
static_f1 = [r["Static_Role"]["f1"] for r in w4_records]
learned_f1 = [r["Learned_SparseComm"]["f1"] for r in w4_records]

print(f"Complete Matched (6 edges, cap floor(2*L_q/6)): F1 = {np.mean(cm_f1)*100:.2f}%, EM = {np.mean([r['Complete_Broadcast_Matched']['is_em'] for r in w4_records])*100:.1f}%")
print(f"Complete Full (6 edges, unconstrained 50 tok/edge): F1 = {np.mean(cf_f1)*100:.2f}%, EM = {np.mean([r['Complete_Broadcast_Full']['is_em'] for r in w4_records])*100:.1f}%")
print(f"Static Role (2 edges, cap L_q): F1 = {np.mean(static_f1)*100:.2f}%, EM = {np.mean([r['Static_Role']['is_em'] for r in w4_records])*100:.1f}%")

# Edge cap distribution
edge_caps = [max(0, (2 * r["L_q"]) // 6) for r in w4_records]
print(f"Complete Matched Edge Token Cap: mean = {np.mean(edge_caps):.1f} tokens/edge, min = {min(edge_caps)}, max = {max(edge_caps)}")
zero_cap = sum(1 for c in edge_caps if c == 0)
print(f"Questions where Complete Matched edge cap was 0 tokens (total blackout): {zero_cap}/100")

print("\n" + "=" * 80)
print("5. W4: LEARNED EDGE SELECTION PATTERNS (WHAT DID THE CONTROLLER LEARN?)")
print("=" * 80)
learned_edges_list = []
for r in w4_records:
    edges = r["Learned_SparseComm"]["active_edges"][0] # round 1 edges
    edges_tuple = tuple(sorted([tuple(e) for e in edges]))
    learned_edges_list.append(edges_tuple)

edge_counts = Counter(learned_edges_list)
print("Top-selected edge pairs by Learned Controller in Round 1:")
for edges, count in edge_counts.most_common(10):
    print(f"  Edges {edges}: {count}/100 ({count}%)")

print("\n" + "=" * 80)
print("6. THE RETRIEVAL RECALL vs DISTRACTOR NOISE PHENOMENON")
print("=" * 80)
# Inspect why Noisy evidence had higher EM (60%) than Clean top-3 BM25 (52-53%)
from dataset_loaders import load_open_book_hotpotqa
subset = json.load(open("web_eval_subset_100.json"))[:100]

# Check supporting facts recall in top-3 BM25 vs full 10 candidate articles
from web_suite import LocalWikipediaRetriever
top3_gold_recall = []
all10_gold_recall = []

for sample in subset:
    q = sample["question"]
    articles = sample["articles"]
    gold_titles = set(sample.get("supporting_facts", {}).get("title", []))
    if not gold_titles:
        # fallback: find titles that contain answers or were provided
        pass
    retriever = LocalWikipediaRetriever(articles)
    top3 = retriever.retrieve(q, top_k=3)
    top3_titles = set(p["title"] for p in top3)
    all10_titles = set(a["title"] for a in articles)
    
    # check recall of gold titles
    if gold_titles:
        top3_recall = len(gold_titles.intersection(top3_titles)) / len(gold_titles)
        all10_recall = len(gold_titles.intersection(all10_titles)) / len(gold_titles)
        top3_gold_recall.append(top3_recall)
        all10_gold_recall.append(all10_recall)

if top3_gold_recall:
    print(f"Gold Document Recall in Top-3 BM25 ('Clean'): {np.mean(top3_gold_recall)*100:.1f}%")
    print(f"Gold Document Recall in All 10 Articles ('Noisy'): {np.mean(all10_gold_recall)*100:.1f}%")
