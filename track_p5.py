#!/usr/bin/env python3
"""
Live Progress Tracker for Master Experiment P5
==============================================
Run in any terminal:
  python track_p5.py
Or live auto-refreshing:
  watch -n 2 python track_p5.py
"""

import os
import json
import glob
from collections import Counter

STREAM_DIR = "results/p5_communication_control/stream"

CELLS = [
    ("Cell_A_Static_Fixed", "Cell A: Static S1 + Fixed Depth"),
    ("Cell_B_Static_Adaptive", "Cell B: Static S1 + Adaptive Stopping"),
    ("Cell_C_Learned_Fixed", "Cell C: Learned Top-2 + Fixed Depth"),
    ("Cell_D_Learned_Adaptive", "Cell D: Learned Top-2 + Adaptive Stopping"),
]

print("=" * 80)
print("  MASTER EXPERIMENT P5: LIVE PROGRESS MONITOR")
print("=" * 80)

total_completed_all = 0
for cell_key, cell_name in CELLS:
    stream_file = os.path.join(STREAM_DIR, f"{cell_key}_stream.jsonl")
    if not os.path.exists(stream_file):
        print(f"[-] {cell_name:<42} | Waiting to start (0/100)")
        continue

    records = []
    with open(stream_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line.strip()))
                except Exception:
                    pass

    n = len(records)
    total_completed_all += n
    if n == 0:
        print(f"[-] {cell_name:<42} | Initializing (0/100)")
        continue

    correct = sum(1 for r in records if r.get("is_correct", False))
    acc = (correct / n) * 100.0
    tokens = [r.get("total_tokens_transmitted", 0) for r in records]
    mean_tok = sum(tokens) / n if n > 0 else 0
    rounds = [r.get("halt_round", 3) for r in records]
    mean_rnd = sum(rounds) / n if n > 0 else 0

    status = "[DONE]" if n >= 100 else f"[{n:3d}%]"
    print(f"{status} {cell_name:<42} | N: {n:3d}/100 | Acc: {acc:5.1f}% | Avg Tokens: {mean_tok:4.1f} | Avg Rounds: {mean_rnd:3.2f}")

print("-" * 80)
print(f"Total question-cell records processed: {total_completed_all}/400")
print("=" * 80)
