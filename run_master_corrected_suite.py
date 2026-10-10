#!/usr/bin/env python3
"""
Master Execution Suite for Corrected WWW 2027 Experiments
=========================================================
Orchestrates:
  1. Completion of W3_corrected (monitors or resumes until 100 samples complete)
  2. Execution of W4_corrected (100 samples under strict volume matching)
  3. Execution of Independent Analysis Engine (recomputing all metrics from raw JSONL)
  4. Execution of Automated Integrity Validator (verifying all 6 empirical gates)
"""

import os
import sys
import time
import json
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(BASE_DIR, ".venv_linux", "bin", "python")

def run_cmd(cmd, desc):
    print("\n" + "=" * 80)
    print(f"STAGE: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"FAILED: {desc} exited with code {res.returncode}")
        sys.exit(res.returncode)
    print(f"COMPLETED: {desc} in {elapsed:.1f}s")

def main():
    print("=" * 80)
    print("WWW 2027 MASTER CORRECTED EXPERIMENTS ORCHESTRATOR")
    print("=" * 80)

    # 1. Check / Wait for W3_corrected
    w3_manifest = os.path.join(BASE_DIR, "results", "W3_corrected", "manifest.json")
    w3_done = False
    if os.path.exists(w3_manifest):
        try:
            with open(w3_manifest) as f:
                if json.load(f).get("status") == "completed":
                    w3_done = True
        except Exception:
            pass

    if not w3_done:
        print("Ensuring W3-Corrected completes all 100 samples...")
        run_cmd([PYTHON, "run_w3_corrected.py", "--n_samples", "100", "--seed", "42"], "Run W3-Corrected")
    else:
        print("W3-Corrected is already completed.")

    # 2. Run W4_corrected
    w4_manifest = os.path.join(BASE_DIR, "results", "W4_corrected", "manifest.json")
    w4_done = False
    if os.path.exists(w4_manifest):
        try:
            with open(w4_manifest) as f:
                if json.load(f).get("status") == "completed":
                    w4_done = True
        except Exception:
            pass

    if not w4_done:
        run_cmd([PYTHON, "run_w4_corrected.py", "--n_samples", "100", "--seed", "42"], "Run W4-Corrected")
    else:
        print("W4-Corrected is already completed.")

    # 3. Run Independent Recomputation & Verification
    run_cmd([PYTHON, "run_independent_analysis.py"], "Run Independent Recomputation")

    # 4. Run Automated Integrity Check
    run_cmd([PYTHON, "verify_corrected_integrity.py"], "Run Automated Integrity Check")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTAL PHASES COMPLETED AND VERIFIED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
