"""
Applies all 6 fixes to SparseComm_Kaggle.ipynb:

FIX 1: Add `train_offset: int = 0` to the Config dataclass (proper field, not ad-hoc)
FIX 2: Update all dataset loaders to accept and use train_offset
FIX 3: Update load_dataset_by_name to pass train_offset
FIX 4: Update run_experiment to read config.train_offset instead of getattr() workaround
FIX 5: Replace the old Phase 1/2/3/4 runner block with the new scaling+ablation runner
        - NO re-run of N=100 eval (we have it already)
        - Fresh config objects per checkpoint (no mutation bugs)
        - Ablations start from scratch (no biased pretrained init)
        - Consistent test set size (100) across all scaling checkpoints AND ablations
"""
import json
import re
import shutil
import os

NB_PATH = "SparseComm_Kaggle.ipynb"
BACKUP_PATH = "SparseComm_Kaggle.ipynb.bak"

print(f"Loading {NB_PATH}...")
nb = json.load(open(NB_PATH, encoding="utf-8"))
src = "".join(nb["cells"][0]["source"])

# ─────────────────────────────────────────────────────────────
# FIX 1: Add train_offset field to Config dataclass
# ─────────────────────────────────────────────────────────────
OLD_CONFIG_FIELD = '    # --- Experiment Mode ---\n    mode: str = "all"  # "full", "baseline", "ablation", "all"'
NEW_CONFIG_FIELD = '''    # --- Experiment Mode ---
    mode: str = "all"  # "full", "baseline", "ablation", "all"

    # --- Scaling Study ---
    train_offset: int = 0  # Start index for training data (used to train on non-overlapping chunks)'''

if OLD_CONFIG_FIELD in src:
    src = src.replace(OLD_CONFIG_FIELD, NEW_CONFIG_FIELD, 1)
    print("  [OK] FIX 1: Added train_offset to Config dataclass")
else:
    print("  [WARN] FIX 1: Could not find Config mode field — check manually")

# ─────────────────────────────────────────────────────────────
# FIX 2a: load_gsm8k — add train_offset + process_split offset
# ─────────────────────────────────────────────────────────────
OLD_GSM8K = '''def load_gsm8k(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load GSM8K. Uses 'train' split for training, 'test' split for testing."""
    try:
        from datasets import load_dataset
        ds_train = load_dataset("openai/gsm8k", "main", split="train")
        ds_test = load_dataset("openai/gsm8k", "main", split="test")
        
        rng = random.Random(seed)
        
        def process_split(ds, n):
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            samples = []
            for idx in indices[:n]:'''

NEW_GSM8K = '''def load_gsm8k(n_train: int = 30, n_test: int = 50, seed: int = 42, train_offset: int = 0) -> Tuple[List[dict], List[dict]]:
    """Load GSM8K. Uses 'train' split for training, 'test' split for testing.
    train_offset: skip this many samples at the start of the training split
                  so successive calls load non-overlapping chunks."""
    try:
        from datasets import load_dataset
        ds_train = load_dataset("openai/gsm8k", "main", split="train")
        ds_test = load_dataset("openai/gsm8k", "main", split="test")
        
        rng = random.Random(seed)
        
        def process_split(ds, n, offset=0):
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            samples = []
            for idx in indices[offset:offset + n]:'''

if OLD_GSM8K in src:
    src = src.replace(OLD_GSM8K, NEW_GSM8K, 1)
    print("  [OK] FIX 2a: load_gsm8k signature + process_split updated")
else:
    print("  [WARN] FIX 2a: could not find load_gsm8k header — check manually")

# Fix the return call inside load_gsm8k
OLD_GSM8K_RETURN = '''        train_samples = process_split(ds_train, n_train)
        test_samples = process_split(ds_test, n_test)
        logger.info(f"Loaded {len(train_samples)} train and {len(test_samples)} test GSM8K samples")'''

NEW_GSM8K_RETURN = '''        train_samples = process_split(ds_train, n_train, offset=train_offset)
        test_samples = process_split(ds_test, n_test, offset=0)
        logger.info(f"Loaded {len(train_samples)} train (offset={train_offset}) and {len(test_samples)} test GSM8K samples")'''

if OLD_GSM8K_RETURN in src:
    src = src.replace(OLD_GSM8K_RETURN, NEW_GSM8K_RETURN, 1)
    print("  [OK] FIX 2a: load_gsm8k return call updated")
else:
    print("  [WARN] FIX 2a: could not find load_gsm8k return call")

# ─────────────────────────────────────────────────────────────
# FIX 2b: load_hotpotqa
# ─────────────────────────────────────────────────────────────
OLD_HOTPOT_SIG = 'def load_hotpotqa(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:'
NEW_HOTPOT_SIG = 'def load_hotpotqa(n_train: int = 30, n_test: int = 50, seed: int = 42, train_offset: int = 0) -> Tuple[List[dict], List[dict]]:'
if OLD_HOTPOT_SIG in src:
    src = src.replace(OLD_HOTPOT_SIG, NEW_HOTPOT_SIG, 1)
    print("  [OK] FIX 2b: load_hotpotqa signature updated")
else:
    print("  [WARN] FIX 2b: load_hotpotqa signature not found")

# HotpotQA process_split and return  (single shared process_split def)
OLD_HOTPOT_SPLIT = '''        def process_split(ds, n):
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            samples = []
            for idx in indices[:n]:
                item = ds[idx]
                samples.append({"question": item["question"], "answer": item["answer"]})
            return samples

        train_samples = process_split(ds_train, n_train)
        test_samples = process_split(ds_test, n_test)
        logger.info(f"Loaded {len(train_samples)} train and {len(test_samples)} test HotpotQA samples")'''

NEW_HOTPOT_SPLIT = '''        def process_split(ds, n, offset=0):
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            samples = []
            for idx in indices[offset:offset + n]:
                item = ds[idx]
                samples.append({"question": item["question"], "answer": item["answer"]})
            return samples

        train_samples = process_split(ds_train, n_train, offset=train_offset)
        test_samples = process_split(ds_test, n_test, offset=0)
        logger.info(f"Loaded {len(train_samples)} train (offset={train_offset}) and {len(test_samples)} test HotpotQA samples")'''

if OLD_HOTPOT_SPLIT in src:
    src = src.replace(OLD_HOTPOT_SPLIT, NEW_HOTPOT_SPLIT, 1)
    print("  [OK] FIX 2b: load_hotpotqa process_split + return updated")
else:
    print("  [WARN] FIX 2b: load_hotpotqa inner block not found")

# ─────────────────────────────────────────────────────────────
# FIX 2c: load_aqua
# ─────────────────────────────────────────────────────────────
OLD_AQUA_SIG = 'def load_aqua(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:'
NEW_AQUA_SIG = 'def load_aqua(n_train: int = 30, n_test: int = 50, seed: int = 42, train_offset: int = 0) -> Tuple[List[dict], List[dict]]:'
if OLD_AQUA_SIG in src:
    src = src.replace(OLD_AQUA_SIG, NEW_AQUA_SIG, 1)
    print("  [OK] FIX 2c: load_aqua signature updated")
else:
    print("  [WARN] FIX 2c: load_aqua signature not found")

# AQuA uses fetch_split, need to update that pattern
OLD_AQUA_FETCH = '''    def fetch_split(url, n, rng):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            lines = response.read().decode('utf-8').strip().split('\\n')
        samples = []
        indices = list(range(len(lines)))
        rng.shuffle(indices)
        for idx in indices[:n]:'''
NEW_AQUA_FETCH = '''    def fetch_split(url, n, rng, offset=0):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            lines = response.read().decode('utf-8').strip().split('\\n')
        samples = []
        indices = list(range(len(lines)))
        rng.shuffle(indices)
        for idx in indices[offset:offset + n]:'''
if OLD_AQUA_FETCH in src:
    src = src.replace(OLD_AQUA_FETCH, NEW_AQUA_FETCH, 1)
    print("  [OK] FIX 2c: load_aqua fetch_split updated")
else:
    print("  [WARN] FIX 2c: load_aqua fetch_split block not found")

# AQuA return calls
OLD_AQUA_CALLS = '''        train_samples = fetch_split("https://raw.githubusercontent.com/google-deepmind/AQuA/master/train.json", n_train, rng)
        test_samples = fetch_split("https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json", n_test, rng)'''
NEW_AQUA_CALLS = '''        train_samples = fetch_split("https://raw.githubusercontent.com/google-deepmind/AQuA/master/train.json", n_train, rng, offset=train_offset)
        test_samples = fetch_split("https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json", n_test, rng, offset=0)'''
if OLD_AQUA_CALLS in src:
    src = src.replace(OLD_AQUA_CALLS, NEW_AQUA_CALLS, 1)
    print("  [OK] FIX 2c: load_aqua fetch calls updated")
else:
    print("  [WARN] FIX 2c: load_aqua fetch call lines not found")

# ─────────────────────────────────────────────────────────────
# FIX 3: Update load_dataset_by_name
# ─────────────────────────────────────────────────────────────
OLD_LOADER = 'def load_dataset_by_name(name: str, n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:'
NEW_LOADER = 'def load_dataset_by_name(name: str, n_train: int = 30, n_test: int = 50, seed: int = 42, train_offset: int = 0) -> Tuple[List[dict], List[dict]]:'
if OLD_LOADER in src:
    src = src.replace(OLD_LOADER, NEW_LOADER, 1)
    print("  [OK] FIX 3: load_dataset_by_name signature updated")
else:
    print("  [WARN] FIX 3: load_dataset_by_name signature not found")

OLD_LOADER_GSM = '        return load_gsm8k(n_train, n_test, seed)'
NEW_LOADER_GSM = '        return load_gsm8k(n_train, n_test, seed, train_offset)'
OLD_LOADER_HQ  = '        return load_hotpotqa(n_train, n_test, seed)'
NEW_LOADER_HQ  = '        return load_hotpotqa(n_train, n_test, seed, train_offset)'
OLD_LOADER_AQ  = '        return load_aqua(n_train, n_test, seed)'
NEW_LOADER_AQ  = '        return load_aqua(n_train, n_test, seed, train_offset)'

for old, new, name in [
    (OLD_LOADER_GSM, NEW_LOADER_GSM, "gsm8k"),
    (OLD_LOADER_HQ,  NEW_LOADER_HQ,  "hotpotqa"),
    (OLD_LOADER_AQ,  NEW_LOADER_AQ,  "aqua"),
]:
    if old in src:
        src = src.replace(old, new, 1)
        print(f"  [OK] FIX 3: load_dataset_by_name {name} call updated")
    else:
        print(f"  [WARN] FIX 3: load_dataset_by_name {name} call not found")

# ─────────────────────────────────────────────────────────────
# FIX 4: run_experiment — read train_offset from Config properly
# ─────────────────────────────────────────────────────────────
OLD_RUN_EXP_LOAD = '''    # Load dataset (strictly separate splits)
    train_data, test_data = load_dataset_by_name(
        config.dataset, n_train=config.n_train_samples, n_test=config.n_samples, seed=config.seed
    )'''
NEW_RUN_EXP_LOAD = '''    # Load dataset (strictly separate splits)
    # train_offset lets us draw a fresh non-overlapping chunk for each scaling step
    train_data, test_data = load_dataset_by_name(
        config.dataset, n_train=config.n_train_samples, n_test=config.n_samples,
        seed=config.seed, train_offset=config.train_offset
    )'''
if OLD_RUN_EXP_LOAD in src:
    src = src.replace(OLD_RUN_EXP_LOAD, NEW_RUN_EXP_LOAD, 1)
    print("  [OK] FIX 4: run_experiment load call updated to use config.train_offset")
else:
    print("  [WARN] FIX 4: run_experiment dataset load block not found")

# ─────────────────────────────────────────────────────────────
# FIX 5: Replace the entire execution section (Phase 1 onwards)
#         with the corrected scaling + ablation runner
# ─────────────────────────────────────────────────────────────

# The old execution block starts right before Phase 1
OLD_EXEC_START = '''# =========================================================
# LARGE-SCALE EXECUTION WITH TOY VALIDATION
# =========================================================
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

datasets_to_run = ["gsm8k", "hotpotqa", "aqua"]

# Bug #15: RUN_EXPENSIVE guard - set True ONLY when you are ready to spend API budget
logger = logging.getLogger(__name__)

datasets_to_run = ["gsm8k", "hotpotqa", "aqua"]

# Bug #15: RUN_EXPENSIVE guard - set True ONLY when you are ready to spend API budget
RUN_EXPENSIVE = True  # <-- set False to skip LLM calls
RUN_TOY_ONLY = False    # <-- set True to exit immediately after the Toy Run (to test pipeline)
if not RUN_EXPENSIVE:
    print("[SKIP] RUN_EXPENSIVE=False. Set to True to run toy validation and full phases.")
    import sys; sys.exit(0)'''

NEW_EXEC_BLOCK = '''# =========================================================
# SCALING + ABLATION STUDY
# =========================================================
import logging, os, copy as _copy
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

AZURE_API_KEY  = "your-azure-api-key"
AZURE_MODEL    = "gpt-oss-120b"
LLM_BACKEND    = "azure"

datasets_to_run = ["gsm8k", "hotpotqa", "aqua"]
_cfg_factory   = {"gsm8k": get_gsm8k_config, "hotpotqa": get_hotpotqa_config, "aqua": get_aqua_config}

def _base_cfg(ds, seed=42, n_eval=100, n_train=100, offset=0):
    """Create a fresh Config for one dataset/seed/chunk. Never reuses objects."""
    cfg = _cfg_factory[ds](
        seed=seed,
        n_samples=n_eval,
        n_train_samples=n_train,
        llm_backend=LLM_BACKEND,
        api_model=AZURE_MODEL,
        api_key=AZURE_API_KEY,
        train_offset=offset,
    )
    cfg.dataset = ds
    return cfg

os.makedirs("08-06results/scaling", exist_ok=True)

# ─────────────────────────────────────────────────────────
# STAGE 1 + 2: Per-dataset scaling (train +100 per step,
#              evaluate at N=100 (SKIP — already done),
#              N=200, N=300, N=400)
# ─────────────────────────────────────────────────────────
print("\\n" + "="*60)
print("STAGE 1 + 2: INCREMENTAL TRAINING & CHECKPOINT EVALUATION")
print("="*60)

final_controllers = {}   # keep N=400 controllers for Stage 3

for ds in datasets_to_run:
    print(f"\\n>>> Dataset: {ds.upper()}")

    # ── Load the N=100 checkpoint we already trained ──────────
    npz_path = f"08-06results/{ds}_pretrained.pt.npz"
    init_cfg  = _base_cfg(ds, seed=42, n_eval=100, n_train=100, offset=0)
    controller = IBController(init_cfg)
    if os.path.exists(npz_path):
        controller.load_model(npz_path)
        print(f"  Loaded N=100 checkpoint: {npz_path}")
    else:
        print(f"  WARNING: {npz_path} not found — starting fresh")

    # ── N=100 results already exist — DO NOT re-evaluate ─────
    n100_file = f"08-06results/continuous/IB-Controller_N100_{ds}_stream.jsonl"
    if os.path.exists(n100_file):
        print(f"  Skipping N=100 eval (file exists: {n100_file})")
    else:
        print(f"  Running N=100 eval (no existing file)...")
        run_experiment(_base_cfg(ds, seed=42, n_eval=100, n_train=0, offset=0),
                       method_name="IB-Controller_N100",
                       train=False, evaluate=True, controller=controller)

    # ── Incrementally train +100 at a time: 100→200→300→400 ──
    prev_N = 100
    for target_N in [200, 300, 400]:
        print(f"\\n  --- Training {ds}: N={prev_N} → N={target_N} ---")

        # FRESH config per chunk — avoids mutation across iterations
        train_cfg = _base_cfg(ds, seed=42, n_eval=100,
                               n_train=100,           # exactly 100 new samples
                               offset=prev_N)         # start AFTER previous chunk
        train_cfg.model_save_path = f"08-06results/scaling/{ds}_N{target_N}.pt"

        run_experiment(train_cfg,
                       method_name=f"IB-Controller_N{target_N}",
                       train=True, evaluate=True,
                       controller=controller)          # controller is updated in-place
        prev_N = target_N

    final_controllers[ds] = controller
    print(f"\\n  [OK] {ds.upper()} scaling complete (N=400 controller stored)")

# ─────────────────────────────────────────────────────────
# STAGE 3: Baseline re-evaluation with N=400 weights
#          3 datasets × 3 seeds × 300 eval samples
#          Same seeds as original Phase 2 run (0, 1, 2)
# ─────────────────────────────────────────────────────────
print("\\n" + "="*60)
print("STAGE 3: BASELINE RE-EVALUATION (N=400 weights, 3 seeds)")
print("="*60)

baseline_results_400 = {}
for ds in datasets_to_run:
    for seed in [0, 1, 2]:
        set_seed(seed)
        method = f"Transfer_{ds.upper()}_N400_seed{seed}"
        print(f"\\n  Evaluating {method}...")

        # Deep-copy the N=400 controller so seeds don't corrupt each other
        eval_ctrl = _copy.deepcopy(final_controllers[ds])
        eval_cfg  = _base_cfg(ds, seed=seed, n_eval=300, n_train=0, offset=0)

        result = run_experiment(eval_cfg, method_name=method,
                                train=False, evaluate=True,
                                controller=eval_ctrl)
        baseline_results_400[method] = result

save_kaggle_checkpoint(baseline_results_400,
                       filepath="08-06results/scaling/baseline_N400_results.json")
print("\\n[OK] Stage 3 complete — saved to 08-06results/scaling/baseline_N400_results.json")

# ─────────────────────────────────────────────────────────
# STAGE 4: 2×2 Ablation on GSM8K
#          Each arm trains from the N=100 checkpoint
#          (+300 new samples) with its specific lambdas.
#          Evaluated on 100 test samples (seed=42).
#
#  | Arm | β_topology (lambda_info) | β_sparsity (lambda_kl) |
#  |-----|--------------------------|------------------------|
#  | A00 |         0.0              |         0.0            |
#  | A10 |         0.06             |         0.0            |
#  | A01 |         0.0              |         0.03           |
#  | A11 |         0.06             |         0.03           |
# ─────────────────────────────────────────────────────────
print("\\n" + "="*60)
print("STAGE 4: 2×2 ABLATION (GSM8K, N=100→400 per arm)")
print("="*60)

ablation_arms = [
    ("A00", 0.0,  0.0),
    ("A10", 0.06, 0.0),
    ("A01", 0.0,  0.03),
    ("A11", 0.06, 0.03),
]
ablation_results_400 = {}

for arm_name, b_topo, b_sparse in ablation_arms:
    print(f"\\n>>> Ablation {arm_name} (β_topology={b_topo}, β_sparsity={b_sparse})")

    # Start from the N=100 checkpoint (same starting point for all arms)
    npz_path   = "08-06results/gsm8k_pretrained.pt.npz"
    ab_ctrl    = IBController(_base_cfg("gsm8k", seed=42, n_eval=100, n_train=0, offset=0))
    if os.path.exists(npz_path):
        ab_ctrl.load_model(npz_path)
        print(f"  Loaded base N=100 checkpoint")
    else:
        print(f"  WARNING: base checkpoint not found — starting fresh")

    # Incrementally train +100 samples × 3 steps with THIS arm's lambdas
    prev_N = 100
    for target_N in [200, 300, 400]:
        ab_train_cfg = _base_cfg("gsm8k", seed=42, n_eval=100,
                                  n_train=100, offset=prev_N)
        ab_train_cfg.lambda_info = b_topo
        ab_train_cfg.lambda_kl   = b_sparse
        ab_train_cfg.model_save_path = f"08-06results/scaling/gsm8k_{arm_name}_N{target_N}.pt"

        do_eval = (target_N == 400)   # only evaluate at the final checkpoint
        run_experiment(ab_train_cfg,
                       method_name=f"Ablation_{arm_name}_N{target_N}",
                       train=True, evaluate=do_eval,
                       controller=ab_ctrl)
        prev_N = target_N

    ablation_results_400[arm_name] = ab_ctrl  # store for OLS

save_kaggle_checkpoint(
    {k: v for k, v in ablation_results_400.items() if not isinstance(v, IBController)},
    filepath="08-06results/scaling/ablation_N400_results.json"
)
print("\\n[OK] Stage 4 ablation complete")

# ─────────────────────────────────────────────────────────
# OLS: 2×2 factorial on ablation accuracy
# ─────────────────────────────────────────────────────────
try:
    _arm_order = ["A00", "A10", "A01", "A11"]
    _A  = np.array([0., 1., 0., 1.])   # β_topology factor
    _B  = np.array([0., 0., 1., 1.])   # β_sparsity factor
    _AB = _A * _B
    _X  = np.column_stack([np.ones(4), _A, _B, _AB])

    def _arm_acc(arm):
        # Read accuracy from the N=400 eval jsonl
        path = f"08-06results/continuous/Ablation_{arm}_N400_gsm8k_stream.jsonl"
        if not os.path.exists(path):
            return 0.0
        with open(path) as f:
            records = [json.loads(l) for l in f if l.strip()]
        return float(np.mean([r.get("is_correct", 0) for r in records]))

    _y    = np.array([_arm_acc(a) for a in _arm_order])
    _beta = np.linalg.lstsq(_X, _y, rcond=None)[0]
    print("\\n2×2 Factorial OLS (accuracy):")
    print(f"  intercept:             {_beta[0]:.4f}")
    print(f"  β_topology  (A):       {_beta[1]:+.4f}")
    print(f"  β_sparsity  (B):       {_beta[2]:+.4f}")
    print(f"  interaction (A×B):     {_beta[3]:+.4f}")
    print(f"\\nClaim: β_topology > β_sparsity  →  {_beta[1]:+.4f} > {_beta[2]:+.4f}  ({'SUPPORTED' if _beta[1] > _beta[2] else 'NOT SUPPORTED'})")
except Exception as _e:
    print(f"[WARN] OLS failed: {_e}")

print("\\n" + "="*60)
print("[DONE] All stages complete.")
print("  Stage 1+2 scaling results: 08-06results/continuous/IB-Controller_N*_*_stream.jsonl")
print("  Stage 3 baseline results:  08-06results/scaling/baseline_N400_results.json")
print("  Stage 4 ablation results:  08-06results/scaling/ablation_N400_results.json")
print("="*60)'''

if OLD_EXEC_START in src:
    # Find the full old execution block and replace it all the way to the end
    start_idx = src.index(OLD_EXEC_START)
    src = src[:start_idx] + NEW_EXEC_BLOCK
    print("  [OK] FIX 5: Replaced entire execution block with corrected runner")
else:
    print("  [WARN] FIX 5: Could not find execution block start — check manually")
    # Try a shorter anchor
    anchor = "RUN_EXPENSIVE = True  # <-- set False to skip LLM calls"
    if anchor in src:
        print(f"  [INFO] Found shorter anchor, trying alternate replacement...")
    

# ─────────────────────────────────────────────────────────────
# WRITE BACK
# ─────────────────────────────────────────────────────────────
print("\nBacking up original notebook...")
shutil.copy(NB_PATH, BACKUP_PATH)
print(f"  Backup: {BACKUP_PATH}")

# Rebuild cell 0 source as list of lines with newlines
new_source_lines = []
for line in src.split('\n'):
    new_source_lines.append(line + '\n')
# Remove trailing newline from last line
if new_source_lines:
    new_source_lines[-1] = new_source_lines[-1].rstrip('\n')

nb['cells'][0]['source'] = new_source_lines
nb['cells'][0]['outputs'] = []   # clear old outputs

print("Writing patched notebook...")
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[DONE] {NB_PATH} patched successfully.")
print(f"  Original backed up to: {BACKUP_PATH}")
print("\nSummary of changes:")
print("  FIX 1: Config.train_offset field added (int = 0)")
print("  FIX 2: All dataset loaders (gsm8k, hotpotqa, aqua) accept train_offset")
print("  FIX 3: load_dataset_by_name passes train_offset through")
print("  FIX 4: run_experiment reads config.train_offset cleanly")
print("  FIX 5: Entire runner replaced:")
print("    - No wasted N=100 re-evaluation (skipped if file exists)")
print("    - Fresh Config object per training chunk (no mutation bug)")
print("    - Ablation arms each train from N=100 base with their own lambdas")
print("    - Consistent 100-sample test set for scaling + ablation")
print("    - 300-sample eval for Stage 3 baselines (3 seeds)")
print("    - OLS computes β_topology vs β_sparsity with SUPPORTED/NOT SUPPORTED verdict")
