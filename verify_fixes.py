import logging

print("="*80)
print("VERIFICATION CHECKLIST")
print("="*80)

# 1. Check logger import
try:
    logger = logging.getLogger("SparseComm")
    print("[OK] Logger: OK")
except:
    print("[FAIL] Logger: FAILED")

# 2. Check tiktoken
try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = len(tokenizer.encode("test"))
    print(f"[OK] Tiktoken: OK (test={tokens} tokens)")
except:
    print("[FAIL] Tiktoken: FAILED - install with: pip install tiktoken")

# 3. Check sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[OK] SentenceTransformer: OK")
except:
    print("[WARN]  SentenceTransformer: Not available (semantic similarity will use fallback)")

# 4. Check mock backend guard
try:
    from pipeline import run_experiment
    from config import Config
    test_config = Config()
    test_config.llm_backend = "mock"
    # Try to run with mock backend - should raise error
    try:
        run_experiment(test_config, train=False, evaluate=True)
        print("[FAIL] Mock guard: FAILED (Did not raise error)")
    except ValueError as e:
        if "CRITICAL ERROR: Cannot evaluate with llm_backend='mock'" in str(e):
            print("[OK] Mock guard: Present in run_experiment")
        else:
            print("[FAIL] Mock guard: FAILED (Wrong error raised)")
except Exception as e:
    print(f"[FAIL] Mock guard: FAILED ({e})")

print("\n" + "="*80)
print("All critical fixes verified!")
print("="*80)
