import py_compile, nbformat

with open('temp_check.py', 'r', encoding='utf-8') as f:
    src = f.read()

try:
    py_compile.compile('temp_check.py', doraise=True)
    print('[OK] Syntax check passed')
except py_compile.PyCompileError as e:
    print('[FAIL]', e)
    exit(1)

checks = [
    ('V3-1 results table dict safe', 'hasattr(stats'),
    ('V3-2 Nvidia max_tokens forwarded', 'max_tokens=getattr(config'),
    ('V3-3 encode_state normalize', '_normalize_for_vote(a, _dataset_type)'),
    ('V3-3b get_weights _dataset_type', 'self.semantic_encoder._dataset_type = dataset_type'),
    ('V3-3c early exit normalize', '_normalize_for_vote(r.answer, config.dataset) for r in round0_responses'),
    ('V3-4 answer_diversity', 'answer_diversity'),
    ('V3-4 length_variance', 'length_variance'),
    ('V3-5a full_comm calibrator param', 'run_full_communication(config: Config, calibrator'),
    ('V3-5b full_comm vote calibrator', 'calibrator=calibrator)  # V3-5'),
    ('V3-5c no_comm calibrator param', 'run_no_communication(config: Config, calibrator'),
    ('V3-6 numeric median integrated', 'len(_num_entries) == len(responses)'),
    ('V3-7 save calibrator', 'calibrator_a=np.array(self.calibrator.a)'),
    ('V3-7 load calibrator', "calibrator_a' in data"),
    ('V3-8 dead comment removed', 'EMA prior is updated once per epoch'),
    ('V3-9 debate unanimous exit', 'V3-9: unanimous Round-0'),
    ('V3-10 unanimous fast-path', 'V3-10: unanimous fast-path'),
    ('full_comm log_continuous_result present', 'log_continuous_result("Multi-Agent Debate"'),
]
all_ok = True
for label, pat in checks:
    found = pat in src
    print('[OK]' if found else '[FAIL]', label)
    if not found:
        all_ok = False

print()
print('[ALL GOOD]' if all_ok else '[SOME FAILED - check above]')

nb = nbformat.read('SparseComm_Kaggle.ipynb', as_version=4)
nb.cells[2].source = src
nbformat.write(nb, 'SparseComm_Kaggle.ipynb')
print('[OK] Notebook updated')
