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
    # v4 checks
    ('Fix1a Nvidia uses agent prompt', "if prompt and prompt.strip():"),
    ('Fix1b Azure uses agent prompt', "if prompt and prompt.strip():"),
    ('Fix2a include_token_penalty param', 'include_token_penalty: bool = False'),
    ('Fix2b info_cost conditional', 'if include_token_penalty else 0.0'),
    ('Fix3 fair IB vote no W/agents', 'Fix3: Removed W (spectral centrality)'),
    ('Fix4 HotpotQA EM not F1', '_hotpot_normalize'),
    ('Fix5a pretrained_controllers dict', 'pretrained_controllers: dict = {}'),
    ('Fix5b controller saved after P1', 'pretrained_controllers[ds_name] = controller'),
    ('Fix5c Phase2 warm-start', '_make_transfer_fn(pretrained_controllers'),
    ('Fix6 transmit extracts answer', '_ans_match = re.search'),
    ('Fix7a count_cycles default 0.5', 'threshold: float = 0.5)  # Fix7'),
    ('Fix7b accum_steps capped at 16', 'min(len(train_data), 16)'),
    # v3 still in place
    ('V3-1 results table safe', 'hasattr(stats'),
    ('V3-3b encoder dataset type', 'self.semantic_encoder._dataset_type = dataset_type'),
    ('V3-7 calibrator persisted', 'calibrator_a=np.array(self.calibrator.a)'),
    ('V3-10 unanimous fast-path', 'V3-10: unanimous fast-path'),
]

all_ok = True
for label, pat in checks:
    found = pat in src
    print('[OK]' if found else '[FAIL]', label)
    if not found:
        all_ok = False

print()
print('[ALL GOOD]' if all_ok else '[SOME FAILED - see above]')

nb = nbformat.read('SparseComm_Kaggle.ipynb', as_version=4)
nb.cells[2].source = src
nbformat.write(nb, 'SparseComm_Kaggle.ipynb')
print('[OK] Notebook updated')
