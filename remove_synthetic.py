import nbformat
import re

nb = nbformat.read('C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\SparseComm_Kaggle.ipynb', as_version=4)
source = nb.cells[2].source

# 1. Remove _generate_synthetic_samples definition entirely
synth_def_pattern = r'def _generate_synthetic_samples.*?return all_samples\[:n_train\], all_samples\[n_train:n_train\+n_test\]'
source = re.sub(synth_def_pattern, '', source, flags=re.DOTALL)

# 2. Replace the synthetic fallbacks with raise statements
source = re.sub(
    r'logger\.warning\(f"Could not load ([^"]+) from HuggingFace: \{e\}"\)\s+return _generate_synthetic_samples\(n_train, n_test, seed\)',
    r'raise RuntimeError(f"Could not load \1 from HuggingFace: {e}")',
    source
)

source = re.sub(
    r'logger\.warning\(f"Could not load ([^"]+) directly: \{e\}"\)\s+return _generate_synthetic_samples\(n_train, n_test, seed\)',
    r'raise RuntimeError(f"Could not load \1 directly: {e}")',
    source
)

source = re.sub(
    r'logger\.warning\(f"Could not load ([^"]+): \{e\}"\)\s+return _generate_synthetic_samples\(n_train, n_test, seed\)',
    r'raise RuntimeError(f"Could not load \1: {e}")',
    source
)

# 3. Change avg_accuracy = 0.5 to avg_accuracy = 0.0 in initialize
source = source.replace('avg_accuracy = 0.5', 'avg_accuracy = 0.0')

nb.cells[2].source = source
nbformat.write(nb, 'C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\SparseComm_Kaggle.ipynb')
print('Synthetic fallbacks and hardcoded 0.5 accuracy initialized successfully removed from notebook.')
