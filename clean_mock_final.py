import nbformat
import re

nb = nbformat.read('C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\SparseComm_Kaggle.ipynb', as_version=4)
source = nb.cells[2].source

# 1. Remove dangling MockLLMBackend methods absorbed into LLMBackend
# It starts at `def __init__(self, base_accuracy...` and ends right before `class NvidiaDeepSeekBackend`
mock_methods_pattern = r'    def __init__\(\n        self,\n        base_accuracy: float = 0\.65.*?(?=\nclass NvidiaDeepSeekBackend)'
source = re.sub(mock_methods_pattern, '', source, flags=re.DOTALL)

# 2. Remove the guard clause in run_experiment
guard_clause_pattern = r'    # ADD THIS GUARD CLAUSE\n    if config\.llm_backend == "mock":\n        if evaluate:\n            logger\.warning\([^)]+\)\n        else:\n            logger\.info\([^)]+\)\n\n'
source = re.sub(guard_clause_pattern, '', source, flags=re.DOTALL)

nb.cells[2].source = source
nbformat.write(nb, 'C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\SparseComm_Kaggle.ipynb')
print('Final mock cleanup complete!')
