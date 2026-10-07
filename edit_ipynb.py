import nbformat
import re

nb = nbformat.read('C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\SparseComm_Kaggle.ipynb', as_version=4)
source = nb.cells[2].source

new_estimate = '''    def _estimate_confidence(self, text: str) -> float:
        """Extract explicit confidence score from response text."""
        if not text:
            return 0.5
            
        import re
        match = re.search(r'[Cc]onfidence:\s*([0-9]+(?:\\.[0-9]+)?)', text)
        if match:
            try:
                score = float(match.group(1))
                return max(0.1, min(1.0, score / 10.0))
            except ValueError:
                pass
        return 0.5'''

import re

match_est = re.search(r'    def _estimate_confidence\(self, text: str\) -> float:.*?(?=\n    def )', source, re.DOTALL)
if match_est:
    source = source[:match_est.start()] + new_estimate + "\n\n" + source[match_est.end():]

match_mock = re.search(r'class MockLLMBackend\(LLMBackend\):.*?(?=\nclass )', source, re.DOTALL)
if match_mock:
    source = source[:match_mock.start()] + source[match_mock.end():]

nb.cells[2].source = source
nbformat.write(nb, 'C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\SparseComm_Kaggle.ipynb')
print('Notebook updated directly. MockLLMBackend and heuristic estimation completely removed.')
