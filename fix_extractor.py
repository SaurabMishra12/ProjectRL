import nbformat

nb = nbformat.read('C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\SparseComm_Kaggle.ipynb', as_version=4)
source = nb.cells[2].source

old_fallback = """        # Fallback: last line or last number
        lines = [l.strip() for l in text.strip().split('\\n') if l.strip()]"""

new_fallback = """        # Fallback: remove confidence line before looking for last number
        text_no_conf = re.sub(r'(?i)confidence:\s*\d+(?:\.\d+)?', '', text)
        lines = [l.strip() for l in text_no_conf.strip().split('\\n') if l.strip()]"""

source = source.replace(old_fallback, new_fallback)

nb.cells[2].source = source
nbformat.write(nb, 'C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\SparseComm_Kaggle.ipynb')
print('Extractor fixed.')
