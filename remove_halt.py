import nbformat
import re

nb = nbformat.read('C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\SparseComm_Kaggle.ipynb', as_version=4)
source = nb.cells[2].source

# Remove the HALT and sys.exit(0)
halt_pattern = r'import sys\nprint\("\\n\[HALT\] Stopping execution here to prevent full 50\+ hour run. Remove sys\.exit\(0\) to proceed to full phases\."\)\nsys\.exit\(0\)\n'
source = re.sub(halt_pattern, '', source)

nb.cells[2].source = source
nbformat.write(nb, 'C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\SparseComm_Kaggle.ipynb')
print('Removed HALT sequence from notebook!')
