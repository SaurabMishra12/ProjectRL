import json
import os

def fix_py(filepath):
    if not os.path.exists(filepath): return
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return
    
    modified = False
    new_lines = []
    for line in lines:
        new_line = line
        if "np.fill_diagonal(W, 1.0)" in line:
            new_line = line.replace("np.fill_diagonal(W, 1.0)", "np.fill_diagonal(W, 1.0)")
        if "pass # np.fill_diagonal(W_raw, -500.0)" in line:
            new_line = line.replace("pass # np.fill_diagonal(W_raw, -500.0)", "pass # pass # np.fill_diagonal(W_raw, -500.0)")
        if "np.fill_diagonal(self.W, 1.0)" in line:
            new_line = line.replace("np.fill_diagonal(self.W, 1.0)", "np.fill_diagonal(self.W, 1.0)")
        
        if new_line != line:
            modified = True
        new_lines.append(new_line)
        
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Fixed py: {filepath}")

import glob
python_files = glob.glob('/home/saurab/Documents/projectRL/*.py')
for py in python_files:
    fix_py(py)
