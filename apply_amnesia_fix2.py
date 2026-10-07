import json
import os
import glob

def fix_notebook(filepath):
    if not os.path.exists(filepath): return
    with open(filepath, 'r') as f:
        try:
            nb = json.load(f)
        except Exception:
            return
    
    modified = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                new_line = line
                if "np.fill_diagonal(W, 1.0)" in line:
                    new_line = line.replace("np.fill_diagonal(W, 1.0)", "np.fill_diagonal(W, 1.0)")
                if "np.fill_diagonal(full_W, 1.0)" in line:
                    new_line = line.replace("np.fill_diagonal(full_W, 1.0)", "np.fill_diagonal(full_W, 1.0)")
                
                if new_line != line:
                    modified = True
                new_source.append(new_line)
            cell['source'] = new_source
            
    if modified:
        with open(filepath, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Fixed notebook: {filepath}")

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
        if "np.fill_diagonal(full_W, 1.0)" in line:
            new_line = line.replace("np.fill_diagonal(full_W, 1.0)", "np.fill_diagonal(full_W, 1.0)")
        
        if new_line != line:
            modified = True
        new_lines.append(new_line)
        
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Fixed py: {filepath}")

notebooks = glob.glob('/home/saurab/Documents/projectRL/*.ipynb')
for nb in notebooks:
    fix_notebook(nb)

python_files = glob.glob('/home/saurab/Documents/projectRL/*.py')
for py in python_files:
    fix_py(py)
