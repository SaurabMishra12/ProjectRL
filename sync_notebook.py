import json
import sys

def sync_nb_from_py(py_path="temp_nb.py", ipynb_path="SparseComm_Kaggle.ipynb"):
    with open(py_path, "r", encoding="utf-8") as f:
        py_code = f.read()
        
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    # We find the cell that contains the main code. 
    # Usually it is the last cell or cell index 2.
    # Let's search for a code cell that has more than 100 lines.
    found_idx = None
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell["cell_type"] == "code" and idx > 1:
            found_idx = idx
            break
            
    if found_idx is None:
        # Fallback to index 2
        found_idx = 2
        
    print(f"Syncing temp_nb.py -> {ipynb_path} (cell index {found_idx})")
    
    # Notebook source expects list of lines, each ending with \n, or a single string.
    # A single string is also supported in nbformat, but a list of lines is safer.
    lines = py_code.splitlines(keepends=True)
    nb["cells"][found_idx]["source"] = lines
    
    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
        
    print("Sync complete.")

if __name__ == "__main__":
    sync_nb_from_py()
