import nbformat

notebook_path = "SparseComm_Kaggle.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

cell_code = nb.cells[-1].source

if "importlib.reload(llm_backend)" not in cell_code:
    reload_code = "import importlib\ntry:\n    import llm_backend\n    importlib.reload(llm_backend)\nexcept Exception:\n    pass\n\n"
    nb.cells[-1].source = reload_code + cell_code
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Added reload to notebook cell")
else:
    print("Reload already present")
