import json
import ast

def verify_notebook(filepath="SparseComm_Kaggle.ipynb"):
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    full_code = ""
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell["cell_type"] == "code":
            # Skip pip install lines
            source = "".join(cell["source"])
            lines = []
            for line in source.split("\n"):
                if not line.strip().startswith("!"):
                    lines.append(line)
            
            cell_code = "\n".join(lines)
            full_code += "\n" + cell_code
            
    print("Notebook parsed. Attempting Syntax Verification via CPython...")
    with open("full_notebook_code.py", "w", encoding="utf-8") as f:
        f.write(full_code)
        
    try:
        compile(full_code, "<notebook>", "exec")
        print("✅ SUCCESS: The notebook structure has 0 syntax errors and is perfectly formatted!")
    except SyntaxError as e:
        print(f"❌ SYNTAX ERROR DETECTED:\\n{e}\\nLine: {e.lineno}\\nText: {e.text}")
        
if __name__ == "__main__":
    verify_notebook()
