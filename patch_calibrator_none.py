import json

with open("SparseComm_Kaggle.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

text = "".join(nb['cells'][0]['source'])

target = "                 calibrator_fitted=np.array(self.calibrator._fitted))"
replacement = "                 calibrator_fitted=np.array(self.calibrator._fitted) if self.calibrator is not None else np.array(False))"

if target in text:
    text = text.replace(target, replacement)
    print("Patched save_model in notebook!")
else:
    print("Could not find target in notebook!")

new_source = []
for line in text.splitlines():
    new_source.append(line + "\n")
if new_source and not text.endswith("\n"):
    new_source[-1] = new_source[-1].rstrip("\n")

nb['cells'][0]['source'] = new_source

with open("SparseComm_Kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

try:
    with open("temp_check_lib.py", "r", encoding="utf-8") as f:
        lib_text = f.read()

    if target in lib_text:
        lib_text = lib_text.replace(target, replacement)
        print("Patched save_model in temp_check_lib.py!")

    with open("temp_check_lib.py", "w", encoding="utf-8") as f:
        f.write(lib_text)
except Exception as e:
    print(f"Error patching temp_check_lib.py: {e}")
