import json

with open("SparseComm_Kaggle.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

text = "".join(nb['cells'][0]['source'])

bad_dump = """            pickle.dump({
                'ir': self.calibrator.ir if self.calibrator._fitted else None,
                '_fitted': self.calibrator._fitted,
                '_agent_vars': self.calibrator._agent_vars,
                '_agent_historical_acc': self.calibrator._agent_historical_acc,
            }, f)"""

good_dump = """            if self.calibrator is not None:
                pickle.dump({
                    'ir': self.calibrator.ir if self.calibrator._fitted else None,
                    '_fitted': self.calibrator._fitted,
                    '_agent_vars': self.calibrator._agent_vars,
                    '_agent_historical_acc': self.calibrator._agent_historical_acc,
                }, f)
            else:
                pickle.dump(None, f)"""

if bad_dump in text:
    text = text.replace(bad_dump, good_dump)
    print("Patched pickle.dump in notebook!")
else:
    print("Could not find bad_dump in notebook!")

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

    if bad_dump in lib_text:
        lib_text = lib_text.replace(bad_dump, good_dump)
        print("Patched pickle.dump in temp_check_lib.py!")

    with open("temp_check_lib.py", "w", encoding="utf-8") as f:
        f.write(lib_text)
except Exception as e:
    print("No temp check lib or error.")
