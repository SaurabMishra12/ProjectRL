import json

with open("SparseComm_Kaggle.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

text = "".join(nb['cells'][0]['source'])

bad_loader = """                        MockS = dataclasses.make_dataclass('MockS', [('is_correct', bool), ('total_tokens_transmitted', float)])
                        results.sample_results.append(MockS(data.get('is_correct', False), data.get('total_tokens_transmitted', 0.0)))"""

good_loader = """                        fields = [(k, type(v)) for k, v in data.items()]
                        MockS = dataclasses.make_dataclass('MockS', fields)
                        results.sample_results.append(MockS(**data))"""

if bad_loader in text:
    text = text.replace(bad_loader, good_loader)
    print("Patched loader in notebook!")
else:
    print("Could not find bad_loader in notebook!")

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

    if bad_loader in lib_text:
        lib_text = lib_text.replace(bad_loader, good_loader)
        print("Patched loader in temp_check_lib.py!")

    with open("temp_check_lib.py", "w", encoding="utf-8") as f:
        f.write(lib_text)
except Exception as e:
    pass
