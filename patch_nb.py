import json

with open('SparseComm_Kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

source = nb['cells'][0]['source']

exec_start = -1
for i, line in enumerate(source):
    if 'datasets_to_run =' in line and 'gsm8k' in line:
        exec_start = i - 2
        break

if exec_start == -1:
    print('Execution block not found!')
    import sys; sys.exit(1)

with open('temp_check_lib.py', 'r', encoding='utf-8') as f:
    lib_lines = f.readlines()

lib_lines = [line if line.endswith('\n') else line + '\n' for line in lib_lines]

exec_lines = source[exec_start:]

new_exec = []
for line in exec_lines:
    if 'RUN_TOY_ONLY =' in line:
        new_exec.append('RUN_TOY_ONLY = False    # <-- set True to exit immediately after the Toy Run (to test pipeline)\n')
    elif 'current_config.n_train_samples =' in line:
        new_exec.append('    current_config.n_train_samples = 100\n')
    elif 'current_config.n_samples =' in line:
        new_exec.append('    current_config.n_samples = 300\n')
    elif 'current_config.controller_epochs =' in line:
        new_exec.append('    # current_config.controller_epochs override removed\n')
    else:
        new_exec.append(line)

nb['cells'][0]['source'] = lib_lines + new_exec

with open('SparseComm_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('SparseComm_Kaggle.ipynb updated successfully!')
