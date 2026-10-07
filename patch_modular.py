import os

files_to_patch = ['pipeline.py', 'dataset_loaders.py', 'controller.py', 'llm_backend.py']

for fname in files_to_patch:
    if not os.path.exists(fname): continue
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read()

    # Math to AQuA
    text = text.replace('"math"', '"aqua"')
    text = text.replace("'math'", "'aqua'")
    text = text.replace('get_math_config', 'get_aqua_config')
    text = text.replace('def load_math', 'def load_aqua')
    text = text.replace('load_math(', 'load_aqua(')
    text = text.replace('AQuA Math Problem', 'AQuA Problem')

    # Token penalty in pipeline.py
    if fname == 'pipeline.py':
        text = text.replace(
            "is_final_step=is_final,\n    )",
            "is_final_step=is_final,\n        include_token_penalty=is_final,\n    )"
        )
    
    # HF temperature in llm_backend.py
    if fname == 'llm_backend.py':
        text = text.replace(
            "temperature=0.7,\n                top_p=0.95,\n                do_sample=True,",
            "do_sample=False,  # Deterministic generation for evaluation"
        )
        text = text.replace(
            "temperature=0.0,  # Deterministic generation for evaluation\n                top_p=0.95,\n                do_sample=True,",
            "do_sample=False,  # Deterministic generation for evaluation"
        )

    with open(fname, 'w', encoding='utf-8') as f:
        f.write(text)

print('Modular files patched!')
