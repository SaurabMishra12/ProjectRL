import re

with open('temp_check.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix include_token_penalty
text = text.replace(
    "is_final_step=is_final,\n    )",
    "is_final_step=is_final,\n        include_token_penalty=is_final,\n    )"
)

# Replace dataset name 'math' with 'aqua' in key places
text = text.replace('"math"', '"aqua"')
text = text.replace("'math'", "'aqua'")

text = text.replace('get_math_config', 'get_aqua_config')
text = text.replace('def load_math', 'def load_aqua')
text = text.replace('load_math(', 'load_aqua(')
text = text.replace('AQuA Math Problem', 'AQuA Problem')

with open('temp_check.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done!')
