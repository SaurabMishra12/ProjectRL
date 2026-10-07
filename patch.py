
import re

with open('temp_check.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix include_token_penalty
text = re.sub(
    r'is_final_step=is_final,
\s*\)',
    'is_final_step=is_final,\n        include_token_penalty=is_final, # Bug fix: Apply token penalty on final round\n    )',
    text
)

text = text.replace(chr(34) + 'math' + chr(34), chr(34) + 'aqua' + chr(34))
text = text.replace(chr(39) + 'math' + chr(39), chr(39) + 'aqua' + chr(39))

text = text.replace('get_math_config', 'get_aqua_config')
text = text.replace('def load_math', 'def load_aqua')
text = text.replace('load_math(', 'load_aqua(')
text = text.replace('AQuA Math Problem', 'AQuA Problem')

with open('temp_check.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done!')
