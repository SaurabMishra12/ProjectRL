import re

with open('config.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('"math"', '"aqua"')
text = text.replace("'math'", "'aqua'")
text = text.replace('get_math_config', 'get_aqua_config')

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done config!')
