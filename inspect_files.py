import json, pathlib, re

d = pathlib.Path('resultsall')
for f in sorted(d.glob('*.json')):
    if f.name == 'analysis.json':
        continue
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
        sizes = {k: len(v) for k, v in data.get('detailed_samples', {}).items()}
        print(f'{f.name}: {sizes}')
    except Exception as e:
        txt = f.read_text(encoding='utf-8', errors='replace')
        keys = re.findall(r'"(Self-Consistency[^"]{0,30}?)"\s*:\s*\[', txt)
        keys = list(dict.fromkeys(keys))
        print(f'{f.name}: PARSE_ERROR  detected_keys={keys}  size={f.stat().st_size}')
