import json
import glob

files = glob.glob('resultsall/*.json') + glob.glob('08-06results/*.json') + ['Master_SparseComm_Full_Results.json']
for f in files:
    try:
        data = json.load(open(f))
    except:
        continue
    if isinstance(data, dict):
        # Look for 'Experiments'
        if 'Experiments' in data:
            for k, v in data['Experiments'].items():
                if 'gsm8k' in k.lower():
                    stats = v.get('Aggregate_Stats', {})
                    if stats:
                        print(f"File: {f} | Key: {k} | Acc: {stats.get('Accuracy_Percentage')} | Tokens: {stats.get('Average_Communication_Tokens')}")
        
        # Look for 'stats' in ablation files
        if 'stats' in data and data['stats']:
            print(f"File: {f} | Stats: {data['stats']}")
