import json
import glob
import os
import numpy as np

master_data = {
    "Project_Name": "SparseComm RL",
    "Factorial_OLS_Regression": {},
    "Experiments": {}
}

# 1. Extract Factorial OLS Regression from kaggle_latest_results.json
try:
    with open('08-06results/kaggle_latest_results.json', 'r', encoding='utf-8') as f:
        latest = json.load(f)
        master_data["Factorial_OLS_Regression"] = latest.get("stats", {}).get("factorial_ols_betas", {})
except Exception as e:
    master_data["Factorial_OLS_Regression"] = {"error": str(e)}

# 2. Iterate over all continuous stream files
stream_files = glob.glob('08-06results/continuous/*.jsonl')

for filepath in stream_files:
    basename = os.path.basename(filepath).replace('_stream.jsonl', '')
    
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                samples.append(json.loads(line))
            except:
                pass
                
    if not samples:
        continue
        
    # Calculate aggregate stats
    total_samples = len(samples)
    accuracy = np.mean([s.get('is_correct', False) for s in samples])
    
    # Try to calculate comm tokens if present
    comm_tokens = [s.get('total_tokens_transmitted', 0) for s in samples if 'total_tokens_transmitted' in s]
    avg_comm_tokens = np.mean(comm_tokens) if comm_tokens else 0.0
    
    master_data["Experiments"][basename] = {
        "Aggregate_Stats": {
            "Total_Samples": total_samples,
            "Accuracy_Percentage": float(accuracy * 100),
            "Average_Communication_Tokens": float(avg_comm_tokens)
        },
        "Detailed_Samples": samples
    }

# 3. Save Master JSON
output_path = 'Master_SparseComm_Full_Results.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(master_data, f, indent=2)

print(f"Master JSON created successfully at {output_path} with {len(master_data['Experiments'])} experiments!")
