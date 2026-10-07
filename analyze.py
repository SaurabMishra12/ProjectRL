import json, glob, numpy as np

for f in sorted(glob.glob('08-06results/toy_*.json')):
    dataset = f.split('_')[-1].split('.')[0]
    with open(f, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    key = list(data['detailed_samples'].keys())[0]
    samples = data['detailed_samples'][key]
    
    acc = np.mean([s['is_correct'] for s in samples])
    halt_rounds = [s.get('halt_round', 0) for s in samples]
    token_red = [s.get('token_reduction', 0) for s in samples]
    
    # Calculate some PPO-specific metrics if available
    # Check if any messages were blocked (sparse comm) vs FULL
    modes_str = []
    for s in samples:
        if 'mode_distribution' in s:
            md = s['mode_distribution']
            modes_str.append(f"F:{md.get('FULL', 0)} P:{md.get('PARTIAL', 0)} B:{md.get('BLOCK', 0)}")
    
    print(f'=== {dataset.upper()} ===')
    print(f'Accuracy: {acc*100:.1f}% ({sum([s["is_correct"] for s in samples])}/{len(samples)})')
    print(f'Avg Halt Round: {np.mean(halt_rounds):.1f} (rounds: {halt_rounds})')
    print(f'Avg Token Reduction: {np.mean(token_red)*100:.1f}%')
    if modes_str:
        print(f'Mode distributions (by sample): {modes_str}')
    print()
