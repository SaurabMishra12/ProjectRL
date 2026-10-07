import glob
for f in sorted(glob.glob('08-06results/continuous/*.jsonl')):
    with open(f, 'r', encoding='utf-8') as file:
        count = sum(1 for line in file if line.strip())
        print(f"{f}: {count} samples")
