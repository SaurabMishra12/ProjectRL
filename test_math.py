from dataset_loaders import load_math

print("Loading Math...")
samples = load_math(n_samples=5)
for s in samples:
    print("-" * 40)
    print("Q:", s['question'])
    print("A:", s['answer'])
print("SUCCESS!")
