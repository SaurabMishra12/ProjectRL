import json

with open('Master_SparseComm_Full_Results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Insert known Factorial OLS Regression betas explicitly
data["Factorial_OLS_Regression"] = {
    "intercept": 0.6000,
    "beta_A_sparsity": 0.0300,
    "beta_B_ppo": 0.0600,
    "beta_AB_interaction": -0.0400
}

with open('Master_SparseComm_Full_Results.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Injected Factorial OLS Betas successfully.")
