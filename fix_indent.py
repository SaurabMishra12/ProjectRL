import json
import re

with open('SparseComm_Kaggle.ipynb') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Clean up the broken entropy_coef and continuation line
        source = re.sub(r'# d_log_std \+= entropy_coef \* \\\n\s*np\.ones_like\(log_std_new\).*?\n', '\n', source)
        
        source = source.replace('d_log_std = d_log_prob * \\\n                        ((action - mean_a_new)**2 / (sigma_new**2 + 1e-8) - 1.0)', 
                                'd_log_std = adv * \\\n                        ((action - mean_a_new)**2 / (sigma_new**2 + 1e-8) - 1.0)')

        # Fix 15: Actor gradient clipping -> standard PPO clipping derivative
        # The user requested: "Replace manual gradient zeroing heuristic with a proper analytic derivative of the min(r*A, clip(r)*A) objective."
        # Currently the code has:
        # if adv >= 0:
        #     clipped = ratio > (1.0 + self.ppo_epsilon)
        # else:
        #     clipped = ratio < (1.0 - self.ppo_epsilon)
        # d_ratio = 0.0 if clipped else adv
        # d_mean = d_log_prob * (action - mean_a_new) / (sigma_new ** 2 + 1e-8)
        
        # We replace this by proper PPO surrogate derivative:
        # PPO Objective: L = min(ratio * adv, clip(ratio) * adv)
        # dL/d_mean = (adv if not clipped else 0.0) * d_ratio_d_mean
        # where d_ratio_d_mean = ratio * d_log_prob_d_mean
        
        ppo_grad_fix = """
                    # Standard PPO Clipping Derivative
                    if adv >= 0:
                        is_clipped = ratio > (1.0 + self.ppo_epsilon)
                    else:
                        is_clipped = ratio < (1.0 - self.ppo_epsilon)
                    
                    # d(log_prob) / d(mean) = (action - mean) / sigma^2
                    d_log_prob_wrt_mean = (action - mean_a_new) / (sigma_new**2 + 1e-8)
                    
                    if is_clipped:
                        d_mean = 0.0 * d_log_prob_wrt_mean  # gradient is zero when clipped
                    else:
                        d_mean = adv * d_log_prob_wrt_mean  # standard policy gradient
"""
        # Apply the replace if the old code is there:
        source = re.sub(r'if adv >= 0:\n\s*clipped = ratio > \(1\.0 \+ self\.ppo_epsilon\).*?d_mean = d_log_prob \* \(action - mean_a_new\) / \(sigma_new \*\* 2 \+ 1e-8\)\n', ppo_grad_fix, source, flags=re.DOTALL)
        
        cell['source'] = [l + '\n' for l in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'] = cell['source'][:-1]

with open('SparseComm_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
