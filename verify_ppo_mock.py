import numpy as np

def test_gae():
    # Mocking transitions
    # Values predicted by critic
    values = [0.1, 0.2, 0.3]
    # Rewards received
    rewards = [1.0, 0.0, 2.0]
    # Dones
    dones = [False, False, True]
    
    saved_transitions = []
    for t in range(3):
        saved_transitions.append({'reward': rewards[t], 'value': values[t], 'done': dones[t]})
    
    gamma_discount = 0.99
    gae_lambda = 0.95
    
    # Run the GAE loop as implemented in our updated temp_nb.py
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(3)):
        is_done = saved_transitions[t].get('done', False)
        mask = 0.0 if is_done else 1.0
        if is_done:
            next_value = 0.0
            gae = 0.0
            
        delta = saved_transitions[t]['reward'] + gamma_discount * next_value * mask - saved_transitions[t]['value']
        gae = delta + gamma_discount * gae_lambda * gae * mask
        
        saved_transitions[t]['advantage'] = gae
        saved_transitions[t]['return'] = gae + saved_transitions[t]['value']
        next_value = saved_transitions[t]['value']
        
    # Manual Calculation
    delta_2 = 2.0 + 0.99 * 0.0 - 0.3 # = 1.7
    gae_2 = 1.7
    
    delta_1 = 0.0 + 0.99 * 0.3 - 0.2 # = 0.097
    gae_1 = delta_1 + 0.99 * 0.95 * gae_2 # = 0.097 + 1.59885 = 1.69585
    
    delta_0 = 1.0 + 0.99 * 0.2 - 0.1 # = 1.098
    gae_0 = delta_0 + 0.99 * 0.95 * gae_1 # = 1.098 + 1.5949469 = 2.6929469
    
    expected_advantages = [gae_0, gae_1, gae_2]
    expected_returns = [gae_0 + values[0], gae_1 + values[1], gae_2 + values[2]]
    
    computed_adv = [t['advantage'] for t in saved_transitions]
    computed_ret = [t['return'] for t in saved_transitions]
    
    print("=== GAE Math Verification ===")
    print(f"Expected Adv: {expected_advantages}")
    print(f"Computed Adv: {computed_adv}")
    print(f"Adv Match: {np.allclose(expected_advantages, computed_adv)}")

def test_ppo_gradients():
    np.random.seed(42)
    mean_a = np.random.randn(10)
    log_std = np.random.randn(10) * 0.1
    sigma = np.exp(log_std)
    
    action = mean_a + np.random.randn(10) * sigma
    
    old_log_prob = -0.5 * np.sum(((action - mean_a) / sigma)**2 + 2*log_std + np.log(2*np.pi))
    adv = 1.5
    ppo_epsilon = 0.2
    entropy_coef = 0.01
    
    def compute_loss(mean_vec, log_std_vec):
        sig = np.exp(log_std_vec)
        new_log_prob = -0.5 * np.sum(((action - mean_vec) / sig)**2 + 2*log_std_vec + np.log(2*np.pi))
        ratio = np.exp(new_log_prob - old_log_prob)
        surr1 = ratio * adv
        surr2 = np.clip(ratio, 1.0 - ppo_epsilon, 1.0 + ppo_epsilon) * adv
        actor_loss = -np.minimum(surr1, surr2)
        entropy = np.sum(log_std_vec + 0.5 + 0.5 * np.log(2 * np.pi))
        return actor_loss - entropy_coef * entropy
        
    eps = 1e-5
    
    num_grad_mean = np.zeros_like(mean_a)
    for i in range(len(mean_a)):
        mean_plus = mean_a.copy(); mean_plus[i] += eps
        loss_plus = compute_loss(mean_plus, log_std)
        mean_minus = mean_a.copy(); mean_minus[i] -= eps
        loss_minus = compute_loss(mean_minus, log_std)
        num_grad_mean[i] = (loss_plus - loss_minus) / (2 * eps)
        
    num_grad_log_std = np.zeros_like(log_std)
    for i in range(len(log_std)):
        std_plus = log_std.copy(); std_plus[i] += eps
        loss_plus = compute_loss(mean_a, std_plus)
        std_minus = log_std.copy(); std_minus[i] -= eps
        loss_minus = compute_loss(mean_a, std_minus)
        num_grad_log_std[i] = (loss_plus - loss_minus) / (2 * eps)
        
    # Analytic
    ratio = 1.0
    surr1 = ratio * adv
    surr2 = np.clip(ratio, 1.0 - ppo_epsilon, 1.0 + ppo_epsilon) * adv
    d_log_prob_wrt_mean = (action - mean_a) / (sigma**2 + 1e-8)
    
    if surr1 <= surr2:
        d_mean = (adv * ratio) * d_log_prob_wrt_mean
    else:
        d_mean = 0.0
        
    d_log_std_analytic = (adv * ratio) * ((action - mean_a)**2 / (sigma**2 + 1e-8) - 1.0)
    
    analytic_grad_mean = -d_mean
    analytic_grad_log_std = -d_log_std_analytic - entropy_coef
    
    print("\n=== PPO Gradient Verification ===")
    print(f"Max Diff Mean: {np.max(np.abs(num_grad_mean - analytic_grad_mean)):.2e}")
    print(f"Max Diff LogStd: {np.max(np.abs(num_grad_log_std - analytic_grad_log_std)):.2e}")
    print(f"Match Mean (tol 1e-4): {np.allclose(num_grad_mean, analytic_grad_mean, atol=1e-4)}")
    print(f"Match LogStd (tol 1e-4): {np.allclose(num_grad_log_std, analytic_grad_log_std, atol=1e-4)}")

if __name__ == '__main__':
    test_gae()
    test_ppo_gradients()
