import numpy as np
import sys
# Import the classes we just modified from temp_nb
from temp_nb import IBController, Config

def test_gae():
    config = Config()
    config.n_agents = 3
    config.n_rounds = 3
    controller = IBController(config)
    
    # Toy episode 3 steps
    # Values predicted by critic
    values = [0.1, 0.2, 0.3]
    # Rewards received
    rewards = [1.0, 0.0, 2.0]
    # Dones
    dones = [False, False, True]
    
    for t in range(3):
        controller.store_transition(
            state=np.zeros(10), action=np.zeros(10), reward=rewards[t], 
            value=values[t], log_prob=0.0, done=dones[t]
        )
    
    # Compute GAE manually
    gamma = controller.gamma_discount
    lam = controller.gae_lambda
    
    # Step 2 (done=True):
    # delta_2 = r_2 + gamma * 0.0 - v_2 = 2.0 + 0.0 - 0.3 = 1.7
    # gae_2 = 1.7
    delta_2 = 2.0 - 0.3
    gae_2 = delta_2
    
    # Step 1:
    # delta_1 = r_1 + gamma * v_2 - v_1 = 0.0 + 0.99 * 0.3 - 0.2 = 0.097
    # gae_1 = delta_1 + gamma * lam * gae_2 = 0.097 + 0.99 * 0.95 * 1.7 = 1.69615
    delta_1 = 0.0 + gamma * values[2] - values[1]
    gae_1 = delta_1 + gamma * lam * gae_2
    
    # Step 0:
    # delta_0 = r_0 + gamma * v_1 - v_0 = 1.0 + 0.99 * 0.2 - 0.1 = 1.098
    # gae_0 = delta_0 + gamma * lam * gae_1 = 1.098 + 0.99 * 0.95 * 1.69615 = 2.69353
    delta_0 = 1.0 + gamma * values[1] - values[0]
    gae_0 = delta_0 + gamma * lam * gae_1
    
    expected_advantages = [gae_0, gae_1, gae_2]
    expected_returns = [gae_0 + values[0], gae_1 + values[1], gae_2 + values[2]]
    
    controller._compute_gae()
    
    computed_adv = [t['advantage'] for t in controller.saved_transitions]
    computed_ret = [t['return'] for t in controller.saved_transitions]
    
    print("GAE Test Results:")
    print(f"Expected Adv: {expected_advantages}")
    print(f"Computed Adv: {computed_adv}")
    print(f"Adv Match: {np.allclose(expected_advantages, computed_adv)}")
    print(f"Expected Ret: {expected_returns}")
    print(f"Computed Ret: {computed_ret}")
    print(f"Ret Match: {np.allclose(expected_returns, computed_ret)}")
    print("-" * 40)

def test_ppo_gradients():
    # Setup dummy data
    config = Config()
    config.n_agents = 2
    controller = IBController(config)
    
    state = np.random.randn(config.n_agents * config.agent_dim)
    # Forward pass to get mean and std
    mean_a, log_std, v = controller.actor_critic(state[None, :])
    mean_a = mean_a[0]
    log_std = log_std[0]
    sigma = np.exp(log_std)
    
    # Sample an action
    action = mean_a + np.random.randn(*mean_a.shape) * sigma
    
    # Mock transition data
    old_log_prob = -0.5 * np.sum(((action - mean_a) / sigma)**2 + 2*log_std + np.log(2*np.pi))
    adv = 1.5 # Positive advantage
    
    eps = 1e-5
    
    # We want to test the gradient of the surrogate objective wrt mean_a and log_std
    # Surrogate: min(ratio * adv, clip(ratio) * adv) + entropy_coef * H
    # H = sum(log_std + 0.5 + 0.5*log(2pi))
    entropy_coef = 0.01
    
    def compute_loss(mean_vec, log_std_vec):
        sig = np.exp(log_std_vec)
        new_log_prob = -0.5 * np.sum(((action - mean_vec) / sig)**2 + 2*log_std_vec + np.log(2*np.pi))
        ratio = np.exp(new_log_prob - old_log_prob)
        surr1 = ratio * adv
        surr2 = np.clip(ratio, 1.0 - controller.ppo_epsilon, 1.0 + controller.ppo_epsilon) * adv
        actor_loss = -np.minimum(surr1, surr2)
        entropy = np.sum(log_std_vec + 0.5 + 0.5 * np.log(2 * np.pi))
        return actor_loss - entropy_coef * entropy

    # 1. Numerical gradient wrt mean
    num_grad_mean = np.zeros_like(mean_a)
    for i in range(len(mean_a)):
        mean_plus = mean_a.copy()
        mean_plus[i] += eps
        loss_plus = compute_loss(mean_plus, log_std)
        
        mean_minus = mean_a.copy()
        mean_minus[i] -= eps
        loss_minus = compute_loss(mean_minus, log_std)
        
        num_grad_mean[i] = (loss_plus - loss_minus) / (2 * eps)
        
    # 2. Numerical gradient wrt log_std
    num_grad_log_std = np.zeros_like(log_std)
    for i in range(len(log_std)):
        std_plus = log_std.copy()
        std_plus[i] += eps
        loss_plus = compute_loss(mean_a, std_plus)
        
        std_minus = log_std.copy()
        std_minus[i] -= eps
        loss_minus = compute_loss(mean_a, std_minus)
        
        num_grad_log_std[i] = (loss_plus - loss_minus) / (2 * eps)
        
    # 3. Analytic gradients (what our code does)
    # ratio = 1.0 since old_log_prob == new_log_prob initially
    ratio = 1.0
    surr1 = ratio * adv
    surr2 = np.clip(ratio, 1.0 - controller.ppo_epsilon, 1.0 + controller.ppo_epsilon) * adv
    
    d_log_prob_wrt_mean = (action - mean_a) / (sigma**2 + 1e-8)
    
    if surr1 <= surr2:
        d_mean = (adv * ratio) * d_log_prob_wrt_mean
    else:
        d_mean = 0.0
        
    d_log_std_analytic = (adv * ratio) * ((action - mean_a)**2 / (sigma**2 + 1e-8) - 1.0)
    # The loss we optimize is NEGATIVE surrogate, so analytic grad of loss is -d_mean, -d_log_std
    # PLUS the entropy bonus which is -entropy_coef for log_std (since we minimize -H)
    analytic_grad_mean = -d_mean
    analytic_grad_log_std = -d_log_std_analytic - entropy_coef
    
    print("PPO Gradient Test Results:")
    print(f"Mean Grad Match: {np.allclose(num_grad_mean, analytic_grad_mean, atol=1e-4)}")
    print(f"Log_Std Grad Match: {np.allclose(num_grad_log_std, analytic_grad_log_std, atol=1e-4)}")
    print(f"Max Diff Mean: {np.max(np.abs(num_grad_mean - analytic_grad_mean))}")
    print(f"Max Diff LogStd: {np.max(np.abs(num_grad_log_std - analytic_grad_log_std))}")

if __name__ == '__main__':
    test_gae()
    test_ppo_gradients()
