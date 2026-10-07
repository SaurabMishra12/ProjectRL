import numpy as np

def test_gae():
    """3-step toy episode: verify GAE against manual computation."""
    values = [0.1, 0.2, 0.3]
    rewards = [1.0, 0.0, 2.0]
    dones = [False, False, True]
    gamma = 0.99
    lam = 0.95

    n_steps = 3
    advantages = np.zeros(n_steps)
    returns = np.zeros(n_steps)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(n_steps)):
        is_done = dones[t]
        mask = 0.0 if is_done else 1.0
        if is_done:
            next_value = 0.0
            gae = 0.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * gae * mask
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]

    # Manual computation:
    # t=2 (done=True): mask=0, next_value->0, gae->0
    #   delta = 2.0 + 0.99*0*0 - 0.3 = 1.7
    #   gae = 1.7 + 0.99*0.95*0*0 = 1.7
    # t=1 (done=False): mask=1
    #   delta = 0.0 + 0.99*0.3*1 - 0.2 = 0.097
    #   gae = 0.097 + 0.99*0.95*1.7*1 = 0.097 + 1.59885 = 1.69585
    # t=0 (done=False): mask=1
    #   delta = 1.0 + 0.99*0.2*1 - 0.1 = 1.098
    #   gae = 1.098 + 0.99*0.95*1.69585*1 = 1.098 + 1.594954... = 2.692954...

    expected = [
        1.098 + 0.99*0.95*1.69585,
        0.097 + 0.99*0.95*1.7,
        1.7
    ]
    print("=== GAE Verification ===")
    for i in range(3):
        match = abs(advantages[i] - expected[i]) < 1e-10
        print(f"  t={i}: computed={advantages[i]:.10f} expected={expected[i]:.10f} {'✅' if match else '❌'}")
    print(f"  Overall: {'PASS' if np.allclose(advantages, expected) else 'FAIL'}")

def test_ppo_gradients():
    """Numerical gradient check for PPO actor update."""
    np.random.seed(42)
    mean_a = np.random.randn(5)
    log_std = np.random.randn(5) * 0.1
    sigma = np.exp(log_std)
    action = mean_a + np.random.randn(5) * sigma
    old_log_prob = -0.5 * np.sum(((action - mean_a)/sigma)**2 + 2*log_std + np.log(2*np.pi))
    adv = 1.5
    ppo_epsilon = 0.2
    entropy_coef = 0.01

    def compute_loss(mv, lsv):
        s = np.exp(lsv)
        nlp = -0.5 * np.sum(((action - mv)/s)**2 + 2*lsv + np.log(2*np.pi))
        r = np.exp(nlp - old_log_prob)
        s1 = r * adv
        s2 = np.clip(r, 1-ppo_epsilon, 1+ppo_epsilon) * adv
        H = np.sum(lsv + 0.5 + 0.5*np.log(2*np.pi))
        return -min(s1, s2) - entropy_coef * H

    eps = 1e-5
    num_grad_mean = np.zeros_like(mean_a)
    for i in range(len(mean_a)):
        mp = mean_a.copy(); mp[i] += eps
        mm = mean_a.copy(); mm[i] -= eps
        num_grad_mean[i] = (compute_loss(mp, log_std) - compute_loss(mm, log_std)) / (2*eps)

    num_grad_logstd = np.zeros_like(log_std)
    for i in range(len(log_std)):
        lp = log_std.copy(); lp[i] += eps
        lm = log_std.copy(); lm[i] -= eps
        num_grad_logstd[i] = (compute_loss(mean_a, lp) - compute_loss(mean_a, lm)) / (2*eps)

    # Analytic (at ratio=1.0):
    ratio = 1.0
    surr1 = ratio * adv
    surr2 = np.clip(ratio, 1-ppo_epsilon, 1+ppo_epsilon) * adv
    d_log_prob_wrt_mean = (action - mean_a) / (sigma**2 + 1e-8)
    clipped_ratio = np.clip(ratio, 1.0 - ppo_epsilon, 1.0 + ppo_epsilon)
    use_clipped = (ratio > 1.0 + ppo_epsilon and adv > 0) or (ratio < 1.0 - ppo_epsilon and adv < 0)
    ana_d_mean = adv * (clipped_ratio if use_clipped else ratio) * d_log_prob_wrt_mean
    ana_d_logstd = (adv * ratio) * ((action - mean_a)**2 / (sigma**2 + 1e-8) - 1.0)
    ana_d_logstd_full = ana_d_logstd + entropy_coef

    # Loss gradient = -d_surrogate/d_theta
    ana_grad_mean = -ana_d_mean
    ana_grad_logstd = -ana_d_logstd_full

    print("\n=== PPO Gradient Verification ===")
    print(f"  Mean grad max diff: {np.max(np.abs(num_grad_mean - ana_grad_mean)):.2e}")
    print(f"  LogStd grad max diff: {np.max(np.abs(num_grad_logstd - ana_grad_logstd)):.2e}")
    print(f"  Mean: {'PASS' if np.allclose(num_grad_mean, ana_grad_mean, atol=1e-4) else 'FAIL'}")
    print(f"  LogStd: {'PASS' if np.allclose(num_grad_logstd, ana_grad_logstd, atol=1e-4) else 'FAIL'}")

if __name__ == '__main__':
    test_gae()
    test_ppo_gradients()
