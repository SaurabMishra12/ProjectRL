"""Clean up all remaining duplicate occurrences of removed code."""

with open('temp_check.py', 'r', encoding='utf-8', newline='') as f:
    src = f.read().replace('\r\n', '\n')

# ---- Fix 1 residual: second _compute_confidence with consistency_bonus ----
# Replace ALL instances of the consistency bonus block
import re
old_conf_block = (
    '        # Check consistency with previous rounds\n'
    '        prev_answers = [r.answer for r in self.history[-3:]]\n'
    '        if prev_answers and llm_response.answer in prev_answers:\n'
    '            # Consistent with previous answers -> boost confidence\n'
    '            consistency_count = prev_answers.count(llm_response.answer)\n'
    '            consistency_bonus = 0.1 * consistency_count / len(prev_answers)\n'
    '            return min(0.95, base_confidence + consistency_bonus)\n'
    '        elif prev_answers:\n'
    '            # Changed answer -> reduce confidence slightly\n'
    '            return max(0.1, base_confidence - 0.05)\n'
    '\n'
    '        return base_confidence'
)
new_conf_block = '        # Fix 1: Consistency bonus removed (see _compute_confidence for details)\n        return base_confidence'
count1 = src.count(old_conf_block)
src = src.replace(old_conf_block, new_conf_block)
print(f'Fix 1 residual: replaced {count1} occurrence(s) of consistency bonus block.')

# ---- Fix 2 residual: second compute_reward with shaping_reward ----
old_shaping = (
    '        # Shaping rewards (only at final step to avoid double-counting)\n'
    '        shaping_reward = 0.0\n'
    '        if is_final_step:\n'
    '            shaping_reward += confidence_improvement * 0.3\n'
    '            shaping_reward += answer_entropy_reduction * 0.2\n'
    '            shaping_reward += 0.05\n'
)
new_shaping = '        # Fix 2: Shaping rewards removed (see compute_reward docstring)\n'
count2 = src.count(old_shaping)
src = src.replace(old_shaping, new_shaping)
print(f'Fix 2 residual: replaced {count2} occurrence(s) of shaping block.')

# Also remove leftover shaping_reward from return statements
old_ret = '            task_rew\n            + shaping_reward\n'
new_ret = '            task_rew\n'
count2b = src.count(old_ret)
src = src.replace(old_ret, new_ret)
print(f'Fix 2b residual: replaced {count2b} shaping_reward in return statements.')

# ---- Fix 4 residual: second update_adaptive_prior method ----
# The comment "Fix 4: update_adaptive_prior() removed..." appears before
# "def update_policy" — means the actual method body is still there.
# Find and remove any remaining real method body.
F4D_OLD = (
    '    def update_adaptive_prior(self, W: np.ndarray):\n'
    '        """Tier4 #22: EMA update of sparsity prior from observed topology.\n'
    '        Called once per training epoch (not per round) to reduce non-stationarity.\n'
    '        """\n'
    '        observed_density = float(np.mean(W[~np.eye(W.shape[0], dtype=bool)]))\n'
    '        self.adaptive_prior = 0.9 * self.adaptive_prior + 0.1 * observed_density\n'
    '        self.adaptive_prior = float(np.clip(self.adaptive_prior, 0.05, 0.9))\n'
)
count4 = src.count(F4D_OLD)
src = src.replace(F4D_OLD, '    # Fix 4: update_adaptive_prior() removed. SPARSITY_PRIOR=0.1 used.\n')
print(f'Fix 4 residual: replaced {count4} remaining update_adaptive_prior method(s).')

# Verify
print()
for term in ['consistency_bonus', 'shaping_reward', 'update_adaptive_prior', 'self.adaptive_prior']:
    cnt = src.count(term)
    status = 'OK (0)' if cnt == 0 else f'REMAINING: {cnt}'
    print(f'  {term}: {status}')

# Count update_adaptive_prior comment refs (OK to keep)
comment_refs = src.count('update_adaptive_prior() removed')
print(f'  (comment refs to update_adaptive_prior: {comment_refs})')

with open('temp_check.py', 'w', encoding='utf-8', newline='') as f:
    f.write(src)
print('\nWritten.')
