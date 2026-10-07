"""Fix calibrator None guard and s.correct -> s.is_correct."""

with open('temp_check.py', 'r', encoding='utf-8', newline='') as f:
    src = f.read().replace('\r\n', '\n')

def apply(src, old, new, tag):
    old = old.replace('\r\n', '\n')
    new = new.replace('\r\n', '\n')
    assert old in src, f'{tag}: target not found\n{repr(old[:120])}'
    result = src.replace(old, new)
    assert result != src, f'{tag}: no change made'
    return result

# Fix 1a: Guard calibrator.accumulate (classification)
F1A_OLD = (
    '                # Platt calibration: accumulate (conf, correct) during training (Improvement VI)\n'
    '                for aid, r in enumerate(agent_pool.agents):\n'
    '                    if r.history:\n'
    '                        last_resp = r.history[-1]\n'
    '                        agent_correct = evaluate_answer(last_resp.answer, sample["answer"], config.dataset)\n'
    '                        controller.calibrator.accumulate(aid, last_resp.confidence, agent_correct)\n'
)
F1A_NEW = (
    '                # Platt calibration: accumulate (conf, correct) during training (Improvement VI)\n'
    '                if controller.calibrator is not None:\n'
    '                    for aid, r in enumerate(agent_pool.agents):\n'
    '                        if r.history:\n'
    '                            last_resp = r.history[-1]\n'
    '                            agent_correct = evaluate_answer(last_resp.answer, sample["answer"], config.dataset)\n'
    '                            controller.calibrator.accumulate(aid, last_resp.confidence, agent_correct)\n'
)
src = apply(src, F1A_OLD, F1A_NEW, 'Fix 1a (accumulate guard)')
print('Fix 1a applied.')

# Fix 1b: Guard calibrator.accumulate_numeric (regression)
F1B_OLD = (
    '                            if y_pred is not None and y_true is not None:\n'
    '                                controller.calibrator.accumulate_numeric(aid, y_pred, y_true)\n'
)
F1B_NEW = (
    '                            if y_pred is not None and y_true is not None and controller.calibrator is not None:\n'
    '                                controller.calibrator.accumulate_numeric(aid, y_pred, y_true)\n'
)
src = apply(src, F1B_OLD, F1B_NEW, 'Fix 1b (accumulate_numeric guard)')
print('Fix 1b applied.')

# Fix 1c: Guard calibrator.fit()
F1C_OLD = (
    '            # Improvement VI: re-fit Platt calibrator after each training epoch\n'
    '            controller.calibrator.fit()\n'
)
F1C_NEW = (
    '            # Improvement VI: re-fit Platt calibrator after each training epoch\n'
    '            if controller.calibrator is not None:\n'
    '                controller.calibrator.fit()\n'
)
src = apply(src, F1C_OLD, F1C_NEW, 'Fix 1c (fit() guard)')
print('Fix 1c applied.')

# Fix 2: s.correct -> s.is_correct in OLS
F2_OLD = '        float(np.mean([s.correct for s in ablation_results[arm].sample_results]))\n'
F2_NEW = '        float(np.mean([s.is_correct for s in ablation_results[arm].sample_results]))\n'
src = apply(src, F2_OLD, F2_NEW, 'Fix 2 (s.is_correct)')
print('Fix 2 applied.')

with open('temp_check.py', 'w', encoding='utf-8', newline='') as f:
    f.write(src)
print('\nAll fixes written.')
