"""Controller notebook entrypoints. Import regenerates the nine original walkthroughs."""
from build_notebooks import ROOT,BOOT,save,md,code
SETUP=BOOT+"\nSETTINGS=json.loads((ROOT/'controller_config.json').read_text())\nfrom controller_study import forecast, study_directory\nSTUDY_DIR,SPLIT_DIR=study_directory(CFG,SETTINGS)\nprint(json.dumps(forecast(CFG,SETTINGS),indent=2))\n"
save('09_Controller_Splits_Leakage_Pilot.ipynb',[
md('''# Controller retraining: real data, protected splits, and invariant pilot

The trained-policy extension retains SparseComm's routing, message budgets and communication/quality tradeoff. It is a **new one-step categorical contextual-bandit PPO implementation**, not a reproduced original continuous-action checkpoint. E1–E4 remain controlled interventions; E5 is a learned whole-policy comparison.

Default allocation: 300 optimization questions, 100 validation questions, 300 held-out development questions, and 12 pilot questions. Optimization/validation/pilot come from publisher-pinned HotpotQA train. Held-out questions come from fullwiki development and exclude the 100 known legacy Web evaluation IDs. Publisher parquet hashes authenticate all 90,447 training rows.

Pairwise IDs, normalized questions and candidate-document titles must be disjoint. Title disjointness changes representativeness and does not detect all paraphrases, title aliases or model pretraining contamination. It is a stronger split protection, not a promise that reviewers cannot raise concerns. The supplied training contexts and fullwiki development candidates have different retrieval distributions; report this limit.

Use the configured existing Azure gpt-oss-120b deployment. Shell exports are loaded in memory, and credentials are never serialized. The tokenizer is pinned to its official publisher commit. The Azure deployment alias has no independently established immutable snapshot: saved provider responses/model identifiers are the reproducibility record.'''),code(SETUP),
code('''manifest=json.loads((SPLIT_DIR/'manifest.json').read_text())
display({k:{'n':v['n'],'types':v['types']} for k,v in manifest['partitions'].items()})
display(manifest['leakage_checks'])
print('Known legacy Web IDs excluded:',len(manifest['known_legacy_eval_ids_excluded']))
print('Leakage limitations:',manifest['limitations'])'''),
md('''## Real invariant pilot

Check formats, source-citation validity, exact tokens, measured input/output distributions and API compatibility on these separate 12 questions before training. No fabricated rewards or alternate-model fallback is allowed. A parser/context/API error halts and saves its actual request or raw response when available. Do not choose methods by whether pilot answers are favorable. Authentication can be checked with `python machine_connection.py` without inference.'''),
code('''from controller_study import pilot
print(pilot(CFG,SETTINGS,execute=EXECUTE))
status=STUDY_DIR/'pilot_status.json'
print(json.loads(status.read_text()) if status.exists() else 'No completed pilot for this frozen study')''')])
save('10_Controller_Train_Validate_Select.ipynb',[
md('''# Train and select SparseComm without held-out tuning

The actor chooses one of 55 explicit actions: NoComm, or one of all 15 directed two-edge supports, two opposite cycles, or broadcast at quota 64/128/256. This makes the sampled action exactly the graph/quota executed. NoComm incurs zero additional calls and is a legitimate learned decision. Do not force nonzero edges or claim adaptation if the policy selects one action for every question.

Observable features are citation/evidence availability, source-message lengths, answer disagreement, lexical relevance and fixed role assignments. No gold correctness, answer labels, ID hashes or accuracy history is a controller input. Gold answer F1 is used only as a training reward after choosing and executing the action.

Reward: `(F1_after−F1_initial)/100 − .05×(payload/full_broadcast_payload) − .02×(distinct_receiver_calls/3)`. Report full measured inference costs separately; this engineering objective is not a dollar-cost formula or universally optimal weighting. A fan-in invokes one receiver, an out-star two, and a cycle/broadcast three; untouched initial responses must never be counted as new calls.

PPO uses autograd clipping, value regression, entropy, gradient-norm clipping and approximate-KL stopping. Three optimizer seeds share the same authentic frozen initial LLM responses, reducing waste. They are optimizer replicates, not independent backbones or additional independent questions.

After each collection epoch, choose maximum prespecified validation utility, with earlier-epoch tie breaking. The 100-question validation set selects both checkpoint and one of three declared static candidates. Held-out scores cannot affect optimizer updates, early stopping, penalty coefficients, prompts or checkpoint selection. Optimizer, actor/critic and Torch RNG state are saved for resume. A held-out evaluation gate closes further training on that frozen study.'''),code(SETUP),
code('''from controller_study import train
print(train(CFG,SETTINGS,execute=EXECUTE))
status=STUDY_DIR/'training_status.json'
if status.exists():
    print(json.loads(status.read_text()))
else:
    print('No real controller training completed. Unit tests are not learned checkpoints.')'''),
code('''for seed in SETTINGS['training_seeds']:
    path=STUDY_DIR/f'seed_{seed}'/'selection.json'
    print(seed,json.loads(path.read_text()) if path.exists() else 'NO VALIDATION-SELECTED CHECKPOINT')''')])
save('11_Controller_Heldout_Comparison.ipynb',[
md('''# Frozen learned policy: 300 held-out questions

Freeze all selected checkpoint hashes and the validation-selected static policy before accessing held-out model results. Apply the learned policy deterministically. Compare NoComm, self-update, selected static, quota-limited broadcast and per-seed random supports matched on learned edge count/quota. Random supports do not guarantee identical recipient participation or actual payload identities, so this is a whole-policy comparison; use E2 for destination-only claims.

Report F1, EM, evidence recall, actual payload, full provider input/output usage and actual recipient call counts. Report each optimizer seed and seed-averaged question-paired contrasts, including learned versus matched random, with uncertainty and exploratory Holm correction. Never count 300×3 as 900 independent questions.

Action/NoComm frequencies expose policy collapse. A null, negative or NoComm-dominant result is retained. Three hundred questions is a feasible target, not a guarantee of sufficient power or reviewer approval. Do not inspect held-out results and then retrain, tune or enlarge this test set.'''),code(SETUP),
code('''from controller_study import evaluate
print(evaluate(CFG,SETTINGS,execute=EXECUTE))'''),
code('''from controller_reporting import report_controller
from IPython.display import display,Image
status=STUDY_DIR/'E5_status.json'
if status.exists() and json.loads(status.read_text()).get('status')=='COMPLETE':
    summary,statistics=report_controller(STUDY_DIR)
    display(summary)
    print(json.dumps(statistics,indent=2))
    display(Image(filename=str(STUDY_DIR/'reports/E5_payload_frontier.png')))
else:
    print('NO COMPLETE HELD-OUT CONTROLLER RESULTS. No table or trained-policy claim is fabricated.')''')])
print('Built',len(list(ROOT.glob('*.ipynb'))),'notebooks including controller study')
