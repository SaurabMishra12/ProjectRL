#!/usr/bin/env python3
"""
SparseComm Architecture Diagram - Clean Version
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(40, 30))
ax.set_xlim(0, 40)
ax.set_ylim(0, 30)
ax.axis('off')
ax.set_facecolor('#0d1117')
fig.patch.set_facecolor('#0d1117')

C = {
    'config': '#58a6ff', 'llm': '#f0883e', 'agent': '#3fb950',
    'comm': '#bc8cff', 'ctrl': '#f778ba', 'vote': '#79c0ff',
    'data': '#d2a8ff', 'metric': '#ffa657', 'pipe': '#56d4dd',
    'text': '#c9d1d9', 'dim': '#8b949e', 'border': '#30363d', 'title': '#f0f6fc',
}

def box(x, y, w, h, label, color, fs=9, bold=False, sub=None):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                       facecolor=color + '18', edgecolor=color, linewidth=2)
    ax.add_patch(r)
    wt = 'bold' if bold else 'normal'
    ax.text(x+w/2, y+h/2+(0.15 if sub else 0), label, ha='center', va='center',
            fontsize=fs, color=color, fontweight=wt, family='monospace')
    if sub:
        ax.text(x+w/2, y+h/2-0.22, sub, ha='center', va='center',
                fontsize=6.5, color=C['dim'], family='monospace')

def txt(x, y, s, color=C['text'], fs=7, ha='center', family='monospace', style='normal'):
    ax.text(x, y, s, ha=ha, va='center', fontsize=fs, color=color,
            family=family, style=style)

def sec(x, y, label, color):
    ax.text(x, y, label, ha='left', va='center', fontsize=13,
            color=color, fontweight='bold', family='monospace')
    ax.plot([x, x+len(label)*0.13], [y-0.2, y-0.2], color=color, lw=0.8, alpha=0.4)

def arrow(x1, y1, x2, y2, color=C['border'], lw=1.0, label=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        txt(mx, my+0.18, label, C['dim'], fs=5.5)

# ══════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════
ax.text(20, 29.5, 'SparseComm — System Architecture', ha='center', va='center',
        fontsize=24, color=C['title'], fontweight='bold', family='monospace')
txt(20, 29.05, 'Sparse Semantic Communication for Multi-Agent Reasoning', C['dim'], 10)
ax.plot([5, 35], [28.8, 28.8], color=C['border'], lw=0.5)

# ══════════════════════════════════════════════════════════════
# ROW 1: CONFIG + DATA
# ══════════════════════════════════════════════════════════════
sec(0.5, 28.2, '① CONFIGURATION & DATA', C['config'])

box(0.5, 26.3, 4.0, 1.6, 'Config', C['config'], 11, True, 'All hyperparameters')
txt(2.5, 27.15, 'seed, n_agents, n_rounds', C['dim'], 6.5)
txt(2.5, 26.85, 'lambda_info, lambda_kl, gamma_loop', C['dim'], 6.5)
txt(2.5, 26.55, 'controller_hidden_dim, lr, epochs', C['dim'], 6.5)

box(5.0, 26.3, 3.0, 1.6, 'Config Presets', C['config'], 9, False, 'Per-dataset defaults')
txt(6.5, 27.15, 'get_gsm8k_config()', C['config'], 7)
txt(6.5, 26.85, 'get_hotpotqa_config()', C['config'], 7)
txt(6.5, 26.55, 'get_aqua_config()', C['config'], 7)

box(8.5, 26.3, 4.5, 1.6, 'Dataset Loaders', C['data'], 10, True, 'Strict train/test split')
txt(10.75, 27.15, 'load_gsm8k()  load_hotpotqa()  load_aqua()', C['text'], 6.5)
txt(10.75, 26.85, 'load_truthfulqa()  load_mbpp()', C['text'], 6.5)
txt(10.75, 26.55, 'load_dataset_by_name() → dispatcher', C['dim'], 6.5)

arrow(4.5, 27.1, 5.0, 27.1, C['config'], 0.8)
arrow(8.0, 27.1, 8.5, 27.1, C['config'], 0.8)

# ══════════════════════════════════════════════════════════════
# ROW 2: LLM BACKENDS
# ══════════════════════════════════════════════════════════════
sec(0.5, 25.7, '② LLM BACKENDS (API / Local)', C['llm'])

box(0.5, 23.7, 3.8, 1.7, 'LLMBackend (ABC)', C['llm'], 10, True, 'Abstract interface')
txt(2.4, 24.75, 'generate(prompt, question, context)', C['text'], 6.5)
txt(2.4, 24.45, 'compress_message(message, mode)', C['text'], 6.5)
txt(2.4, 24.15, '_extract_answer(text, dataset_type)', C['dim'], 6.5)
txt(2.4, 23.85, '_estimate_confidence(text)', C['dim'], 6.5)

box(5.0, 23.7, 3.5, 1.7, 'NvidiaDeepSeekBackend', C['llm'], 9, False, 'integrate.api.nvidia.com')
txt(6.75, 24.7, 'temperature=0.0', C['dim'], 6.5)
txt(6.75, 24.4, 'max_retries=15, backoff', C['dim'], 6.5)
txt(6.75, 24.1, 'Prompt cache (MD5 key)', C['dim'], 6.5)

box(9.0, 23.7, 3.5, 1.7, 'AzureOpenAIBackend', C['llm'], 9, False, 'Azure OpenAI endpoint')
txt(10.75, 24.7, 'api_version=2024-02-15', C['dim'], 6.5)
txt(10.75, 24.4, 'max_retries=5, httpx timeout', C['dim'], 6.5)
txt(10.75, 24.1, 'Prompt cache (same key)', C['dim'], 6.5)

box(13.0, 23.7, 3.5, 1.7, 'HuggingFaceLocalBackend', C['llm'], 9, False, 'Local model (Mistral-7B)')
txt(14.75, 24.7, 'device_map="auto"', C['dim'], 6.5)
txt(14.75, 24.4, 'torch.float16, do_sample=False', C['dim'], 6.5)
txt(14.75, 24.1, 'pipeline("text-generation")', C['dim'], 6.5)

arrow(4.3, 24.55, 5.0, 24.55, C['llm'], 0.8)
arrow(8.5, 24.55, 9.0, 24.55, C['llm'], 0.8)
arrow(12.5, 24.55, 13.0, 24.55, C['llm'], 0.8)

txt(8.5, 23.4, '_PROMPT_CACHE: Dict[str, LLMResponse] — module-level, caches identical API calls', C['dim'], 6.5)

# ══════════════════════════════════════════════════════════════
# ROW 3: AGENTS
# ══════════════════════════════════════════════════════════════
sec(0.5, 23.1, '③ MULTI-AGENT SYSTEM', C['agent'])

box(0.5, 21.0, 4.0, 1.8, 'Agent', C['agent'], 11, True, 'Single LLM reasoning agent')
txt(2.5, 22.25, 'solve(question, context, dataset_type, round_num)', C['text'], 6.5)
txt(2.5, 21.95, '_build_prompt(question, context, dataset_type)', C['dim'], 6.5)
txt(2.5, 21.65, '_compute_confidence(llm_response)', C['dim'], 6.5)
txt(2.5, 21.35, 'history: List[AgentResponse]', C['dim'], 6.5)
txt(2.5, 21.05, 'reasoning_style: analytical|step_by_step|creative', C['dim'], 6.5)

box(5.0, 21.0, 4.0, 1.8, 'AgentPool', C['agent'], 11, True, 'Manages N parallel agents')
txt(7.0, 22.25, 'initial_solve(question) → Round 0', C['text'], 6.5)
txt(7.0, 21.95, 'communicate_and_solve(question, messages)', C['text'], 6.5)
txt(7.0, 21.65, 'reset_all() — clear history for new sample', C['dim'], 6.5)
txt(7.0, 21.35, 'ThreadPoolExecutor(n_agents)', C['dim'], 6.5)
txt(7.0, 21.05, 'get_confidence_vector() → List[float]', C['dim'], 6.5)

arrow(4.5, 21.9, 5.0, 21.9, C['agent'], 1.0)

# Agent ↔ LLM arrow
arrow(2.4, 23.7, 2.4, 22.8, C['agent'], 0.8, 'calls')

# ══════════════════════════════════════════════════════════════
# ROW 4: COMMUNICATION GRAPH
# ══════════════════════════════════════════════════════════════
sec(0.5, 20.4, '④ COMMUNICATION GRAPH (Bandwidth-Aware)', C['comm'])

box(0.5, 17.8, 5.0, 2.3, 'CommunicationGraph', C['comm'], 11, True, 'W[i,j] edge weights')
txt(3.0, 19.7, 'transmit(sender, receiver, message, round)', C['text'], 6.5)
txt(3.0, 19.4, 'get_messages_for_round(responses, round)', C['text'], 6.5)
txt(3.0, 19.1, 'set_weights(W, Q)', C['text'], 6.5)
txt(3.0, 18.8, 'count_cycles(threshold=0.5) → trace(A³)/6', C['dim'], 6.5)
txt(3.0, 18.5, 'get_compression_ratio()', C['dim'], 6.5)
txt(3.0, 18.2, 'get_avg_semantic_similarity()', C['dim'], 6.5)
txt(3.0, 17.9, 'reset() — clear logs for new sample', C['dim'], 6.5)

box(6.0, 18.8, 3.5, 1.6, 'SemanticBandwidthMode', C['comm'], 9, True)
txt(7.75, 20.15, 'w < 0.05 → BLOCK (0 tokens)', '#ff7b72', 7)
txt(7.75, 19.85, '0.05 ≤ w < 0.95 → PARTIAL', '#d29922', 7)
txt(7.75, 19.55, '  (semantic filter)', C['dim'], 6.5)
txt(7.75, 19.25, 'w ≥ 0.95 → FULL (pass-through)', '#3fb950', 7)

box(6.0, 17.8, 3.5, 0.8, 'TransmissionRecord', C['comm'], 8, False, 'Per-edge log entry')

# Semantic Bottleneck
box(10.0, 17.8, 6.5, 2.7, 'Semantic Bottleneck Pipeline', C['comm'], 10, True, 'Latent Semantic Communication')
txt(13.25, 20.15, 'Sentences → SentenceTransformer Embeddings', C['text'], 7)
txt(13.25, 19.85, '→ Attention Pooling (centroid, τ = 1-w_ij)', C['text'], 7)
txt(13.25, 19.55, '→ Compressed latent vector z', C['text'], 7)
txt(13.25, 19.25, '→ Bandwidth Gate: z_gated = z × w_ij', C['text'], 7)
txt(13.25, 18.95, '→ Top-k Sentence Retrieval (k = round(w × n))', C['text'], 7)
txt(13.25, 18.65, '→ Order-preserved filtered message', C['text'], 7)
txt(13.25, 18.35, 'Fallback: TF-IDF centroid → first-k words', C['dim'], 6.5)
txt(13.25, 18.05, 'Semantic sim: cosine(embed_orig, embed_filtered)', C['dim'], 6.5)

arrow(5.5, 18.95, 6.0, 19.6, C['comm'], 0.8)
arrow(9.5, 19.6, 10.0, 19.6, C['comm'], 0.8, 'filters')

# Agent → CommGraph
arrow(3.0, 21.0, 3.0, 20.1, C['comm'], 0.8, 'reasoning texts')

# ══════════════════════════════════════════════════════════════
# ROW 5: PPO CONTROLLER (THE CORE)
# ══════════════════════════════════════════════════════════════
sec(0.5, 17.2, '⑤ PPO CONTROLLER (Information Bottleneck)', C['ctrl'])

box(0.5, 14.0, 5.5, 2.9, 'IBController', C['ctrl'], 12, True, 'PPO policy → W, Q, halt_decision')
txt(3.25, 16.45, 'get_weights(reasoning, answers, confs, prev_W, round)', C['text'], 7)
txt(3.25, 16.15, '  → Actor forward → mean + log_std', C['dim'], 6.5)
txt(3.25, 15.85, '  → split_action → H_src, H_dst → W = sigmoid(H_src @ H_dst.T + 0.5)', C['dim'], 6.5)
txt(3.25, 15.55, 'compute_reward(task, tokens, cycles, W)', C['text'], 7)
txt(3.25, 15.25, '  = 1.5·task - 0.5 - λ_info·cost - λ_kl·KL - γ·loops', C['ctrl'], 7)
txt(3.25, 14.95, 'update_policy() → GAE + PPO clip + AdamW', C['text'], 7)
txt(3.25, 14.65, 'store_transition(meta, reward, done)', C['dim'], 6.5)
txt(3.25, 14.35, 'save_model() / load_model() → .npz + calibrator.pkl', C['dim'], 6.5)

# Actor
box(6.5, 15.2, 3.0, 1.7, 'Actor (MLP)', C['ctrl'], 9, True, 'π(a|s) → continuous action')
txt(8.0, 16.45, 'Input: state features (d_model + 12)', C['dim'], 6)
txt(8.0, 16.15, '2 hidden layers + ReLU', C['dim'], 6)
txt(8.0, 15.85, 'Output: H_src, H_dst, Q, halt_logit', C['dim'], 6)
txt(8.0, 15.55, 'Learnable σ per dimension', C['dim'], 6)

# Critic
box(6.5, 13.5, 3.0, 1.5, 'Critic (MLP)', C['ctrl'], 9, True, 'V(s) → scalar value')
txt(8.0, 14.65, 'Same architecture as Actor', C['dim'], 6)
txt(8.0, 14.35, 'Output: single scalar value', C['dim'], 6)
txt(8.0, 14.05, 'Fits GAE returns target', C['dim'], 6)

# State Encoder
box(10.0, 15.5, 3.5, 1.4, 'SemanticStateEncoder', C['ctrl'], 9, False, '5 info-theoretic features')
txt(11.75, 16.65, '① Shannon Entropy (disagreement)', C['text'], 6.5)
txt(11.75, 16.35, '② Jaccard Similarity (overlap)', C['text'], 6.5)
txt(11.75, 16.05, '③ Answer Diversity (unique/total)', C['text'], 6.5)
txt(11.75, 15.75, '④ Length Variance (confidence spread)', C['text'], 6.5)
txt(11.75, 15.45, '⑤ Consensus Strength (majority fraction)', C['text'], 6.5)

# Calibrator
box(10.0, 14.0, 3.5, 1.3, 'PlattCalibrator (Isotonic)', C['ctrl'], 9, False, 'Confidence calibration')
txt(11.75, 15.05, 'IsotonicRegression(sklearn)', C['dim'], 6.5)
txt(11.75, 14.75, 'P(correct | self-reported confidence)', C['dim'], 6.5)
txt(11.75, 14.45, 'Per-agent historical accuracy tracking', C['dim'], 6.5)
txt(11.75, 14.15, 'Empirical variance for precision weights', C['dim'], 6.5)

# SPRT
box(14.0, 15.5, 3.5, 1.4, 'SPRT Halting', C['ctrl'], 9, False, 'Sequential Probability Ratio Test')
txt(15.75, 16.65, 'H₀: p=0.5 (random guess)', C['text'], 6.5)
txt(15.75, 16.35, 'H₁: p=p_maj (calibrated majority)', C['text'], 6.5)
txt(15.75, 16.05, 'LLR > τ=2.944 → halt early', C['text'], 6.5)
txt(15.75, 15.75, 'Kish ESS: corrects for correlated votes', C['dim'], 6.5)

# Arrows
arrow(6.0, 15.9, 6.5, 15.9, C['ctrl'], 0.8, 'forward')
arrow(6.0, 14.75, 6.5, 14.25, C['ctrl'], 0.8)
arrow(10.0, 16.2, 9.5, 15.9, C['ctrl'], 0.8, 'state')
arrow(10.0, 14.65, 9.5, 15.0, C['ctrl'], 0.8)
arrow(3.25, 14.0, 3.0, 17.8, C['ctrl'], 1.2, 'W, Q')

# ══════════════════════════════════════════════════════════════
# ROW 6: VOTING + METRICS
# ══════════════════════════════════════════════════════════════
sec(0.5, 13.3, '⑥ VOTING & EVALUATION', C['vote'])

box(0.5, 11.2, 4.5, 1.8, 'Voting Functions', C['vote'], 10, True, 'Answer aggregation')
txt(2.75, 12.65, 'majority_vote() — simple mode', C['text'], 7)
txt(2.75, 12.35, 'weighted_majority_vote() — Bayesian multiplicative', C['text'], 7)
txt(2.75, 12.05, '  + spectral centrality + stability + calibration', C['dim'], 6)
txt(2.75, 11.75, 'numeric_median_vote() — L1-optimal for numbers', C['text'], 7)
txt(2.75, 11.45, 'spectral_vote_weights() — in-degree centrality', C['dim'], 6.5)
txt(2.75, 11.15, 'compute_stability_weights() — flip-flop penalty', C['dim'], 6.5)

box(5.5, 11.2, 4.0, 1.8, 'Evaluation Metrics', C['metric'], 10, True, 'Accuracy & efficiency')
txt(7.5, 12.65, 'evaluate_answer(pred, gold, dataset)', C['text'], 7)
txt(7.5, 12.35, '  GSM8K: numeric comparison (|Δ| < 1e-3)', C['dim'], 6.5)
txt(7.5, 12.05, '  HotpotQA: Exact Match (normalized)', C['dim'], 6.5)
txt(7.5, 11.75, '  AQuA/TruthfulQA: string match (cleaned)', C['dim'], 6.5)
txt(7.5, 11.45, 'hotpotqa_f1_score() — auxiliary metric', C['dim'], 6.5)
txt(7.5, 11.15, 'count_tokens() via tiktoken (cl100k_base)', C['dim'], 6.5)

box(10.0, 11.2, 4.0, 1.8, 'Result Aggregation', C['metric'], 10, True, 'Experiment output')
txt(12.0, 12.65, 'compute_accuracy() — % correct', C['text'], 7)
txt(12.0, 12.35, 'compute_token_cost() — avg tokens/sample', C['text'], 7)
txt(12.0, 12.05, 'compute_communication_density() — mean W', C['dim'], 6.5)
txt(12.0, 11.75, 'compute_error_propagation_rate()', C['dim'], 6.5)
txt(12.0, 11.45, 'aggregate_metrics() → dict', C['dim'], 6.5)
txt(12.0, 11.15, 'format_results_table() → markdown', C['dim'], 6.5)

# ══════════════════════════════════════════════════════════════
# ROW 7: DATA STRUCTURES
# ══════════════════════════════════════════════════════════════
sec(0.5, 10.6, '⑦ DATA STRUCTURES', C['dim'])

box(0.5, 9.2, 3.0, 1.1, 'AgentResponse', C['dim'], 8, False, 'Per-agent, per-round')
txt(2.0, 10.05, 'agent_id, answer, confidence', C['dim'], 6)
txt(2.0, 9.75, 'reasoning_text, token_count', C['dim'], 6)

box(4.0, 9.2, 3.0, 1.1, 'SampleResult', C['dim'], 8, False, 'Per-question result')
txt(5.5, 10.05, 'question, gold_answer, final_answer', C['dim'], 6)
txt(5.5, 9.75, 'is_correct, f1_score, compression_ratio', C['dim'], 6)

box(7.5, 9.2, 3.0, 1.1, 'ExperimentResults', C['dim'], 8, False, 'Full experiment')
txt(9.0, 10.05, 'method_name, sample_results[]', C['dim'], 6)
txt(9.0, 9.75, 'training_rewards, training_losses, final_W', C['dim'], 6)

box(11.0, 9.2, 3.0, 1.1, 'LLMResponse', C['dim'], 8, False, 'LLM output')
txt(12.5, 10.05, 'text, answer, token_count', C['dim'], 6)
txt(12.5, 9.75, 'confidence (0-1)', C['dim'], 6)

# ══════════════════════════════════════════════════════════════
# ROW 8: EXECUTION PIPELINE (full width)
# ══════════════════════════════════════════════════════════════
sec(0.5, 8.6, '⑧ EXECUTION PIPELINE', C['pipe'])

# Pipeline boxes
box(0.5, 6.5, 5.0, 1.8, 'run_single_sample()', C['pipe'], 10, True, 'One question → one result')
txt(3.0, 7.95, '1. Round 0: agents.solve() independently', C['text'], 7)
txt(3.0, 7.65, '2. Check unanimous → early exit (synthetic transition)', C['text'], 7)
txt(3.0, 7.35, '3. For t=1..T: controller → W → filter → re-solve', C['text'], 7)
txt(3.0, 7.05, '4. SPRT halt check (Kish ESS correction)', C['text'], 7)
txt(3.0, 6.75, '5. Final: weighted_majority_vote() → answer', C['text'], 7)

box(6.0, 6.5, 5.0, 1.8, 'run_experiment()', C['pipe'], 10, True, 'Full train+eval pipeline')
txt(8.5, 7.95, 'Training: N samples × E epochs', C['text'], 7)
txt(8.5, 7.65, 'update_policy() every 16 samples (GAE + PPO)', C['text'], 7)
txt(8.5, 7.35, 'PlattCalibrator.fit() after each epoch', C['dim'], 6.5)
txt(8.5, 7.05, 'Evaluation: separate test set (no training)', C['text'], 7)
txt(8.5, 6.75, 'Returns ExperimentResults', C['dim'], 6.5)

box(11.5, 6.5, 4.0, 1.8, 'Baselines', C['pipe'], 10, True, 'Comparison methods')
txt(13.5, 7.95, 'run_full_communication() — W=1.0', C['text'], 7)
txt(13.5, 7.65, 'run_no_communication() — W=0.0', C['text'], 7)
txt(13.5, 7.35, 'run_experiment_with_random_weights()', C['text'], 7)
txt(13.5, 7.05, 'FrozenWeightController (ablation)', C['dim'], 6.5)

arrow(5.5, 7.4, 6.0, 7.4, C['pipe'], 1.0)
arrow(11.0, 7.4, 11.5, 7.4, C['pipe'], 1.0)

# ══════════════════════════════════════════════════════════════
# ROW 9: EXPERIMENT PHASES
# ══════════════════════════════════════════════════════════════
sec(0.5, 6.0, '⑨ EXPERIMENT PHASES', C['data'])

phases = [
    ('Phase 1: Pre-Training', 'N=100 per dataset', '3 datasets × 100 samples\ntrain_controller=True', 0),
    ('Phase 2: Transfer Eval', 'Controller generalizes', 'Freeze controller, eval each dataset\ntrain=False, evaluate=True', 5.0),
    ('Phase 3: Baselines', 'NoComm / Debate / Random', 'Compare sparse vs full vs none\nFair vote (same calibrator)', 10.0),
    ('Phase 4: Ablation', '2×2 factorial', 'λ_info × λ_kl grid search\nOLS regression on accuracy', 15.0),
]
for name, sub, desc, xoff in phases:
    box(0.5+xoff, 4.2, 4.5, 1.5, name, C['data'], 9, True, sub)
    for i, line in enumerate(desc.split('\n')):
        txt(2.75+xoff, 4.85-i*0.3, line, C['dim'], 6.5)

# Topology benchmark
box(20.5, 4.2, 5.5, 1.5, 'Topology Benchmark', C['comm'], 9, True, 'Static vs Learned')
txt(23.25, 5.35, 'Fully Connected | Ring | Star', C['text'], 6.5)
txt(23.25, 5.05, 'Random | Small World | Scale Free', C['text'], 6.5)
txt(23.25, 4.75, 'Learned Topology (trained controller)', C['text'], 6.5)
txt(23.25, 4.45, 'FrozenWeightController for each', C['dim'], 6.5)

# ══════════════════════════════════════════════════════════════
# RIGHT SIDE: KEY DATA FLOW
# ══════════════════════════════════════════════════════════════
sec(18.5, 28.2, 'KEY DATA FLOW', C['pipe'])

flow = [
    ('1', 'Config → loads dataset (strict train/test split)', C['config']),
    ('2', 'AgentPool.initial_solve() → Round 0 independent answers', C['agent']),
    ('3', 'SemanticStateEncoder.encode_state() → 5 features', C['ctrl']),
    ('4', 'IBController.get_weights() → W, Q, halt_decision', C['ctrl']),
    ('', '  Actor forward → split_action → sigmoid(H_src @ H_dst.T)', C['dim']),
    ('5', 'CommunicationGraph.set_weights(W, Q)', C['comm']),
    ('6', 'CommunicationGraph.transmit() → semantic bottleneck', C['comm']),
    ('', '  Sentence embeddings → attention pool → bandwidth gate → top-k', C['dim']),
    ('7', 'AgentPool.communicate_and_solve() → refined answers', C['agent']),
    ('8', 'Repeat 3-7 for T rounds (or halt early via SPRT)', C['pipe']),
    ('9', 'weighted_majority_vote() → final answer', C['vote']),
    ('10', 'compute_reward() → PPO update (GAE + clipped surrogate)', C['ctrl']),
    ('11', 'PlattCalibrator.fit() → recalibrate confidences', C['ctrl']),
]

for i, (num, line, color) in enumerate(flow):
    y = 27.7 - i * 0.42
    prefix = f'{num}. ' if num else '   '
    txt(18.5, y, prefix + line, color, 7, ha='left')

# ══════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════
legend = [
    (C['config'], '① Configuration & Data Loading'),
    (C['llm'],    '② LLM Backends (API / Local)'),
    (C['agent'],  '③ Multi-Agent System'),
    (C['comm'],   '④ Communication Graph & Semantic Bottleneck'),
    (C['ctrl'],   '⑤ PPO Controller (Information Bottleneck)'),
    (C['vote'],   '⑥ Voting & Answer Aggregation'),
    (C['metric'], '⑦ Evaluation Metrics & Data Structures'),
    (C['pipe'],   '⑧ Execution Pipeline'),
    (C['data'],   '⑨ Experiment Phases'),
]

for i, (color, label) in enumerate(legend):
    x, y = 32, 28.0 - i * 0.55
    r = FancyBboxPatch((x, y-0.18), 0.35, 0.35, boxstyle="round,pad=0.05",
                       facecolor=color+'30', edgecolor=color, linewidth=1.2)
    ax.add_patch(r)
    txt(x+0.55, y, label, C['text'], 7, ha='left')

# Reward formula box
box(18.5, 2.5, 18.0, 1.2, '', C['ctrl'], 8)
ax.text(27.5, 3.4, 'Reward Function', ha='center', va='center', fontsize=12,
        color=C['ctrl'], fontweight='bold', family='monospace')
txt(27.5, 3.05, 'R = 1.5·task_reward - 0.5 - λ_info·token_cost - λ_kl·KL(sparsity) - γ·cycle_penalty - halt_penalty', C['text'], 7.5)
txt(27.5, 2.7, 'GAE Advantage: δₜ = Rₜ + γ·V(sₜ₊₁) - V(sₜ),  Â = Σ (γλ)ˡδₜ₊ₗ,  PPO clip: min(r·Â, clip(r,1±ε)·Â)', C['dim'], 7)

plt.tight_layout(pad=0.3)
plt.savefig('/home/saurab/Documents/projectRL/sparsecomm_architecture.png',
            dpi=150, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
plt.savefig('/home/saurab/Documents/projectRL/sparsecomm_architecture.pdf',
            bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
print("Saved: sparsecomm_architecture.png and .pdf")
