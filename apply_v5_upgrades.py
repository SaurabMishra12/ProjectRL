import py_compile, nbformat, sys

with open('temp_check.py', 'r', encoding='utf-8') as f:
    src = f.read()

fixes = []
def apply(label, old, new):
    global src
    if old in src:
        src = src.replace(old, new, 1)
        fixes.append(f"[OK] {label}")
    else:
        fixes.append(f"[FAIL] {label}")

# 1. Isotonic Calibrator
old_calibrator = '''class PlattCalibrator:
    """Voting improvement VI: Platt scaling for LLM confidence calibration.
    Fitted on training samples. Falls back to raw confidence if not fitted.
    """
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self._fitted = False
        self._train_confs: List[float] = []
        self._train_labels: List[bool] = []

    def accumulate(self, conf: float, is_correct: bool):
        self._train_confs.append(conf)
        self._train_labels.append(is_correct)

    def fit(self):
        if len(self._train_confs) < 10:
            return  # Not enough data yet
        X = np.array(self._train_confs)
        y = np.array(self._train_labels, dtype=float)
        for _ in range(50):
            z = self.a * X + self.b
            p = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            grad_a = np.mean((p - y) * X)
            grad_b = np.mean(p - y)
            h_a = np.mean(p * (1 - p) * X**2) + 1e-6
            h_b = np.mean(p * (1 - p)) + 1e-6
            self.a -= grad_a / h_a * 0.1
            self.b -= grad_b / h_b * 0.1
        self._fitted = True

    def calibrate(self, raw_conf: float) -> float:
        if not self._fitted:
            return raw_conf
        z = self.a * raw_conf + self.b
        return float(1 / (1 + np.exp(-np.clip(z, -500, 500))))'''

new_calibrator = '''from sklearn.isotonic import IsotonicRegression

class PlattCalibrator:
    """V5 Priority #2: Isotonic Calibration (renamed class internally to avoid breaking save_model signatures).
    State-of-the-art non-parametric calibration wrapping sklearn IsotonicRegression.
    """
    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds='clip')
        self._fitted = False
        self._train_confs: List[float] = []
        self._train_labels: List[bool] = []
        # Dummy a/b for backward compatibility with save/load format
        self.a = 1.0; self.b = 0.0

    def accumulate(self, conf: float, is_correct: bool):
        self._train_confs.append(conf)
        self._train_labels.append(is_correct)

    def fit(self):
        if len(self._train_confs) < 10:
            return
        X = np.array(self._train_confs)
        y = np.array(self._train_labels, dtype=float)
        try:
            self.ir.fit(X, y)
            self._fitted = True
        except Exception:
            pass

    def calibrate(self, raw_conf: float) -> float:
        if not self._fitted:
            return raw_conf
        try:
            return float(self.ir.predict([raw_conf])[0])
        except Exception:
            return raw_conf'''
apply("Priority #2: Isotonic Calibration", old_calibrator, new_calibrator)

# 2. Weighted Communication Density
old_density = '''    def get_communication_density(self) -> float:
        """Fraction of possible off-diagonal edges with weight >= BLOCK threshold."""
        n_possible = self.n_agents * (self.n_agents - 1)  # Off-diagonal only
        if n_possible == 0:
            return 0.0
        # Mask diagonal to avoid counting self-loops
        W_off = self.W.copy()
        np.fill_diagonal(W_off, 0.0)
        active_edges = np.sum(W_off >= 0.25)
        return float(active_edges / n_possible)'''

new_density = '''    def get_communication_density(self) -> float:
        """V5 Priority #4: L1-Norm Weighted Communication Density."""
        n_possible = self.n_agents * (self.n_agents - 1)
        if n_possible == 0:
            return 0.0
        W_off = self.W.copy()
        np.fill_diagonal(W_off, 0.0)
        return float(np.sum(W_off) / n_possible)'''
apply("Priority #4: Weighted Density", old_density, new_density)

# 3. Bayesian Aggregation
old_vote = '''        return max(votes.items(), key=lambda x: x[1])[0]

    # Weighted categorical vote
    votes = defaultdict(float)'''

new_vote = '''        return max(votes.items(), key=lambda x: x[1])[0]

    # V5 Priority #5: Bayesian Aggregation (MAP estimate under Gaussian noise) for numeric answers
    # Only applies if all parsed responses are numeric (e.g. math/gsm8k)
    import re as _re
    def _extract_num(s):
        m = _re.findall(r'-?\d+\.?\d*', str(s))
        return float(m[-1]) if m else None
    
    if dataset_type in ['gsm8k', 'math']:
        nums = []
        variances = []
        for r in responses:
            n = _extract_num(r.answer)
            if n is not None:
                nums.append(n)
                # Calibrated variance = inversely proportional to confidence
                conf = calibrator.calibrate(r.confidence) if calibrator else r.confidence
                var = (1.0 - conf)**2 + 1e-4
                variances.append(var)
        if len(nums) == len(responses):
            # All responses are numeric, apply MAP estimate
            nums = np.array(nums)
            variances = np.array(variances)
            precisions = 1.0 / variances
            map_est = np.sum(nums * precisions) / np.sum(precisions)
            # If original answers were integers, return integer string, else float
            if all(float(n).is_integer() for n in nums):
                return str(int(round(map_est)))
            return str(round(map_est, 4))

    # Weighted categorical vote
    votes = defaultdict(float)'''
apply("Priority #5: Bayesian Vote", old_vote, new_vote)

# 4. InfoNCE Retention
old_info_ret = '''    def get_avg_information_retention(self) -> float:
        active = [r for r in self.transmission_log if r.original_tokens > 0]
        if not active: return 1.0
        return float(np.mean([r.information_retention for r in active]))'''

new_info_ret = '''    def get_avg_information_retention(self) -> float:
        """V5 Priority #3: InfoNCE Mutual Information Lower Bound"""
        active = [r for r in self.transmission_log if r.original_tokens > 0]
        if not active or len(active) < 2: return 1.0
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        try:
            orig_msgs = [r.original_msg for r in active]
            comp_msgs = [r.filtered_msg for r in active]
            vectorizer = TfidfVectorizer()
            vectorizer.fit(orig_msgs + comp_msgs)
            X = vectorizer.transform(orig_msgs)
            Z = vectorizer.transform(comp_msgs)
            
            sim_matrix = cosine_similarity(X, Z)  # (N, N)
            tau = 0.1
            sim_matrix /= tau
            
            # log sum exp trick
            max_sim = np.max(sim_matrix, axis=1, keepdims=True)
            sum_exp = np.sum(np.exp(sim_matrix - max_sim), axis=1, keepdims=True)
            log_sum_exp = max_sim + np.log(sum_exp)
            
            # Positive pairs on diagonal
            pos_sim = np.diag(sim_matrix)
            
            # L_InfoNCE
            l_infonce = np.mean(pos_sim - log_sum_exp.flatten())
            # Convert to retention metric [0, 1]
            max_mi = np.log(len(orig_msgs))
            retention = min(1.0, max(0.0, (l_infonce + max_mi) / max_mi))
            return float(retention)
        except Exception:
            # Fallback if TF-IDF fails (e.g. empty vocab)
            return float(np.mean([r.information_retention for r in active]))'''
apply("Priority #3: InfoNCE Retention", old_info_ret, new_info_ret)

# 5. Continuous Latent Bottleneck / Graph Extractor
old_transmit = '''        if mode == SemanticBandwidthMode.BLOCK or w_ij < 1e-3:
            filtered_msg = ""
            transmitted_tokens = 0
            sem_sim = 0.0
            info_ret = 0.0
        elif mode == SemanticBandwidthMode.FULL or w_ij > 0.99:
            # Send the complete structured state instead of raw text
            struct = self.parser.parse(message, dataset_type=self.dataset_type)  # Bug A4 fix
            filtered_msg = struct.to_formatted_string()
            transmitted_tokens = count_tokens(filtered_msg)
            sem_sim = 1.0
            info_ret = 1.0
        else:
            # TRUE SEMANTIC COMPRESSION (Structured Schema Routing) — Tier4 #20
            # Fix6: pre-extract final_answer from raw message text so it survives
            # SUMMARIZE/FILTER mode (which clears claims/evidence/assumptions).
            # Fix6 (improved): capture multi-line answers — stop at next blank line or EOS.
            # re.DOTALL lets . match newlines; the lazy stop at \\n\\n|$ handles paragraphs.
            _ans_match = re.search(
                r'(?:Answer:|ANSWER:|The answer is:?)\\s*(.+?)(?:\\n\\n|\\Z)',
                message,
                re.IGNORECASE | re.DOTALL,
            )
            if not _ans_match:
                # Fallback: last non-empty line often contains the answer
                _last_line = next((l.strip() for l in reversed(message.splitlines()) if l.strip()), '')
                _pre_answer = _last_line
            else:
                _pre_answer = _ans_match.group(1).strip()
            struct = self.parser.parse(message, dataset_type=self.dataset_type)  # Bug A4 fix
            if not struct.final_answer and _pre_answer:
                struct.final_answer = _pre_answer  # Restore extracted answer
            
            # Filter the structure based on bandwidth mode
            if mode == SemanticBandwidthMode.SUMMARIZE: # w in [0.25, 0.5)
                # Severely restricted: Only send Final Answer and Intermediate Conclusions
                struct.claims = []
                struct.evidence = []
                struct.assumptions = []
            elif mode == SemanticBandwidthMode.FILTER: # w in [0.5, 0.65)
                # Moderate: Send Final Answer, Conclusions, and Evidence
                struct.claims = []
                struct.assumptions = []
            elif mode == SemanticBandwidthMode.CONTRAST:  # Tier4 #21: w in [0.65, 0.75)
                # Send only content NOT already in receiver's last response
                # This computes a semantic diff: eliminate shared sentences
                struct.claims = []
                struct.assumptions = []
                # Advanced contrast logic: remove conclusions already known by receiver
                if receiver_id in self.agent_histories and self.agent_histories[receiver_id]:
                    last_recv_msg = self.agent_histories[receiver_id][-1]
                    recv_struct = self.parser.parse(last_recv_msg, dataset_type=self.dataset_type)
                    # Keep only conclusions not present in receiver's last response (naive string match for now)
                    recv_conclusions = [c.lower().strip() for c in recv_struct.intermediate_conclusions]
                    struct.intermediate_conclusions = [c for c in struct.intermediate_conclusions if c.lower().strip() not in recv_conclusions]

            filtered_msg = struct.to_formatted_string()

            # --- V2 Bug Fix: Use actual content sizes instead of heuristic multipliers ---
            transmitted_tokens = count_tokens(filtered_msg)
            
            # True compression metrics
            sem_sim = float(transmitted_tokens / max(original_tokens, 1))
            
            # Information retention based on parse tree preservation
            # (If final answer is intact, high baseline; else based on chunks kept)
            full_struct = self.parser.parse(message, dataset_type=self.dataset_type)
            entity_ret = 1.0 if not full_struct.evidence else (len(struct.evidence) / max(len(full_struct.evidence), 1))
            num_ret = 1.0 if not full_struct.claims else (len(struct.claims) / max(len(full_struct.claims), 1))
            info_ret = 0.5 * entity_ret + 0.5 * num_ret
            # --- End V2 Fix ---'''

new_transmit = '''        # V5 Priority #1 & #7: Continuous Latent Bottleneck / Reasoning Subgraph Extractor
        if w_ij < 0.05:
            filtered_msg = ""
            transmitted_tokens = 0
            sem_sim = 0.0
            info_ret = 0.0
        elif w_ij > 0.95:
            filtered_msg = message
            transmitted_tokens = count_tokens(filtered_msg)
            sem_sim = 1.0
            info_ret = 1.0
        else:
            # Extract final answer
            _ans_match = re.search(
                r'(?:Answer:|ANSWER:|The answer is:?)\\s*(.+?)(?:\\n\\n|\\Z)',
                message,
                re.IGNORECASE | re.DOTALL,
            )
            if _ans_match:
                ans_str = "Answer: " + _ans_match.group(1).strip()
            else:
                ans_str = "Answer: " + next((l.strip() for l in reversed(message.splitlines()) if l.strip()), '')

            # Split into reasoning nodes (sentences)
            sentences = [s.strip() + "." for s in re.split(r'(?<=[.!?])\\s+', message) if s.strip()]
            
            target_k = max(1, int(len(sentences) * w_ij))
            
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                vectorizer = TfidfVectorizer()
                # We want to extract the subgraph most relevant to the conclusion
                X = vectorizer.fit_transform(sentences)
                # Treat the answer string as the target query to find supporting evidence
                ans_vec = vectorizer.transform([ans_str])
                sims = cosine_similarity(X, ans_vec).flatten()
                
                # Keep top K most relevant sentences
                top_indices = np.argsort(sims)[-target_k:]
                top_indices.sort()  # Maintain logical chronological order
                filtered_msg = " ".join([sentences[i] for i in top_indices])
            except Exception:
                # Fallback: keep beginning and end (answer)
                half = target_k // 2
                filtered_msg = " ".join(sentences[:half] + ["..."] + sentences[-(target_k - half):])
            
            if ans_str not in filtered_msg:
                filtered_msg += f"\\n{ans_str}"
                
            transmitted_tokens = count_tokens(filtered_msg)
            sem_sim = w_ij  # Proxy for continuous bandwidth
            info_ret = w_ij  # InfoNCE will calculate true retention at round level'''
apply("Priority #1/7: Continuous Bottleneck", old_transmit, new_transmit)

# 6. SPRT Halting
old_halt = '''        if halt_decision and t > 1:
            halted = True
            break'''

new_halt = '''        # V5 Priority #6: SPRT Halting Logic
        # H1: Consensus (p=0.9), H0: Noise (p=0.1)
        p1 = 0.9; p0 = 0.1
        ans_list = [_normalize_for_vote(r.answer, config.dataset) for r in current_responses]
        agreements = 0
        total_pairs = 0
        for i in range(len(ans_list)):
            for j in range(i+1, len(ans_list)):
                agreements += int(ans_list[i] == ans_list[j])
                total_pairs += 1
                
        if total_pairs > 0:
            k = agreements; n = total_pairs
            llr = k * np.log(p1/p0) + (n - k) * np.log((1-p1)/(1-p0))
            halt_decision = llr > 2.0  # Strong evidence threshold for H1
        else:
            halt_decision = False
            
        if halt_decision and t > 1:
            halted = True
            break'''
apply("Priority #6: SPRT Halting", old_halt, new_halt)

with open('temp_check.py', 'w', encoding='utf-8') as f:
    f.write(src)

print("\n".join(fixes))

try:
    py_compile.compile('temp_check.py', doraise=True)
    print("\n[OK] Syntax check passed!")
except py_compile.PyCompileError as e:
    print(f"\n[FAIL] Syntax error: {e}")
    sys.exit(1)

nb = nbformat.read('SparseComm_Kaggle.ipynb', as_version=4)
nb.cells[2].source = src
nbformat.write(nb, 'SparseComm_Kaggle.ipynb')
print("[OK] Notebook updated with V5 fixes.")
