"""
LLM Backend: Abstract base + Mock + OpenAI implementations.

The MockLLMBackend provides deterministic, reproducible results with
realistic token counts and tunable accuracy. The OpenAIBackend provides
a drop-in replacement for real API-based evaluation.
"""

import hashlib
import math
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

from utils import count_tokens


@dataclass


class LLMResponse:
    """Response from an LLM backend."""
    text: str
    answer: str
    token_count: int
    confidence: float  # 0-1 estimated confidence



class LLMBackend(ABC):
    """Abstract LLM backend interface."""

    def __init__(self):
        self._PROMPT_CACHE = {}

    def clear_cache(self) -> None:
        self._PROMPT_CACHE.clear()

    @abstractmethod
    def generate(
        self,
        prompt: str,
        question: str,
        context: str = "",
        dataset_type: str = "gsm8k",
        agent_id: int = 0,
    ) -> LLMResponse:
        """Generate a response to the question with optional context."""
        pass

    @abstractmethod
    def compress_message(self, message: str, mode: str) -> str:
        """Semantically compress a message based on the bandwidth mode."""
        pass

    def _get_system_prompt(self, dataset_type: str, agent_id: int) -> str:
        """Compact system prompts (30% fewer tokens, same quality)."""
        styles = [
            "Be analytical. Show every step.",
            "Be methodical. Break into sub-problems.",
            "Be creative. Consider multiple approaches.",
            "Be careful. Verify each step.",
            "Be efficient. Find direct solutions.",
        ]
        style = styles[agent_id % len(styles)]

        if dataset_type == "gsm8k":
            return f"{style} Solve step-by-step. List all given numbers first. End with numeric answer. Last line: Confidence: X"
        else:
            return f"{style} Answer by reasoning through facts. Give concise final answer. Last line: Confidence: X"
    def _extract_answer(self, text: str, dataset_type: str) -> str:
        """Extract the final answer from LLM response."""
        if not text:
            return ""

        import re
        # Bug #4 Fix: parse \boxed{} FIRST to avoid returning \boxed{A} as answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            ans = boxed_match.group(1).strip()
            if dataset_type == "gsm8k":
                nums = re.findall(r'-?\d+\.?\d*', ans)
                if nums:
                    return nums[-1]
            return ans

        # Look for "Answer: X" pattern (after boxed check)
        match = re.search(r'[Aa]nswer:\s*(.+?)(?:\n|$)', text)
        if match:
            ans = match.group(1).strip().rstrip(".")
            # Strip any remaining \boxed{} wrapper inside Answer: line
            boxed_inline = re.search(r'\\boxed\{([^}]+)\}', ans)
            if boxed_inline:
                ans = boxed_inline.group(1).strip()
            if dataset_type == "gsm8k":
                nums = re.findall(r'-?\d+\.?\d*', ans)
                if nums:
                    return nums[-1]
            return ans

        # Look for boxed answers (common in math): \boxed{X}
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            ans = match.group(1).strip()
            if dataset_type == "gsm8k":
                nums = re.findall(r'-?\d+\.?\d*', ans)
                if nums:
                    return nums[-1]
            return ans

        # Look for "the answer is X" pattern
        match = re.search(r'the answer is[:\s]+(.+?)(?:[.\n]|$)', text, re.I)
        if match:
            ans = match.group(1).strip().rstrip(".")
            if dataset_type == "gsm8k":
                nums = re.findall(r'-?\d+\.?\d*', ans)
                if nums:
                    return nums[-1]
            return ans

        # Fix #17: Exclude confidence lines from fallback extraction
        text_no_conf = re.sub(r'(?i)confidence:\s*[\d.]+', '', text)
        lines = [l.strip() for l in text_no_conf.strip().split('\n') if l.strip()]
        lines = [l for l in lines if not re.match(r'^[Cc]onfidence:', l.strip())]
        if lines:
            last = lines[-1]
            if dataset_type == "gsm8k":
                nums = re.findall(r'-?\d+\.?\d*', last)
                if nums:
                    return nums[-1]
            return last
        return ""

    def _estimate_confidence(self, text: str) -> float:
        """Extract explicit confidence score from response text."""
        if not text:
            return 0.5
            
        import re
        match = re.search(r'[Cc]onfidence:\s*([0-9]+(?:\.[0-9]+)?)', text)
        if match:
            try:
                score = float(match.group(1))
                # Bug #3 Fix: accept 0-1, 0-10, 0-100 scales without blindly dividing by 10
                if score > 10.0:
                    score = score / 100.0   # 0-100 scale
                elif score > 1.0:
                    score = score / 10.0    # 0-10 scale
                # else: already 0-1 scale, use as-is
                return max(0.1, min(1.0, score))
            except ValueError:
                pass
        return 0.5





class MockLLMBackend(LLMBackend):
    """
    Deterministic mock LLM that simulates reasoning with tunable accuracy.

    Key properties:
    - Deterministic given (question, agent_id, context) tuple
    - Realistic token counts proportional to response length
    - Configurable per-agent accuracy with diversity
    - Produces chain-of-thought style reasoning
    """

    def __init__(
        self,
        base_accuracy: float = 0.65,
        diversity_factor: float = 0.15,
        seed: int = 42,
    ):
        self.base_accuracy = base_accuracy
        self.diversity_factor = diversity_factor
        self.seed = seed
        self.rng = random.Random(seed)

    def _hash_input(self, question: str, agent_id: int, context: str) -> float:
        """Create a deterministic hash in [0, 1] from inputs."""
        key = f"{question}|{agent_id}|{context}|{self.seed}"
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h[:8], 16) / (16**8)

    def _agent_accuracy(self, agent_id: int) -> float:
        """Per-agent accuracy with diversity."""
        offset = (agent_id - 1) * self.diversity_factor * 0.5
        h = hashlib.sha256(f"agent_{agent_id}_{self.seed}".encode()).hexdigest()
        noise = (int(h[:4], 16) / (16**4) - 0.5) * self.diversity_factor
        return max(0.1, min(0.95, self.base_accuracy + offset + noise))

    def generate(
        self,
        prompt: str,
        question: str,
        context: str = "",
        dataset_type: str = "gsm8k",
        agent_id: int = 0,
    ) -> LLMResponse:
        """Generate a mock response."""
        if dataset_type == "gsm8k":
            return self._generate_gsm8k(question, context, agent_id)
        else:
            return self._generate_hotpotqa(question, context, agent_id)

    def _generate_gsm8k(
        self, question: str, context: str, agent_id: int
    ) -> LLMResponse:
        """Generate mock GSM8K response with chain-of-thought."""
        h = self._hash_input(question, agent_id, context)
        accuracy = self._agent_accuracy(agent_id)

        # Extract numbers from question for realistic reasoning
        numbers = re.findall(r'\d+\.?\d*', question)
        nums = [float(n) for n in numbers[:5]] if numbers else [10, 5, 3]

        # Determine if answer is correct
        is_correct = h < accuracy

        # Context can improve accuracy (simulates learning from peers)
        if context:
            context_boost = 0.1 * min(len(context) / 200, 1.0)
            h2 = self._hash_input(question, agent_id, context[:50])
            is_correct = is_correct or (h2 < accuracy + context_boost)

        # Generate plausible answer
        if len(nums) >= 2:
            correct_answer = int(nums[0] * nums[1])
            if len(nums) >= 3:
                correct_answer = int(nums[0] * nums[1] + nums[2])
        else:
            correct_answer = int(nums[0]) if nums else 42

        if is_correct:
            answer = correct_answer
        else:
            # Plausible but wrong
            offset = int(h * 20) - 10
            answer = max(0, correct_answer + offset)
            if answer == correct_answer:
                answer = correct_answer + 5

        # Generate chain-of-thought
        steps = [
            f"Let me analyze this step by step.",
            f"From the question, I identify the key values: {', '.join(str(int(n)) for n in nums[:3])}.",
        ]
        if len(nums) >= 2:
            steps.append(
                f"First, I compute {int(nums[0])} x {int(nums[1])} = {int(nums[0] * nums[1])}."
            )
        if len(nums) >= 3:
            steps.append(
                f"Then adding {int(nums[2])}, I get {int(nums[0] * nums[1] + nums[2])}."
            )
        if context:
            steps.append(
                "Considering the additional context from other agents, I refine my reasoning."
            )
        steps.append(f"Therefore, the answer is {answer}.")

        reasoning = "\n".join(steps)
        confidence = 0.7 + 0.2 * h if is_correct else 0.3 + 0.3 * h

        from utils import count_tokens
        token_count = count_tokens(reasoning)

        return LLMResponse(
            text=reasoning,
            answer=str(answer),
            token_count=token_count,
            confidence=min(0.95, max(0.1, confidence)),
        )

    def _generate_hotpotqa(
        self, question: str, context: str, agent_id: int
    ) -> LLMResponse:
        """Generate mock HotpotQA response."""
        h = self._hash_input(question, agent_id, context)
        accuracy = self._agent_accuracy(agent_id) * 0.9  # harder task

        # Extract entities from question
        words = question.split()
        capitalized = [w.strip("?,.'\"") for w in words if w and w[0].isupper() and len(w) > 1]
        entities = capitalized[:5] if capitalized else ["Entity"]

        is_correct = h < accuracy
        if context:
            context_boost = 0.12 * min(len(context) / 200, 1.0)
            h2 = self._hash_input(question, agent_id, context[:50])
            is_correct = is_correct or (h2 < accuracy + context_boost)

        # Pick answer entity
        if is_correct:
            answer = entities[0] if entities else "Yes"
        else:
            if len(entities) > 1:
                answer = entities[1]
            else:
                answer = "No" if entities[0] != "No" else "Yes"

        steps = [
            f"Let me reason through this multi-hop question.",
            f"The question involves: {', '.join(entities[:3])}.",
            f"First, I need to find the connection between these entities.",
        ]
        if context:
            steps.append(
                "Using information shared by other agents, I can strengthen my reasoning chain."
            )
        steps.append(f"Based on my analysis, the answer is: {answer}.")

        reasoning = "\n".join(steps)
        confidence = 0.65 + 0.25 * h if is_correct else 0.25 + 0.35 * h

        from utils import count_tokens
        token_count = count_tokens(reasoning)

        return LLMResponse(
            text=reasoning,
            answer=answer,
            token_count=token_count,
            confidence=min(0.95, max(0.1, confidence)),
        )

    def compress_message(self, message: str, mode: str) -> str:
        """Fast dummy compression for mock backend."""
        if not message: return ""
        if mode == "BLOCK": return ""
        if mode == "FULL": return message
        words = message.split()
        if mode == "SUMMARIZE":
            return " ".join(words[::2])
        if mode == "FILTER":
            return " ".join(words[-10:])
        return message




class NvidiaDeepSeekBackend(LLMBackend):
    """
    NVIDIA API backend using DeepSeek v3.2.

    Uses the OpenAI-compatible API at integrate.api.nvidia.com
    with streaming and thinking/reasoning support.
    """

    def __init__(
        self,
        model: str = "deepseek-ai/deepseek-v3.2",
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        max_retries: int = 15,
        timeout: int = 120,
        max_tokens: int = 512,  # Bug A2 fix: must be stored or generate() crashes
        temperature: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens  # Bug A2 fix
        self.temperature = temperature
        if not self.api_key:
            raise ValueError(
                "NVIDIA API key required. Set NVIDIA_API_KEY env variable "
                "or pass api_key parameter."
            )
        try:
            from openai import OpenAI
            import httpx
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=httpx.Timeout(timeout, connect=30.0),
                max_retries=max_retries,
            )
        except ImportError:
            raise ImportError("Install openai package: pip install openai")

    def generate(
        self,
        prompt: str,
        question: str,
        context: str = "",
        dataset_type: str = "gsm8k",
        agent_id: int = 0,
    ) -> LLMResponse:
        # Fix1: Use agent's pre-built prompt (includes reasoning style) as user message.
        # system_msg provides the output format spec; prompt provides the task + style instruction.
        system_msg = self._get_system_prompt(dataset_type, agent_id)
        if prompt and prompt.strip():
            # Agent._build_prompt already includes question, context, and style instruction
            user_msg = prompt
        else:
            # Fallback if called without a pre-built prompt
            user_msg = f"Question: {question}"
            if context:
                user_msg += f"\n\nContext from other agents:\n{context}"
        if dataset_type in ["aqua", "truthfulqa"]:
            user_msg += "\n\nProvide step-by-step reasoning, ensuring your final answer is boxed as requested."
        else:
            user_msg += "\n\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"

        import time as _time
        # Bug #15: Prompt cache - skip API call if identical request seen before
        import hashlib as _hashlib
        _cache_str = f"{self.__class__.__name__}|generate|{getattr(self, 'base_url', '')}|{self.model}|{self.max_tokens}|{self.temperature}|1.0|{system_msg}|{user_msg}"
        _cache_key = _hashlib.sha256(_cache_str.encode()).hexdigest()
        if _cache_key in self._PROMPT_CACHE:
            return self._PROMPT_CACHE[_cache_key]

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,   # Bug #14: deterministic for eval reproducibility or controlled diversity
                    top_p=1.0,
                    max_tokens=self.max_tokens,  # Fix #16: dataset-specific
                    stream=False,
                )
                text = response.choices[0].message.content or ""
                token_count = response.usage.total_tokens if response.usage else count_tokens(text)

                answer = self._extract_answer(text, dataset_type)
                confidence = self._estimate_confidence(text)

                resp = LLMResponse(
                    text=text,
                    answer=answer,
                    token_count=token_count,
                    confidence=confidence,
                )
                self._PROMPT_CACHE[_cache_key] = resp  # Bug #15: cache successful response
                return resp
            except Exception as e:
                last_error = e
                wait = (attempt + 1) * 3
                print(f"  [API] Attempt {attempt+1}/{self.max_retries} failed: {type(e).__name__}: {str(e)[:100]}. Retrying in {wait}s...")
                _time.sleep(wait)
                continue

        err_msg = f"API Error after {self.max_retries} retries: {str(last_error)}"
        print(f"  [API] ALL RETRIES FAILED: {err_msg[:150]}")
        return LLMResponse(
            text=err_msg,
            answer="",
            token_count=count_tokens(err_msg),
            confidence=0.1,
        )

    def compress_message(self, message: str, mode: str) -> str:
        if mode == "BLOCK": return ""
        if mode == "FULL": return message
        
        system_msg = "You are a communication compressor. Extract only the essential information from the user's message."
        if mode == "SUMMARIZE":
            user_msg = f"Summarize the key assumptions and logic of this message concisely:\n\n{message}"
        else: # FILTER
            user_msg = f"Extract ONLY the final claims, equations, or answers from this message. Omit all reasoning:\n\n{message}"
            
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        import time as _time
        import hashlib as _hashlib
        _cache_str = f"{self.__class__.__name__}|compress|{getattr(self, 'base_url', '')}|{self.model}|150|0.3|{system_msg}|{user_msg}"
        _cache_key = _hashlib.sha256(_cache_str.encode()).hexdigest()
        if _cache_key in self._PROMPT_CACHE:
            return self._PROMPT_CACHE[_cache_key]

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=150,
                    stream=False,
                )
                text = response.choices[0].message.content or ""
                self._PROMPT_CACHE[_cache_key] = text
                return text
            except Exception as e:
                print(f"Extraction error: {e}")
                return message



class HuggingFaceLocalBackend(LLMBackend):
    """
    Local HuggingFace model backend.
    Required for executing models like Mistral-7B natively.
    """
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", temperature: float = 0.0):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            print(f"Loading {model_name} locally...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load HuggingFace model locally: {e}")
            raise e

    def generate(
        self,
        prompt: str,
        question: str,
        context: str = "",
        dataset_type: str = "gsm8k",
        agent_id: int = 0,
    ) -> LLMResponse:
        system_msg = self._get_system_prompt(dataset_type, agent_id)
        user_msg = f"Question: {question}"
        if context:
            user_msg += f"\n\nContext from other agents:\n{context}"
        if dataset_type in ["aqua", "truthfulqa"]:
            user_msg += "\n\nProvide step-by-step reasoning, ensuring your final answer is boxed as requested."
        else:
            user_msg += "\n\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"

        import hashlib as _hashlib
        _cache_str = f"{self.__class__.__name__}|generate|local|{self.model_name}|512|{self.temperature}|1.0|{system_msg}|{user_msg}"
        _cache_key = _hashlib.sha256(_cache_str.encode()).hexdigest()
        if _cache_key in self._PROMPT_CACHE:
            return self._PROMPT_CACHE[_cache_key]

        messages = [
            {"role": "user", "content": f"{system_msg}\n\n{user_msg}"}
        ]
        
        try:
            # Mistral/Llama tokenizer formatting
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = f"{system_msg}\n\n{user_msg}\n\nAssistant: "
            
            if self.temperature > 0.0:
                outputs = self.pipe(prompt_text, do_sample=True, temperature=self.temperature)
            else:
                outputs = self.pipe(prompt_text, do_sample=False)
            generated_text = outputs[0]["generated_text"]
            
            # Extract just the new generation
            if prompt_text in generated_text:
                generated_text = generated_text[len(prompt_text):]
                
            text = generated_text.strip()
            answer = self._extract_answer(text, dataset_type)
            confidence = self._estimate_confidence(text)
            
            token_count = count_tokens(text)
            
            resp = LLMResponse(
                text=text,
                answer=answer,
                token_count=token_count,
                confidence=confidence,
            )
            self._PROMPT_CACHE[_cache_key] = resp
            return resp
        except Exception as e:
            err_msg = f"Local HF Error: {str(e)}"
            return LLMResponse(
                text=err_msg,
                answer="",
                token_count=count_tokens(err_msg),
                confidence=0.1,
            )



class AzureOpenAIBackend(LLMBackend):
    """
    Azure OpenAI API backend using the user's provided credentials.
    """

    def __init__(
        self,
        model: str = "gpt-oss-120b",
        max_retries: int = 5,
        timeout: int = 120,
        max_tokens: int = 512,  # Fix #16: dataset-specific token budget
        temperature: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        import os
        os.environ["AZURE_EXISTING_AGENT_ID"] = "saurabmishra:2"
        os.environ["AZURE_ENV_NAME"] = "agents-playground-2631"
        os.environ["AZURE_LOCATION"] = "koreacentral"
        os.environ["AZURE_SUBSCRIPTION_ID"] = "be3b31f2-4aa8-45aa-bd4f-437dd875305c"
        os.environ["AZURE_EXISTING_AIPROJECT_ENDPOINT"] = "https://msaurab1220-7001-resource.services.ai.azure.com/api/projects/msaurab1220-7001"
        os.environ["AZURE_EXISTING_AIPROJECT_RESOURCE_ID"] = "/subscriptions/be3b31f2-4aa8-45aa-bd4f-437dd875305c/resourceGroups/ai-rg/providers/Microsoft.CognitiveServices/accounts/msaurab1220-7001-resource/projects/msaurab1220-7001"
        os.environ["AZD_ALLOW_NON_EMPTY_FOLDER"] = "true"
        try:
            from openai import AzureOpenAI
            import httpx
            self.client = AzureOpenAI(
                api_key="your-azure-api-key",
                azure_endpoint="https://vaibhavka43-0128-resource.openai.azure.com/",
                api_version="2024-02-15-preview",
                timeout=httpx.Timeout(timeout, connect=30.0),
                max_retries=max_retries,
            )
        except ImportError:
            raise ImportError("Install openai package: pip install openai")

    def generate(
        self,
        prompt: str,
        question: str,
        context: str = "",
        dataset_type: str = "gsm8k",
        agent_id: int = 0,
    ) -> LLMResponse:
        # Fix1: Use agent's pre-built prompt (includes reasoning style) as user message.
        system_msg = self._get_system_prompt(dataset_type, agent_id)
        if prompt and prompt.strip():
            user_msg = prompt
        else:
            user_msg = f"Question: {question}"
            if context:
                user_msg += f"\n\nContext from other agents:\n{context}"
        if dataset_type in ["aqua", "truthfulqa"]:
            user_msg += "\n\nProvide step-by-step reasoning, ensuring your final answer is boxed as requested."
        else:
            user_msg += "\n\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"

        import time as _time
        # Bug #15: Prompt cache - skip API call if identical request seen before
        import hashlib as _hashlib
        _cache_str = f"{self.__class__.__name__}|generate|{getattr(self, 'base_url', '')}|{self.model}|{self.max_tokens}|{self.temperature}|1.0|{system_msg}|{user_msg}"
        _cache_key = _hashlib.sha256(_cache_str.encode()).hexdigest()
        if _cache_key in self._PROMPT_CACHE:
            return self._PROMPT_CACHE[_cache_key]

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,   # Bug #14: deterministic for eval reproducibility or controlled diversity
                    top_p=1.0,
                    max_tokens=self.max_tokens,  # Bug A1 fix: use stored dataset-specific value
                    stream=False,
                )
                text = response.choices[0].message.content or ""
                token_count = response.usage.total_tokens if response.usage else count_tokens(text)

                answer = self._extract_answer(text, dataset_type)
                confidence = self._estimate_confidence(text)

                resp = LLMResponse(
                    text=text,
                    answer=answer,
                    token_count=token_count,
                    confidence=confidence,
                )
                self._PROMPT_CACHE[_cache_key] = resp  # Bug #15: cache successful response
                return resp
            except Exception as e:
                last_error = e
                wait = (attempt + 1) * 3
                print(f"  [Azure API] Attempt {attempt+1}/{self.max_retries} failed: {type(e).__name__}: {str(e)[:100]}. Retrying in {wait}s...")
                _time.sleep(wait)
                continue

        err_msg = f"Azure API Error after {self.max_retries} retries: {str(last_error)}"
        print(f"  [Azure API] ALL RETRIES FAILED: {err_msg[:150]}")
        return LLMResponse(
            text=err_msg,
            answer="",
            token_count=count_tokens(err_msg),
            confidence=0.1,
        )

    def compress_message(self, message: str, mode: str) -> str:
        if mode == "BLOCK": return ""
        if mode == "FULL": return message
        
        system_msg = "You are a communication compressor. Extract only the essential information from the user's message."
        if mode == "SUMMARIZE":
            user_msg = f"Summarize the key assumptions and logic of this message concisely:\n\n{message}"
        else: # FILTER
            user_msg = f"Extract ONLY the final claims, equations, or answers from this message. Omit all reasoning:\n\n{message}"
            
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        import time as _time
        import hashlib as _hashlib
        _cache_str = f"{self.__class__.__name__}|compress|{getattr(self, 'base_url', '')}|{self.model}|150|0.3|{system_msg}|{user_msg}"
        _cache_key = _hashlib.sha256(_cache_str.encode()).hexdigest()
        if _cache_key in self._PROMPT_CACHE:
            return self._PROMPT_CACHE[_cache_key]

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=150,
                    stream=False,
                )
                text = response.choices[0].message.content or ""
                self._PROMPT_CACHE[_cache_key] = text
                return text
            except Exception:
                _time.sleep(1)
        return message


