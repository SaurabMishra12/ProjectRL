import os
from config import Config
from pipeline import create_llm_backend

def test_groq():
    config = Config()
    config.llm_backend = "api"
    config.api_model = "llama-3.1-8b-instant"
    config.api_key = "your-groq-api-key"
    config.api_base_url = "https://api.groq.com/openai/v1"
    
    backend = create_llm_backend(config)
    
    # Try generating a simple response
    res = backend.generate(
        prompt="Solve:",
        question="What is 2+2?",
        dataset_type="gsm8k",
        agent_id=0,
        context=None
    )
    print("GROQ Response Answer:", res.answer)
    print("Tokens used:", res.token_count)

if __name__ == "__main__":
    test_groq()
