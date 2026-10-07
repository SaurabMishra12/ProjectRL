import os
from config import Config
from pipeline import create_llm_backend

def test_hf_local():
    print("Initializing HuggingFace Local Backend test...")
    config = Config()
    config.llm_backend = "hf"
    config.api_model = "mistralai/Mistral-7B-Instruct-v0.2"
    
    try:
        backend = create_llm_backend(config)
        
        print("Testing generation with Mistral...")
        res = backend.generate(
            prompt="Solve:",
            question="What is the square root of 144?",
            dataset_type="math",
            agent_id=0,
            context=None
        )
        print("\\n=== GENERATION SUCCESS ===")
        print("Response:", res.answer)
        print("Tokens used:", res.token_count)
    except Exception as e:
        import traceback
        print("\\n=== LOCAL HARDWARE OR OS FAILURE ===")
        traceback.print_exc()

if __name__ == "__main__":
    test_hf_local()
