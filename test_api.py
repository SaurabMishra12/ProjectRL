"""Quick API test with timeout"""
from openai import OpenAI
import httpx

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="your-nvidia-api-key",
    timeout=httpx.Timeout(60.0, connect=15.0),
    max_retries=2,
)

print("Testing NVIDIA DeepSeek API (non-streaming, 60s timeout)...")
try:
    response = client.chat.completions.create(
        model="deepseek-ai/deepseek-v3.2",
        messages=[{"role": "user", "content": "What is 3 + 5? Answer with just the number."}],
        temperature=0.7,
        max_tokens=64,
        stream=False,
    )
    text = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    print(f"Answer: {text}")
    print(f"Tokens: {tokens}")
    print("SUCCESS!")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
