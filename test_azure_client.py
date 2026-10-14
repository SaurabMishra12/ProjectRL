import os
from openai import AzureOpenAI, OpenAI
import httpx

api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://vaibhavka43-0128-resource.openai.azure.com/")
base_url = os.getenv("AZURE_OPENAI_BASE_URL", "https://vaibhavka43-0128-resource.services.ai.azure.com/openai/v1")

print("--- Test 1: AzureOpenAI client ---")
try:
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version="2024-02-15-preview",
    )
    completion = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print("AzureOpenAI Success:", completion.choices[0].message.content)
except Exception as e:
    print("AzureOpenAI Error:", e)

print("\n--- Test 2: OpenAI client with base_url ---")
try:
    client2 = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    completion2 = client2.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print("OpenAI Success:", completion2.choices[0].message.content)
except Exception as e:
    print("OpenAI Error:", e)
