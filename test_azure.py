import os
import sys
from openai import OpenAI

api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
endpoint = os.getenv("AZURE_OPENAI_BASE_URL", "https://vaibhavka43-0128-resource.services.ai.azure.com/openai/v1")

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

try:
    completion = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?",
            }
        ],
    )
    print("Direct key:", completion.choices[0].message.content)
except Exception as e:
    print("Direct key error:", e)

