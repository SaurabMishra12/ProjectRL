from openai import OpenAI
import sys

endpoint = "https://vaibhavka43-0128-resource.services.ai.azure.com/openai/v1"
token_str = "your-azure-api-key"

# If it's a bearer token or compatible key, we can try passing it to OpenAI
client = OpenAI(
    base_url=endpoint,
    api_key=token_str
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
    print("Success with token string:", completion.choices[0].message.content)
except Exception as e:
    print("Error with token string:", e)

    # Let's try the DefaultAzureCredential if that failed
    print("Trying DefaultAzureCredential...")
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    try:
        token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")
        client2 = OpenAI(
            base_url=endpoint,
            api_key=token_provider
        )
        completion2 = client2.chat.completions.create(
            model="gpt-oss-120b",
            messages=[{"role": "user", "content": "Capital of France?"}]
        )
        print("Success with token_provider:", completion2.choices[0].message.content)
    except Exception as e2:
        print("Error with token_provider:", e2)

