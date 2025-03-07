from anthropic import AnthropicVertex
import os
from functools import lru_cache

@lru_cache(maxsize=1000)
def call_llm(prompt: str) -> str:
    client = AnthropicVertex(
        region=os.getenv("ANTHROPIC_REGION", "us-east5"),
        project_id=os.getenv("ANTHROPIC_PROJECT_ID", "your-project-id")
    )
    response = client.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        model="claude-3-7-sonnet@20250219"
    )
    return response.content[0].text

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"
    response = call_llm(test_prompt)
    print(f"Test successful. Response: {response}")
