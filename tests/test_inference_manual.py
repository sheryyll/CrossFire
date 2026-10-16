import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import LLMClient

MODEL_ID = "openai/gpt-oss-20b"  # confirmed available via list_models.py - fast, good default

if __name__ == "__main__":
    client = LLMClient(provider="groq", model=MODEL_ID, temperature=0.7)

    response = client.generate(
        system_prompt=(
            "You are Agent A in a strategy game called Shifting Alliances. "
            "You are calculating, a little ruthless, but not reckless."
        ),
        user_prompt=(
            "It's round 1. You have 10 tokens. Agent B has 10 tokens. "
            "Agent C has 10 tokens. In one short sentence, tell the group "
            "your opening strategy."
        ),
    )

    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Latency: {response.latency_seconds:.2f}s")
    print(f"Response:\n{response.text}")
