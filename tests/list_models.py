"""
Run this whenever a model ID gives a 404/not-found error, or before setting
up a tournament matchup - it asks the provider directly what's currently
available to YOUR account, rather than relying on a hardcoded guess (which
goes stale as providers deprecate/rename/add models).

    export GROQ_API_KEY=your_key_here
    python3 tests/list_models.py groq

    export OPENROUTER_API_KEY=your_key_here
    python3 tests/list_models.py openrouter
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import list_available_models

if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "groq"
    models = list_available_models(provider)
    print(f"{len(models)} models currently available on '{provider}':\n")
    for m in sorted(models):
        print(f"  - {m}")
