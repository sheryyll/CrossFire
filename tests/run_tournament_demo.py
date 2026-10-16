"""
Run a real tournament with real LLM-backed agents, on your own machine:

    export GROQ_API_KEY=your_key_here
    python3 tests/run_tournament_demo.py

Starts SMALL on purpose (repeats_per_rotation=1, max_rounds=4) so you can
see it work end-to-end quickly and check your rate limits before scaling up.
Once this runs cleanly, bump max_rounds and repeats_per_rotation up in this
same file for a more thorough tournament.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tournament import run_tournament, print_leaderboard

PARTICIPANTS = [
    {"kind": "llm", "provider": "groq", "model": "openai/gpt-oss-20b", "persona": "calculating and a little ruthless"},
    {"kind": "llm", "provider": "groq", "model": "openai/gpt-oss-120b", "persona": "diplomatic, prefers alliances over conflict"},
    {"kind": "llm", "provider": "groq", "model": "qwen/qwen3.8-27b", "persona": "cautious, avoids unnecessary risk"},
]

GAME_PARAMS = {"starting_tokens": 10, "max_rounds": 6, "alliance_bonus": 1}

if __name__ == "__main__":
    output_dir = "logs"
    print("Starting tournament - this makes real API calls and will take a few minutes...")

    paths = run_tournament(
        PARTICIPANTS,
        game_params=GAME_PARAMS,
        repeats_per_rotation=5,   
        output_dir=output_dir,
        message_subrounds=2,
        verbose=True,             # prints every game live, same as run_real_game.py
    )

    print(f"\nSaved {len(paths)} game logs to '{output_dir}/'\n")
    print_leaderboard(output_dir)
