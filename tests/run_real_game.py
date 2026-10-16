import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Game
from agents import LLMAgent
from inference import LLMClient
from runner import run_game

if __name__ == "__main__":
    game = Game(["A", "B", "C"], starting_tokens=10, max_rounds=6, alliance_bonus=1, seed=1)

    agents = {
        "A": LLMAgent("A", "calculating and a little ruthless",
                       LLMClient(provider="groq", model="openai/gpt-oss-20b")),
        "B": LLMAgent("B", "diplomatic, prefers alliances over conflict",
                       LLMClient(provider="groq", model="openai/gpt-oss-120b")),
        "C": LLMAgent("C", "cautious, avoids unnecessary risk",
                       LLMClient(provider="groq", model="qwen/qwen3.8-27b")),
    }

    run_game(game, agents, message_subrounds=2, verbose=True)
