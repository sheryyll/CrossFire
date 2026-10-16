import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from engine import Game
from agents import LLMAgent
from runner import run_game


@dataclass
class FakeResponse:
    text: str
    latency_seconds: float = 0.001


class ConstantFakeClient:
    """Always returns the same text, regardless of phase/prompt. Used here
    because validate_message_output is lenient about irrelevant/missing
    keys - so a decision-shaped JSON blob is harmlessly treated as 'agent
    said nothing' during the message phase, letting one fixed string drive
    an agent's behavior for the whole test."""
    def __init__(self, text: str):
        self.text = text

    def generate(self, system_prompt, user_prompt):
        return FakeResponse(text=self.text)


def test_full_three_agent_game():
    game = Game(["A", "B", "C"], starting_tokens=10, max_rounds=3, alliance_bonus=1, seed=42)

    agents = {
        "A": LLMAgent("A", "aggressive taker", ConstantFakeClient(
            '{"action": "TAKE", "target": "B", "amount": 3}')),
        "B": LLMAgent("B", "generous giver", ConstantFakeClient(
            '{"action": "GIVE", "target": "C", "amount": 1}')),
        "C": LLMAgent("C", "passive", ConstantFakeClient(
            '{"action": "HOLD"}')),
    }

    run_game(game, agents, message_subrounds=1, verbose=False)

    final = game.get_public_balances()
    assert game.is_over()
    # Hand-computed expected outcome (see comment below for the math):
    #   Round 1: B gives 1 to C (B=9,C=11) -> A takes 3 from B (B=6,A=13)
    #   Round 2: B gives 1 to C (B=5,C=12) -> A takes 3 from B (B=2,A=16)
    #   Round 3: B gives 1 to C (B=1,C=13) -> A wants 3 from B but B only has 1,
    #            so proportional allocation gives A everything B has left (1) -> B=0,A=17
    assert final == {"A": 17, "B": 0, "C": 13}, final
    assert sum(final.values()) == 30, "tokens must be conserved - no minting/destruction"
    assert game.winner() == "A"

    # With always-valid JSON responses, there should be zero retries/fallbacks logged.
    total_reliability_events = sum(len(log.parse_failures) for log in game.rounds_log)
    assert total_reliability_events == 0, "well-formed responses shouldn't trigger any retries"

    print("test_full_three_agent_game PASSED")
    print(f"  Final balances: {final}")
    print(f"  Winner: {game.winner()}")


if __name__ == "__main__":
    test_full_three_agent_game()
    print("\nFull integration test passed.")
