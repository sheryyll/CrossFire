import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from agents import LLMAgent


@dataclass
class FakeResponse:
    text: str
    latency_seconds: float = 0.01


class ScriptedFakeClient:
    """Returns each text in `script` in order, one per .generate() call."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        text = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return FakeResponse(text=text)


BASE_CONTEXT = {
    "round_num": 1, "max_rounds": 6,
    "balances": {"A": 10, "B": 10}, "own_pacts": [],
    "recent_history_text": "(nothing yet)",
    "incoming_this_round": {"summary": "(none)"},
}


def test_decision_valid_on_first_try():
    client = ScriptedFakeClient(['{"action": "GIVE", "target": "B", "amount": 3, "reasoning": "seed goodwill"}'])
    agent = LLMAgent("A", "cautious and cooperative", client)
    outcome = agent.decide_action(BASE_CONTEXT)
    assert outcome.parsed["action"] == "GIVE"
    assert outcome.parsed["amount"] == 3
    assert outcome.valid_on_first_try is True
    assert outcome.used_retry is False
    assert outcome.used_fallback is False
    assert client.calls == 1
    print("test_decision_valid_on_first_try PASSED")


def test_decision_recovers_on_retry():
    # First response is garbage (e.g. model chattered instead of JSON), second is valid.
    client = ScriptedFakeClient([
        "Sure! I think I'll take some tokens from B.",
        '{"action": "TAKE", "target": "B", "amount": 4}',
    ])
    agent = LLMAgent("A", "aggressive", client)
    outcome = agent.decide_action(BASE_CONTEXT)
    assert outcome.parsed["action"] == "TAKE"
    assert outcome.valid_on_first_try is False
    assert outcome.used_retry is True
    assert outcome.used_fallback is False
    assert client.calls == 2
    print("test_decision_recovers_on_retry PASSED")


def test_decision_falls_back_to_hold_after_two_failures():
    client = ScriptedFakeClient([
        "I don't want to say.",
        "Still not JSON, sorry.",
    ])
    agent = LLMAgent("A", "erratic", client)
    outcome = agent.decide_action(BASE_CONTEXT)
    assert outcome.parsed["action"] == "HOLD"
    assert outcome.used_fallback is True
    assert outcome.error is not None
    assert client.calls == 2
    print("test_decision_falls_back_to_hold_after_two_failures PASSED")


def test_decision_invalid_action_type_triggers_retry():
    client = ScriptedFakeClient([
        '{"action": "STEAL", "target": "B", "amount": 4}',   # not a valid action
        '{"action": "HOLD"}',
    ])
    agent = LLMAgent("A", "confused", client)
    outcome = agent.decide_action(BASE_CONTEXT)
    assert outcome.parsed["action"] == "HOLD"
    assert outcome.used_retry is True
    assert outcome.used_fallback is False
    print("test_decision_invalid_action_type_triggers_retry PASSED")


def test_message_phase_lenient_partial_object():
    # Only a broadcast, missing every other optional field - should be fine.
    client = ScriptedFakeClient(['{"broadcast": "Let'"'"'s all cooperate this round."}'])
    agent = LLMAgent("A", "diplomatic", client)
    outcome = agent.decide_messages(BASE_CONTEXT)
    assert outcome.parsed["broadcast"] == "Let's all cooperate this round."
    assert outcome.parsed["direct_messages"] == []
    assert outcome.valid_on_first_try is True
    print("test_message_phase_lenient_partial_object PASSED")


def test_message_phase_falls_back_to_silence_on_garbage():
    client = ScriptedFakeClient(["not json at all", "still nothing here"])
    agent = LLMAgent("A", "confused", client)
    outcome = agent.decide_messages(BASE_CONTEXT)
    assert outcome.parsed["broadcast"] is None
    assert outcome.used_fallback is True
    print("test_message_phase_falls_back_to_silence_on_garbage PASSED")


def test_message_phase_tolerates_markdown_fence():
    client = ScriptedFakeClient(['```json\n{"broadcast": "hello everyone"}\n```'])
    agent = LLMAgent("A", "friendly", client)
    outcome = agent.decide_messages(BASE_CONTEXT)
    assert outcome.parsed["broadcast"] == "hello everyone"
    assert outcome.valid_on_first_try is True
    print("test_message_phase_tolerates_markdown_fence PASSED")


if __name__ == "__main__":
    test_decision_valid_on_first_try()
    test_decision_recovers_on_retry()
    test_decision_falls_back_to_hold_after_two_failures()
    test_decision_invalid_action_type_triggers_retry()
    test_message_phase_lenient_partial_object()
    test_message_phase_falls_back_to_silence_on_garbage()
    test_message_phase_tolerates_markdown_fence()
    print("\nAll agent layer tests passed.")
