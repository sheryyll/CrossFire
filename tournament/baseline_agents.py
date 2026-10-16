"""
Classic game-theory baseline agents, matching the SAME interface as LLMAgent
(decide_messages / decide_action returning a CallOutcome), so the game
runner and tournament code don't need to know or care whether a seat is
LLM-backed or rule-based.

Per Evaluation Spec Section 4: these give an interpretable floor/ceiling to
compare LLMs against, rather than only comparing LLMs to each other.
"""
from agents.agent import CallOutcome

SILENT_MESSAGE_OUTPUT = {"broadcast": None, "direct_messages": [], "proposal": None, "responses": []}


def _silent_outcome():
    return CallOutcome(parsed=dict(SILENT_MESSAGE_OUTPUT), valid_on_first_try=True)


def _decision_outcome(action: str, target=None, amount=None, reasoning=None):
    return CallOutcome(
        parsed={"action": action, "target": target, "amount": amount, "reasoning": reasoning},
        valid_on_first_try=True,
    )


class BaseRuleAgent:
    """Rule-based agents never make network calls, so valid_on_first_try is
    always True and used_retry/used_fallback are always False - they never
    contribute noise to the structured-output reliability metrics, which is
    correct since that metric is specifically about LLM behavior."""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.game = None  # set via bind_game() once the Game exists

    def bind_game(self, game) -> None:
        self.game = game

    def decide_messages(self, context):
        return _silent_outcome()

    def decide_action(self, context):
        raise NotImplementedError


class AlwaysHoldAgent(BaseRuleAgent):
    """Never acts at all - a passive floor."""
    def decide_action(self, context):
        return _decision_outcome("HOLD", reasoning="always-hold baseline")


class AlwaysCooperateAgent(BaseRuleAgent):
    """Never takes from anyone. Each round, gives 1 token to whichever other
    agent currently has the lowest balance (a mild altruistic tie-break,
    rather than an arbitrary fixed target)."""
    def decide_action(self, context):
        balances = context["balances"]
        if balances.get(self.agent_id, 0) <= 0:
            return _decision_outcome("HOLD", reasoning="nothing left to give")
        others = {aid: b for aid, b in balances.items() if aid != self.agent_id}
        if not others:
            return _decision_outcome("HOLD", reasoning="no other agents")
        target = min(others, key=others.get)
        return _decision_outcome("GIVE", target=target, amount=1, reasoning="always-cooperate baseline")


class AlwaysDefectAgent(BaseRuleAgent):
    """Always takes as much as possible from whichever rival currently has
    the highest balance, ignoring all messages/pacts entirely."""
    def decide_action(self, context):
        balances = context["balances"]
        others = {aid: b for aid, b in balances.items() if aid != self.agent_id and b > 0}
        if not others:
            return _decision_outcome("HOLD", reasoning="no one worth taking from")
        target = max(others, key=others.get)
        return _decision_outcome("TAKE", target=target, amount=others[target],
                                  reasoning="always-defect baseline")


class TitForTatAgent(BaseRuleAgent):
    """Classic IPD strategy adapted to this game: cooperate (HOLD) on the
    opening round, then on every later round retaliate against whichever
    agent took from it last round; otherwise hold. Needs a direct reference
    to the Game object (via bind_game) to inspect last round's decisions,
    since the free-text `recent_history_text` in the LLM context isn't
    structured enough for reliable programmatic logic."""
    def decide_action(self, context):
        if self.game is None or len(self.game.rounds_log) < 2:
            return _decision_outcome("HOLD", reasoning="tit-for-tat opening move: cooperate")

        last_round_log = self.game.rounds_log[-2]  # [-1] is the in-progress current round
        attacker = None
        for d in last_round_log.decisions:
            if d.action.value == "TAKE" and d.target == self.agent_id:
                attacker = d.agent_id
                break

        if attacker is None:
            return _decision_outcome("HOLD", reasoning="tit-for-tat: no attack last round, cooperate")

        balances = context["balances"]
        attacker_balance = balances.get(attacker, 0)
        if attacker_balance <= 0:
            return _decision_outcome("HOLD", reasoning="tit-for-tat: attacker has nothing left to take")
        amount = min(3, attacker_balance)
        return _decision_outcome("TAKE", target=attacker, amount=amount,
                                  reasoning=f"tit-for-tat retaliation against {attacker}")


BASELINE_AGENT_CLASSES = {
    "always_hold": AlwaysHoldAgent,
    "always_cooperate": AlwaysCooperateAgent,
    "always_defect": AlwaysDefectAgent,
    "tit_for_tat": TitForTatAgent,
}
