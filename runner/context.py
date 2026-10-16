"""
Bridges the engine's raw state (models.py / game.py) into the context dicts
that agents/prompts.py expects, per Agent Decision Contract Sections 2-5.
Kept separate from game_runner.py so the "what does an agent see" logic is
easy to find and tune on its own.
"""
from typing import Dict, Any


def build_history_text(game) -> str:
    """Per Decision Contract Section 5: a compact summary of past rounds.
    MVP keeps every round at this same summarized level - since games are
    only 6-10 rounds long, the full history is short enough not to need a
    separate 'last 2 rounds in full detail' tier yet. That refinement is a
    natural place to expand later if context length becomes a real problem."""
    completed_rounds = game.rounds_log[:-1]  # exclude the in-progress round
    if not completed_rounds:
        return "(nothing yet - this is the first round)"

    lines = []
    for log in completed_rounds:
        parts = []
        for d in log.decisions:
            if d.action.value == "GIVE" and d.target:
                parts.append(f"{d.agent_id} gave {d.amount} to {d.target}")
            elif d.action.value == "TAKE" and d.target:
                parts.append(f"{d.agent_id} took {d.amount} from {d.target}")
        for a in log.alliances:
            parts.append(f"ALLIANCE: {' + '.join(a['allies'])} both took from {a['target']}")
        for b in log.betrayals:
            parts.append(f"BETRAYAL: {b['betrayer']} broke their pact with {b['victim']}")
        lines.append(f"Round {log.round_num}: " + ("; ".join(parts) if parts else "everyone held"))
    return "\n".join(lines)


def build_incoming_summary(agent_id: str, round_log, pending_proposals) -> str:
    """What THIS agent has seen so far this round: broadcasts, DMs addressed
    to it, and proposals awaiting its response. Never includes DMs between
    other agents, per the private-message visibility rule (rules spec Section 7)."""
    lines = []
    for m in round_log.messages:
        if m.is_broadcast and m.from_agent != agent_id:
            lines.append(f"[broadcast from {m.from_agent}] {m.text}")
        elif not m.is_broadcast and m.to_agent == agent_id:
            lines.append(f"[private message from {m.from_agent}] {m.text}")
    for p in pending_proposals:
        if p.to_agent == agent_id and p.status.value == "pending":
            lines.append(
                f"[PROPOSAL {p.proposal_id} from {p.from_agent}] \"{p.terms_text}\" "
                f"for {p.duration_rounds} round(s) - respond in your 'responses' field"
            )
    return "\n".join(lines) if lines else "(nothing addressed to you yet this round)"


def build_context(game, agent_id: str, round_log, phase: str) -> Dict[str, Any]:
    own_pacts = []
    for p in game.get_active_pacts_for(agent_id):
        other = p.agent_b if p.agent_a == agent_id else p.agent_a
        own_pacts.append({
            "other_agent": other, "terms": p.terms_text,
            "expires_after_round": p.expires_after_round,
        })

    context = {
        "round_num": game.current_round,
        "max_rounds": game.max_rounds,
        "balances": game.get_public_balances(),
        "own_pacts": own_pacts,
        "recent_history_text": build_history_text(game),
    }
    if phase == "message":
        context["incoming_this_round"] = {
            "summary": build_incoming_summary(agent_id, round_log, game.current_round_proposals)
        }
    return context
