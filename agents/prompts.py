from typing import Dict, Any

RULES_SUMMARY = """
GAME: Shifting Alliances. You are one of several agents. Everyone starts with
tokens. The game runs for a fixed number of rounds. Your goal: end the game
with the highest token balance.

Each round has two phases:
1. MESSAGE PHASE: you may send a public broadcast, private direct messages,
   propose a pact to another agent, and/or accept/reject pacts proposed to you.
2. DECISION PHASE: you choose exactly one action: GIVE tokens to someone,
   TAKE tokens from someone, or HOLD (do nothing).

ALLIANCES: if two or more agents TAKE from the same target in the same round,
each of them gets a small bonus - so coordinating TAKE actions with others can
pay off.

PACTS: if another agent accepts your proposal, it becomes a binding pact for
a number of rounds. If you TAKE from a pact partner while the pact is still
active, that is logged as a BETRAYAL - it doesn't stop you mechanically, but
it is visible in the public action history once it happens.

All agents' token balances and past GIVE/TAKE actions are PUBLIC. Broadcasts
are public. Direct messages are private between sender and recipient. Pacts
you have with others are known only to you and them unless someone reveals it.
""".strip()


def build_system_prompt(agent_id: str, persona: str) -> str:
    return (
        f"You are {agent_id}, playing a strategy game called Shifting Alliances.\n"
        f"Your personality/strategy: {persona}\n\n"
        f"{RULES_SUMMARY}"
    )


def _format_state(context: Dict[str, Any]) -> str:
    balances_lines = "\n".join(f"  - {aid}: {bal} tokens" for aid, bal in context["balances"].items())
    pacts = context.get("own_pacts") or []
    if pacts:
        pact_lines = "\n".join(
            f"  - With {p['other_agent']}: {p['terms']} (active through round {p['expires_after_round']})"
            for p in pacts
        )
    else:
        pact_lines = "  - (none)"
    return (
        f"Round {context['round_num']} of {context['max_rounds']}.\n\n"
        f"Current balances:\n{balances_lines}\n\n"
        f"Your active pacts:\n{pact_lines}\n\n"
        f"Recent history:\n{context.get('recent_history_text', '(nothing yet)')}"
    )


def build_message_phase_prompt(context: Dict[str, Any]) -> str:
    incoming = context.get("incoming_this_round", {})
    incoming_text = incoming.get("summary", "(no messages to you yet this round)")
    return (
        f"{_format_state(context)}\n\n"
        f"Messages/proposals directed at you so far this round:\n{incoming_text}\n\n"
        "It is the MESSAGE PHASE. Respond with ONLY a JSON object, no other text, "
        "in exactly this shape (all fields optional - omit or null anything you don't need):\n"
        "{\n"
        '  "broadcast": "string or null",\n'
        '  "direct_messages": [{"to": "agent_id", "text": "string"}],\n'
        '  "proposal": {"to": "agent_id", "terms": "string", "duration_rounds": 2} or null,\n'
        '  "responses": [{"proposal_id": "string", "decision": "accept"}]\n'
        "}"
    )


def build_decision_phase_prompt(context: Dict[str, Any]) -> str:
    return (
        f"{_format_state(context)}\n\n"
        "It is the DECISION PHASE. Choose exactly one action. Respond with ONLY "
        "a JSON object, no other text, in exactly this shape:\n"
        "{\n"
        '  "action": "GIVE" or "TAKE" or "HOLD",\n'
        '  "target": "agent_id or null",\n'
        '  "amount": integer or null,\n'
        '  "reasoning": "one short sentence explaining why (for your own records only)"\n'
        "}"
    )


def build_correction_prompt(schema_hint: str, error: str) -> str:
    return (
        f"Your previous response could not be parsed: {error}\n"
        f"Respond again with ONLY a valid JSON object matching this shape:\n{schema_hint}"
    )
