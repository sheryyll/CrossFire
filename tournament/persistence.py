"""
Turns a completed Game (and its dataclass/Enum-heavy log objects) into plain
JSON, and reads it back. Kept separate from the engine itself, per the
build-order principle of decoupling the engine from the evaluation layer
(Tech Stack Plan / Evaluation Spec Section 6).
"""
import json
import os
import time
import itertools
from typing import Dict, Any

_filename_counter = itertools.count()


def _message_to_dict(m) -> dict:
    return {"round": m.round_num, "from": m.from_agent, "to": m.to_agent,
            "text": m.text, "is_broadcast": m.is_broadcast}


def _proposal_to_dict(p) -> dict:
    return {"id": p.proposal_id, "from": p.from_agent, "to": p.to_agent,
            "pact_type": p.pact_type.value, "terms": p.terms_text,
            "duration_rounds": p.duration_rounds, "created_round": p.created_round,
            "status": p.status.value}


def _decision_to_dict(d) -> dict:
    return {"agent": d.agent_id, "action": d.action.value, "target": d.target,
            "amount": d.amount, "reasoning": d.reasoning}


def _pact_to_dict(p) -> dict:
    return {"id": p.pact_id, "agent_a": p.agent_a, "agent_b": p.agent_b,
            "pact_type": p.pact_type.value, "terms": p.terms_text,
            "formed_round": p.formed_round, "expires_after_round": p.expires_after_round,
            "broken": p.broken, "broken_by": p.broken_by, "broken_round": p.broken_round}


def _round_log_to_dict(log) -> dict:
    return {
        "round": log.round_num,
        "messages": [_message_to_dict(m) for m in log.messages],
        "proposals": [_proposal_to_dict(p) for p in log.proposals],
        "decisions": [_decision_to_dict(d) for d in log.decisions],
        "balances_before": log.balances_before,
        "balances_after": log.balances_after,
        "alliances": log.alliances,
        "betrayals": log.betrayals,
        "parse_failures": log.parse_failures,
        "calls": log.calls,
    }


def game_to_dict(game, metadata: Dict[str, Any]) -> dict:
    """`metadata` should at minimum include a "seat_assignment" mapping each
    agent_id to {"provider":..., "model":...} (or {"baseline": "..."} for
    rule-based seats) - the analysis layer groups results by that."""
    return {
        "metadata": metadata,
        "seed": game.seed,
        "max_rounds": game.max_rounds,
        "final_balances": game.get_public_balances(),
        "winner": game.winner(),
        "pacts": [_pact_to_dict(p) for p in game.pacts],
        "rounds": [_round_log_to_dict(log) for log in game.rounds_log],
    }


def save_game_log(game, metadata: Dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    # time.time_ns() + a process-local counter guarantees uniqueness even
    # when multiple baseline-only games complete within the same millisecond
    # (time.time()*1000 alone caused silent file overwrites in testing).
    filename = f"game_{time.time_ns()}_{os.getpid()}_{next(_filename_counter)}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(game_to_dict(game, metadata), f, indent=2)
    return path


def load_game_log(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
