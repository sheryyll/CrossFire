import json
import re
from typing import Optional, Dict, Any


class ValidationError(Exception):
    pass


def extract_json(text: str) -> Optional[dict]:
    """Finds and parses the first {...} JSON object in a blob of text, tolerant
    of surrounding commentary/markdown fences that smaller models often add
    despite being asked for raw JSON."""
    if not text:
        return None
    # Strip common markdown code fences first.
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    # Find the first balanced-looking {...} block.
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def validate_message_output(raw: Optional[dict]) -> Dict[str, Any]:
    """Lenient: always returns a well-formed dict, dropping any malformed
    sub-parts rather than failing the whole turn. An agent going silent
    (empty dict in, empty dict out) is a completely valid outcome."""
    result = {"broadcast": None, "direct_messages": [], "proposal": None, "responses": []}
    if not isinstance(raw, dict):
        return result

    if isinstance(raw.get("broadcast"), str) and raw["broadcast"].strip():
        result["broadcast"] = raw["broadcast"].strip()

    for dm in raw.get("direct_messages") or []:
        if isinstance(dm, dict) and isinstance(dm.get("to"), str) and isinstance(dm.get("text"), str):
            result["direct_messages"].append({"to": dm["to"], "text": dm["text"]})

    proposal = raw.get("proposal")
    if isinstance(proposal, dict) and isinstance(proposal.get("to"), str) and proposal.get("terms"):
        result["proposal"] = {
            "to": proposal["to"],
            "terms": str(proposal.get("terms", "")),
            "pact_type": proposal.get("pact_type", "non_aggression"),
            "duration_rounds": proposal.get("duration_rounds", 1)
            if isinstance(proposal.get("duration_rounds"), int) else 1,
        }

    for resp in raw.get("responses") or []:
        if isinstance(resp, dict) and resp.get("proposal_id") and resp.get("decision") in ("accept", "reject"):
            result["responses"].append({"proposal_id": resp["proposal_id"], "decision": resp["decision"]})

    return result


VALID_ACTIONS = {"GIVE", "TAKE", "HOLD"}


def validate_decision_output(raw: Optional[dict]) -> Dict[str, Any]:
    """Strict: raises ValidationError on anything ambiguous. The caller
    (agents/agent.py) is responsible for retrying or falling back to HOLD -
    this function's job is only to say clearly whether the output was usable."""
    if not isinstance(raw, dict):
        raise ValidationError("Decision output was not a JSON object.")

    action = raw.get("action")
    if action not in VALID_ACTIONS:
        raise ValidationError(f"'action' must be one of {VALID_ACTIONS}, got {action!r}.")

    if action == "HOLD":
        return {"action": "HOLD", "target": None, "amount": None,
                "reasoning": raw.get("reasoning")}

    target = raw.get("target")
    amount = raw.get("amount")
    if not isinstance(target, str) or not target:
        raise ValidationError(f"'{action}' requires a valid string 'target'.")
    if not isinstance(amount, int) or amount <= 0:
        raise ValidationError(f"'{action}' requires a positive integer 'amount', got {amount!r}.")

    return {"action": action, "target": target, "amount": amount, "reasoning": raw.get("reasoning")}
