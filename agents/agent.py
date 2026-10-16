"""
LLMAgent ties together: prompts.py (what to ask) + an LLMClient (how to ask
it) + parsing.py (how to make sense of the answer), implementing the
reliability contract from Agent Decision Contract Section 6:

    1. Call the model.
    2. Try to parse/validate the response.
    3. If invalid, retry ONCE with a corrective prompt.
    4. If still invalid, fall back to a safe default (HOLD / no messages).
    5. Log every outcome - this is where the "structured-output validity
       rate per model" metric (Evaluation Spec Section 3C) comes from.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .parsing import extract_json, validate_message_output, validate_decision_output, ValidationError
from .prompts import build_system_prompt, build_message_phase_prompt, build_decision_phase_prompt, \
    build_correction_prompt

MESSAGE_SCHEMA_HINT = (
    '{"broadcast": "string or null", "direct_messages": [...], '
    '"proposal": {...} or null, "responses": [...]}'
)
DECISION_SCHEMA_HINT = (
    '{"action": "GIVE"|"TAKE"|"HOLD", "target": "agent_id or null", '
    '"amount": integer or null, "reasoning": "string"}'
)

SAFE_DEFAULT_MESSAGE_OUTPUT = {"broadcast": None, "direct_messages": [], "proposal": None, "responses": []}
SAFE_DEFAULT_DECISION_OUTPUT = {"action": "HOLD", "target": None, "amount": None, "reasoning": None}


@dataclass
class CallOutcome:
    parsed: Dict[str, Any]
    raw_text_attempts: list = field(default_factory=list)
    valid_on_first_try: bool = False
    used_retry: bool = False
    used_fallback: bool = False
    error: Optional[str] = None
    total_latency_seconds: float = 0.0


class LLMAgent:
    def __init__(self, agent_id: str, persona: str, llm_client):
        self.agent_id = agent_id
        self.persona = persona
        self.llm_client = llm_client
        self.system_prompt = build_system_prompt(agent_id, persona)

    def decide_messages(self, context: Dict[str, Any]) -> CallOutcome:
        prompt = build_message_phase_prompt(context)
        return self._call_with_retry(
            user_prompt=prompt,
            parse_fn=lambda text: validate_message_output(extract_json(text)),
            schema_hint=MESSAGE_SCHEMA_HINT,
            safe_default=SAFE_DEFAULT_MESSAGE_OUTPUT,
            # message output is always "valid" once it's a dict (validate_message_output
            # never raises) - the only real failure mode is unparseable JSON entirely.
            requires_nonempty_json=True,
        )

    def decide_action(self, context: Dict[str, Any]) -> CallOutcome:
        prompt = build_decision_phase_prompt(context)
        return self._call_with_retry(
            user_prompt=prompt,
            parse_fn=lambda text: validate_decision_output(extract_json(text)),
            schema_hint=DECISION_SCHEMA_HINT,
            safe_default=SAFE_DEFAULT_DECISION_OUTPUT,
            requires_nonempty_json=False,
        )

    def _call_with_retry(self, user_prompt: str, parse_fn, schema_hint: str,
                          safe_default: dict, requires_nonempty_json: bool) -> CallOutcome:
        outcome = CallOutcome(parsed=safe_default)
        current_prompt = user_prompt
        last_error = None

        for attempt in range(2):  # one initial try + one retry, per Section 6
            response = self.llm_client.generate(self.system_prompt, current_prompt)
            outcome.raw_text_attempts.append(response.text)
            outcome.total_latency_seconds += response.latency_seconds

            try:
                json_obj = extract_json(response.text)
                if requires_nonempty_json and json_obj is None:
                    raise ValidationError("No JSON object found in response.")
                parsed = parse_fn(response.text)
                outcome.parsed = parsed
                outcome.valid_on_first_try = (attempt == 0)
                outcome.used_retry = (attempt == 1)
                return outcome
            except ValidationError as e:
                last_error = str(e)
                current_prompt = build_correction_prompt(schema_hint, last_error)
                continue

        # Both attempts failed - safe fallback, per Section 6, step 4.
        outcome.used_fallback = True
        outcome.error = last_error
        outcome.parsed = safe_default
        return outcome
