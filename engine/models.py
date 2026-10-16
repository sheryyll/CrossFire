"""
Core data models for Shifting Alliances.

These are deliberately plain dataclasses with no LLM awareness at all -
the engine only ever deals with structured actions, never natural language.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


class ActionType(str, Enum):
    GIVE = "GIVE"
    TAKE = "TAKE"
    HOLD = "HOLD"


class PactType(str, Enum):
    # MVP only implements one structured pact type. Free-text "terms" can
    # still be stored for flavor/logging, but only this type is mechanically
    # enforced (i.e. can trigger a betrayal event).
    NON_AGGRESSION = "non_aggression"


class ProposalStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Agent:
    agent_id: str
    display_name: str
    balance: int


@dataclass
class Proposal:
    proposal_id: str
    from_agent: str
    to_agent: str
    pact_type: PactType
    terms_text: str
    duration_rounds: int
    created_round: int
    status: ProposalStatus = ProposalStatus.PENDING


@dataclass
class Pact:
    pact_id: str
    agent_a: str
    agent_b: str
    pact_type: PactType
    terms_text: str
    formed_round: int
    expires_after_round: int  # inclusive - pact is active through this round
    broken: bool = False
    broken_by: Optional[str] = None
    broken_round: Optional[int] = None


@dataclass
class Message:
    round_num: int
    from_agent: str
    to_agent: Optional[str]  # None means broadcast
    text: str
    is_broadcast: bool


@dataclass
class DecisionAction:
    agent_id: str
    action: ActionType
    target: Optional[str] = None
    amount: Optional[int] = None
    reasoning: Optional[str] = None


@dataclass
class RoundLog:
    round_num: int
    messages: List[Message] = field(default_factory=list)
    proposals: List[Proposal] = field(default_factory=list)
    decisions: List[DecisionAction] = field(default_factory=list)
    balances_before: Dict[str, int] = field(default_factory=dict)
    balances_after: Dict[str, int] = field(default_factory=dict)
    alliances: List[Dict] = field(default_factory=list)  # [{"allies": [...], "target": ...}]
    betrayals: List[Dict] = field(default_factory=list)  # [{"betrayer":..., "victim":..., "pact_id":...}]
    parse_failures: List[Dict] = field(default_factory=list)  # engineering/reliability metric
    calls: List[Dict] = field(default_factory=list)  # EVERY LLM call made this round, success or not -
    # this is the correct denominator for reliability rates; parse_failures alone is only the numerator.
