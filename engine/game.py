"""
Deterministic game engine for Shifting Alliances.

Implements: rounds, message phase (messages/proposals/accept/reject),
decision phase (GIVE/TAKE/HOLD), resolution rules, alliance detection,
pact tracking, and betrayal detection.

The engine NEVER calls an LLM. It only accepts already-structured input
and produces already-structured output/logs. This is deliberate: it lets
us unit test the entire rules spec with scripted fake agents before any
model variability enters the picture.
"""
import itertools
import random
from typing import Dict, List, Optional

from .models import (
    Agent, Proposal, Pact, Message, DecisionAction, RoundLog,
    ActionType, PactType, ProposalStatus,
)


class InvalidActionError(Exception):
    """Raised when a caller passes a structurally invalid action.
    The Agent layer (Phase 2) is responsible for catching malformed LLM
    output before it ever reaches the engine - the engine assumes valid
    Python objects, not raw LLM text."""
    pass


class Game:
    def __init__(
        self,
        agent_ids: List[str],
        display_names: Optional[Dict[str, str]] = None,
        starting_tokens: int = 10,
        max_rounds: int = 6,
        alliance_bonus: int = 1,
        seed: Optional[int] = None,
    ):
        display_names = display_names or {aid: aid for aid in agent_ids}
        self.agents: Dict[str, Agent] = {
            aid: Agent(agent_id=aid, display_name=display_names[aid], balance=starting_tokens)
            for aid in agent_ids
        }
        self.max_rounds = max_rounds
        self.alliance_bonus = alliance_bonus
        self.rng = random.Random(seed)
        self.seed = seed

        self.current_round = 0
        self.pacts: List[Pact] = []
        self._pact_counter = itertools.count(1)
        self._proposal_counter = itertools.count(1)
        self.current_round_proposals: List[Proposal] = []
        self.rounds_log: List[RoundLog] = []

    
    def get_public_balances(self) -> Dict[str, int]:
        return {aid: a.balance for aid, a in self.agents.items()}

    def get_active_pacts_for(self, agent_id: str) -> List[Pact]:
        """Per rules spec Section 7: an agent knows its OWN pacts, not others'."""
        return [
            p for p in self.pacts
            if not p.broken
            and p.formed_round <= self.current_round <= p.expires_after_round
            and agent_id in (p.agent_a, p.agent_b)
        ]

    def is_pact_active(self, pact: Pact, round_num: int) -> bool:
        return (not pact.broken) and pact.formed_round <= round_num <= pact.expires_after_round

    def is_over(self) -> bool:
        if self.current_round >= self.max_rounds:
            return True
        alive = [a for a in self.agents.values() if a.balance > 0]
        return len(alive) <= 1

    def winner(self) -> Optional[str]:
        if not self.is_over():
            return None
        return max(self.agents.values(), key=lambda a: a.balance).agent_id

    def process_message_subround(self, round_num: int, outputs: Dict[str, dict],
                                  log: RoundLog) -> None:
        """
        outputs: agent_id -> {
            "broadcast": str | None,
            "direct_messages": [{"to": agent_id, "text": str}, ...],
            "proposal": {"to": agent_id, "pact_type": str, "terms": str,
                         "duration_rounds": int} | None,
            "responses": [{"proposal_id": str, "decision": "accept"|"reject"}, ...]
        }
        """
        # 1. Log messages
        for agent_id, out in outputs.items():
            if out.get("broadcast"):
                log.messages.append(Message(round_num, agent_id, None, out["broadcast"], True))
            for dm in out.get("direct_messages", []) or []:
                if dm.get("to") not in self.agents or dm.get("to") == agent_id:
                    continue  # invalid target, silently dropped per decision-contract edge cases
                log.messages.append(Message(round_num, agent_id, dm["to"], dm["text"], False))

        # 2. Register new proposals
        for agent_id, out in outputs.items():
            prop = out.get("proposal")
            if not prop:
                continue
            to_agent = prop.get("to")
            if to_agent not in self.agents or to_agent == agent_id:
                continue  # can't propose to self or nonexistent agent
            try:
                pact_type = PactType(prop.get("pact_type", PactType.NON_AGGRESSION.value))
            except ValueError:
                pact_type = PactType.NON_AGGRESSION
            proposal = Proposal(
                proposal_id=f"P{next(self._proposal_counter)}",
                from_agent=agent_id,
                to_agent=to_agent,
                pact_type=pact_type,
                terms_text=prop.get("terms", ""),
                duration_rounds=max(1, int(prop.get("duration_rounds", 1))),
                created_round=round_num,
            )
            self.current_round_proposals.append(proposal)
            log.proposals.append(proposal)

        # 3. Process accept/reject responses
        by_id = {p.proposal_id: p for p in self.current_round_proposals}
        for agent_id, out in outputs.items():
            for resp in out.get("responses", []) or []:
                proposal = by_id.get(resp.get("proposal_id"))
                if proposal is None or proposal.to_agent != agent_id:
                    continue  # unknown or not addressed to this agent - ignored per spec
                if proposal.status != ProposalStatus.PENDING:
                    continue  # duplicate response - first one wins
                decision = resp.get("decision")
                if decision == "accept":
                    proposal.status = ProposalStatus.ACCEPTED
                    pact = Pact(
                        pact_id=f"PACT{next(self._pact_counter)}",
                        agent_a=proposal.from_agent,
                        agent_b=proposal.to_agent,
                        pact_type=proposal.pact_type,
                        terms_text=proposal.terms_text,
                        formed_round=round_num,
                        expires_after_round=round_num + proposal.duration_rounds - 1,
                    )
                    self.pacts.append(pact)
                elif decision == "reject":
                    proposal.status = ProposalStatus.REJECTED

    def finalize_message_phase(self) -> None:
        """Any proposal still pending at the end of the message phase lapses."""
        for p in self.current_round_proposals:
            if p.status == ProposalStatus.PENDING:
                p.status = ProposalStatus.EXPIRED
        self.current_round_proposals = []


    def run_decision_phase(self, round_num: int, raw_decisions: Dict[str, dict],
                            log: RoundLog) -> RoundLog:
        log.balances_before = self.get_public_balances()

        decisions: List[DecisionAction] = []
        for agent_id, d in raw_decisions.items():
            action = self._safe_action(d)
            decisions.append(DecisionAction(
                agent_id=agent_id,
                action=action["action"],
                target=action.get("target"),
                amount=action.get("amount"),
                reasoning=d.get("reasoning"),
            ))
        log.decisions = decisions

        # Step 1: apply all GIVE actions (only depend on giver's own balance)
        for d in decisions:
            if d.action == ActionType.GIVE:
                amount = max(0, min(d.amount or 0, self.agents[d.agent_id].balance))
                if d.target in self.agents and amount > 0:
                    self.agents[d.agent_id].balance -= amount
                    self.agents[d.target].balance += amount

        # Step 2: apply all TAKE actions, proportionally splitting contested targets.
        # IMPORTANT: all allocation decisions use a SNAPSHOT of balances taken
        # right after GIVEs, not the live (already-mutating) balances. Without
        # this, resolving target X before target Y could let Y's allocation
        # be computed against a balance X's resolution just changed - making
        # the outcome depend on dict iteration order, which breaks the
        # "simultaneous decisions" design intent (rules spec Section 4).
        takes_by_target: Dict[str, List[DecisionAction]] = {}
        for d in decisions:
            if d.action == ActionType.TAKE and d.target in self.agents and d.target != d.agent_id:
                takes_by_target.setdefault(d.target, []).append(d)

        balances_snapshot = {aid: a.balance for aid, a in self.agents.items()}

        alliance_events = []
        for target_id, takers in takes_by_target.items():
            target_balance = balances_snapshot[target_id]
            total_requested = sum(max(0, t.amount or 0) for t in takers)
            if total_requested <= 0 or target_balance <= 0:
                continue
            if total_requested <= target_balance:
                allocations = {t.agent_id: max(0, t.amount or 0) for t in takers}
            else:
                allocations = {
                    t.agent_id: (max(0, t.amount or 0) * target_balance) // total_requested
                    for t in takers
                }
            for agent_id, amt in allocations.items():
                if amt <= 0:
                    continue
                self.agents[target_id].balance -= amt
                self.agents[agent_id].balance += amt

            # Alliance detection: 2+ agents taking from the same target this round.
            # This bonus IS allowed to use the live post-take balance of the
            # target, since it only depends on this one target's own state,
            # not on some other target's resolution - no cross-target ordering
            # issue here.
            if len(takers) >= 2:
                remaining = self.agents[target_id].balance
                per_ally = min(self.alliance_bonus, remaining // len(takers)) if remaining > 0 else 0
                allies = [t.agent_id for t in takers]
                if per_ally > 0:
                    for ally in allies:
                        self.agents[target_id].balance -= per_ally
                        self.agents[ally].balance += per_ally
                alliance_events.append({"allies": allies, "target": target_id, "bonus_each": per_ally})
        log.alliances = alliance_events

        # Step 3: betrayal detection - any TAKE against an active non-aggression pact partner
        betrayals = []
        for d in decisions:
            if d.action != ActionType.TAKE or d.target not in self.agents:
                continue
            for pact in self.pacts:
                if pact.pact_type != PactType.NON_AGGRESSION or pact.broken:
                    continue
                if not self.is_pact_active(pact, round_num):
                    continue
                pair = {pact.agent_a, pact.agent_b}
                if pair == {d.agent_id, d.target}:
                    pact.broken = True
                    pact.broken_by = d.agent_id
                    pact.broken_round = round_num
                    betrayals.append({
                        "betrayer": d.agent_id, "victim": d.target, "pact_id": pact.pact_id
                    })
        log.betrayals = betrayals

        log.balances_after = self.get_public_balances()
        return log

    def _safe_action(self, d: dict) -> dict:
        """Validate a raw decision dict, falling back to HOLD on anything malformed.
        This is the engine-side half of the reliability contract - Phase 2's
        agent layer does its own retry/parse logic BEFORE calling this, but the
        engine still never trusts its input blindly."""
        try:
            action = ActionType(d.get("action", "HOLD"))
        except ValueError:
            return {"action": ActionType.HOLD}

        if action == ActionType.HOLD:
            return {"action": ActionType.HOLD}

        target = d.get("target")
        amount = d.get("amount")
        if target not in self.agents or not isinstance(amount, int) or amount <= 0:
            return {"action": ActionType.HOLD}
        return {"action": action, "target": target, "amount": amount}


    def start_round(self) -> RoundLog:
        self.current_round += 1
        log = RoundLog(round_num=self.current_round)
        self.rounds_log.append(log)
        return log

    def get_log(self) -> List[RoundLog]:
        return self.rounds_log
