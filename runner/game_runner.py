"""
Drives one full game from round 1 to the end condition, using real LLMAgent
instances against the real Game engine. This is the first place all three
layers (engine, inference, agents) actually get wired together.
"""
from typing import Dict
from engine import Game
from agents import LLMAgent
from .context import build_context


def _record_reliability(log, agent_id: str, phase: str, outcome) -> None:
    """Every retry/fallback gets logged, regardless of game outcome - this is
    the raw data behind the 'structured-output validity rate per model'
    metric (Evaluation Spec Section 3C). Successful first-try calls are NOT
    logged as failures here (there's nothing to flag), but their timing is
    still folded into per-round latency if you want to extend this later."""
    if outcome.used_retry or outcome.used_fallback:
        log.parse_failures.append({
            "agent": agent_id,
            "phase": phase,
            "used_retry": outcome.used_retry,
            "used_fallback": outcome.used_fallback,
            "error": outcome.error,
            "latency_seconds": outcome.total_latency_seconds,
        })


def run_game(game: Game, llm_agents: Dict[str, LLMAgent], message_subrounds: int = 2,
             verbose: bool = True) -> Game:
    """Runs `game` to completion, mutating it in place, and returns it.
    `llm_agents` keys must exactly match the agent_ids the game was built with."""
    while not game.is_over():
        log = game.start_round()
        if verbose:
            print(f"\n=== Round {game.current_round}/{game.max_rounds} "
                  f"(balances: {game.get_public_balances()}) ===")

        # --- Message phase: run the configured number of exchange sub-rounds ---
        for sub in range(message_subrounds):
            already_printed_msgs = len(log.messages)
            already_printed_proposals = len(log.proposals)
            pact_count_before = len(game.pacts)

            outputs = {}
            for agent_id, agent in llm_agents.items():
                context = build_context(game, agent_id, log, phase="message")
                outcome = agent.decide_messages(context)
                outputs[agent_id] = outcome.parsed
                log.calls.append({"agent": agent_id, "phase": "message"})
                _record_reliability(log, agent_id, "message", outcome)
            game.process_message_subround(game.current_round, outputs, log)

            if verbose:
                for m in log.messages[already_printed_msgs:]:
                    tag = "BROADCAST" if m.is_broadcast else f"DM->{m.to_agent}"
                    print(f"  [{tag}] {m.from_agent}: {m.text}")
                for p in log.proposals[already_printed_proposals:]:
                    print(f"  [PROPOSAL {p.proposal_id}] {p.from_agent} -> {p.to_agent}: "
                          f"\"{p.terms_text}\" ({p.duration_rounds} round(s))")
                if len(game.pacts) > pact_count_before:
                    for pact in game.pacts[pact_count_before:]:
                        print(f"  [PACT FORMED {pact.pact_id}] {pact.agent_a} <-> {pact.agent_b}: "
                              f"active through round {pact.expires_after_round}")
        game.finalize_message_phase()
        if verbose:
            for p in log.proposals:
                if p.status.value == "expired":
                    print(f"  [PROPOSAL {p.proposal_id} LAPSED, never accepted] "
                          f"{p.from_agent} -> {p.to_agent}")

        # --- Decision phase: exactly one action per agent ---
        raw_decisions = {}
        for agent_id, agent in llm_agents.items():
            context = build_context(game, agent_id, log, phase="decision")
            outcome = agent.decide_action(context)
            raw_decisions[agent_id] = outcome.parsed
            log.calls.append({"agent": agent_id, "phase": "decision"})
            _record_reliability(log, agent_id, "decision", outcome)
        game.run_decision_phase(game.current_round, raw_decisions, log)

        if verbose:
            for d in log.decisions:
                print(f"  [ACTION] {d.agent_id}: {d.action.value}"
                      + (f" {d.amount} -> {d.target}" if d.target else ""))
            for a in log.alliances:
                print(f"  [ALLIANCE] {' + '.join(a['allies'])} vs {a['target']} "
                      f"(+{a['bonus_each']} each)")
            for b in log.betrayals:
                print(f"  [BETRAYAL] {b['betrayer']} broke pact with {b['victim']}")
            if log.parse_failures:
                print(f"  [RELIABILITY] {len(log.parse_failures)} retry/fallback event(s) this round")
            print(f"  Balances after: {game.get_public_balances()}")

    if verbose:
        print(f"\n=== GAME OVER after {game.current_round} rounds ===")
        print(f"Final balances: {game.get_public_balances()}")
        print(f"Winner: {game.winner()}")

    return game
