"""
Validates the deterministic engine against the rules spec using scripted
(hardcoded) agent decisions - no LLM involved. This proves the engine is
correct BEFORE any model variability enters the picture (build order,
Phase 3 plan, Section 6, step 1).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Game, ActionType


def run_message_phase_noop(game, round_num, log):
    """Helper: run an empty message phase (no messages/proposals) for rounds
    that don't need to exercise that part of the engine."""
    game.process_message_subround(round_num, {aid: {} for aid in game.agents}, log)
    game.finalize_message_phase()


def test_basic_give():
    game = Game(["A", "B"], starting_tokens=10, max_rounds=1, seed=1)
    log = game.start_round()
    run_message_phase_noop(game, 1, log)
    game.run_decision_phase(1, {
        "A": {"action": "GIVE", "target": "B", "amount": 3},
        "B": {"action": "HOLD"},
    }, log)
    assert game.agents["A"].balance == 7, game.agents["A"].balance
    assert game.agents["B"].balance == 13, game.agents["B"].balance
    print("test_basic_give PASSED")


def test_basic_take():
    game = Game(["A", "B"], starting_tokens=10, max_rounds=1, seed=1)
    log = game.start_round()
    run_message_phase_noop(game, 1, log)
    game.run_decision_phase(1, {
        "A": {"action": "TAKE", "target": "B", "amount": 4},
        "B": {"action": "HOLD"},
    }, log)
    assert game.agents["A"].balance == 14, game.agents["A"].balance
    assert game.agents["B"].balance == 6, game.agents["B"].balance
    print("test_basic_take PASSED")


def test_contested_take_and_alliance():
    # A and C both take from B (10 tokens). Requested: A wants 6, C wants 6
    # (total 12 > 10 available) -> proportional split: each gets 5.
    # Since 2+ agents took from the same target, an alliance bonus applies.
    game = Game(["A", "B", "C"], starting_tokens=10, max_rounds=1, alliance_bonus=1, seed=1)
    log = game.start_round()
    run_message_phase_noop(game, 1, log)
    game.run_decision_phase(1, {
        "A": {"action": "TAKE", "target": "B", "amount": 6},
        "C": {"action": "TAKE", "target": "B", "amount": 6},
        "B": {"action": "HOLD"},
    }, log)
    # proportional: 6/12*10 = 5 each -> A=15, C=15, B=0 before alliance bonus
    # alliance bonus: B has 0 left, so no bonus can be paid (capped)
    assert game.agents["A"].balance == 15, game.agents["A"].balance
    assert game.agents["C"].balance == 15, game.agents["C"].balance
    assert game.agents["B"].balance == 0, game.agents["B"].balance
    assert len(log.alliances) == 1
    assert set(log.alliances[0]["allies"]) == {"A", "C"}
    print("test_contested_take_and_alliance PASSED")


def test_alliance_bonus_with_leftover_balance():
    # B has more balance than requested, so allies get their full take PLUS
    # the alliance bonus on top, sourced from B's remaining balance.
    game = Game(["A", "B", "C"], starting_tokens=20, max_rounds=1, alliance_bonus=1, seed=1)
    log = game.start_round()
    run_message_phase_noop(game, 1, log)
    game.run_decision_phase(1, {
        "A": {"action": "TAKE", "target": "B", "amount": 3},
        "C": {"action": "TAKE", "target": "B", "amount": 3},
        "B": {"action": "HOLD"},
    }, log)
    # B: 20 - 3 - 3 = 14 remaining, enough for 1 bonus each -> B loses 2 more -> 12
    assert game.agents["B"].balance == 12, game.agents["B"].balance
    assert game.agents["A"].balance == 24, game.agents["A"].balance
    assert game.agents["C"].balance == 24, game.agents["C"].balance
    print("test_alliance_bonus_with_leftover_balance PASSED")


def test_pact_and_betrayal():
    game = Game(["A", "B"], starting_tokens=10, max_rounds=3, seed=1)

    # Round 1: A proposes a non-aggression pact to B, B accepts.
    log1 = game.start_round()
    game.process_message_subround(1, {
        "A": {"proposal": {"to": "B", "pact_type": "non_aggression",
                            "terms": "Won't take from each other for 2 rounds",
                            "duration_rounds": 2}},
        "B": {},
    }, log1)
    proposal_id = log1.proposals[0].proposal_id
    game.process_message_subround(1, {
        "B": {"responses": [{"proposal_id": proposal_id, "decision": "accept"}]},
    }, log1)
    game.finalize_message_phase()
    game.run_decision_phase(1, {"A": {"action": "HOLD"}, "B": {"action": "HOLD"}}, log1)
    assert len(game.pacts) == 1
    assert game.pacts[0].expires_after_round == 2  # formed round 1, duration 2 -> covers rounds 1-2

    # Round 2: A betrays the pact by taking from B anyway.
    log2 = game.start_round()
    run_message_phase_noop(game, 2, log2)
    game.run_decision_phase(2, {
        "A": {"action": "TAKE", "target": "B", "amount": 4},
        "B": {"action": "HOLD"},
    }, log2)
    assert len(log2.betrayals) == 1, log2.betrayals
    assert log2.betrayals[0]["betrayer"] == "A"
    assert log2.betrayals[0]["victim"] == "B"
    assert game.pacts[0].broken is True
    print("test_pact_and_betrayal PASSED")


def test_malformed_action_falls_back_to_hold():
    game = Game(["A", "B"], starting_tokens=10, max_rounds=1, seed=1)
    log = game.start_round()
    run_message_phase_noop(game, 1, log)
    # Missing amount, bogus target, negative amount - all should safely become HOLD
    game.run_decision_phase(1, {
        "A": {"action": "TAKE", "target": "NOBODY", "amount": 5},
        "B": {"action": "GIVE", "target": "A", "amount": -3},
    }, log)
    assert game.agents["A"].balance == 10
    assert game.agents["B"].balance == 10
    print("test_malformed_action_falls_back_to_hold PASSED")


def test_bankruptcy_ends_game():
    game = Game(["A", "B"], starting_tokens=5, max_rounds=10, seed=1)
    log = game.start_round()
    run_message_phase_noop(game, 1, log)
    game.run_decision_phase(1, {
        "A": {"action": "TAKE", "target": "B", "amount": 5},
        "B": {"action": "HOLD"},
    }, log)
    assert game.agents["B"].balance == 0
    assert game.is_over() is True
    assert game.winner() == "A"
    print("test_bankruptcy_ends_game PASSED")


def test_max_rounds_ends_game_and_picks_highest_balance():
    game = Game(["A", "B"], starting_tokens=10, max_rounds=2, seed=1)
    for r in range(2):
        log = game.start_round()
        run_message_phase_noop(game, r + 1, log)
        game.run_decision_phase(r + 1, {"A": {"action": "HOLD"}, "B": {"action": "HOLD"}}, log)
    assert game.is_over() is True
    assert game.winner() in ("A", "B")  # tied, either is acceptable
    print("test_max_rounds_ends_game_and_picks_highest_balance PASSED")


def test_take_resolution_is_order_independent():
    # A and B each try to take 5 from the other, but each only has 3 tokens.
    # This is the exact shape of bug that showed up in a real game: without a
    # frozen balance snapshot, whichever target gets resolved SECOND sees an
    # already-inflated live balance for the other agent, letting them take
    # more than intended - making the result depend on dict iteration order.
    # With the fix, the result must be identical regardless of order.
    def run_with_order(decision_dict):
        game = Game(["A", "B"], starting_tokens=3, max_rounds=1, seed=1)
        log = game.start_round()
        run_message_phase_noop(game, 1, log)
        game.run_decision_phase(1, decision_dict, log)
        return game.get_public_balances()

    order_1 = {
        "A": {"action": "TAKE", "target": "B", "amount": 5},
        "B": {"action": "TAKE", "target": "A", "amount": 5},
    }
    order_2 = {
        "B": {"action": "TAKE", "target": "A", "amount": 5},
        "A": {"action": "TAKE", "target": "B", "amount": 5},
    }
    result_1 = run_with_order(order_1)
    result_2 = run_with_order(order_2)
    assert result_1 == result_2, f"order dependency bug: {result_1} vs {result_2}"
    # Symmetric mutual takes of equal, unaffordable amounts should net out evenly.
    assert result_1 == {"A": 3, "B": 3}, result_1
    print("test_take_resolution_is_order_independent PASSED")


if __name__ == "__main__":
    test_basic_give()
    test_basic_take()
    test_contested_take_and_alliance()
    test_alliance_bonus_with_leftover_balance()
    test_pact_and_betrayal()
    test_malformed_action_falls_back_to_hold()
    test_bankruptcy_ends_game()
    test_max_rounds_ends_game_and_picks_highest_balance()
    test_take_resolution_is_order_independent()
    print("\nAll engine tests passed.")
