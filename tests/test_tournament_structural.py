"""
Runs a full mini-tournament using ONLY the rule-based baseline agents - no
LLM client involved at all, so this validates the entire tournament +
persistence + analysis pipeline (seat rotation, saving, aggregation) with
zero network dependency, before ever spending real API calls on it.
"""
import sys, os, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tournament import run_tournament, build_leaderboard, load_game_log


def test_baseline_only_tournament_end_to_end():
    tmp_dir = tempfile.mkdtemp()
    try:
        participants = [
            {"kind": "baseline", "baseline": "always_defect"},
            {"kind": "baseline", "baseline": "always_cooperate"},
            {"kind": "baseline", "baseline": "tit_for_tat"},
        ]
        game_params = {"starting_tokens": 10, "max_rounds": 4, "alliance_bonus": 1}

        paths = run_tournament(
            participants, game_params=game_params,
            repeats_per_rotation=1, output_dir=tmp_dir,
            message_subrounds=1, verbose=False,
        )

        # 3 participants -> 3 seat rotations -> 3 games (1 repeat each)
        assert len(paths) == 3, paths
        for p in paths:
            assert os.path.exists(p)
            data = load_game_log(p)
            assert "final_balances" in data
            assert sum(data["final_balances"].values()) == 30  # tokens conserved

        leaderboard = build_leaderboard(tmp_dir)
        assert set(leaderboard.keys()) == {
            "baseline/always_defect", "baseline/always_cooperate", "baseline/tit_for_tat"
        }
        for model, stats in leaderboard.items():
            assert stats["games_played"] == 3, (model, stats)  # played once per rotation
            assert stats["decisions_made"] > 0

        # Sanity check on expected behavior: always_defect should win at least
        # as often as always_cooperate against these opponents, since it's
        # the only one actively accumulating tokens from others.
        assert leaderboard["baseline/always_defect"]["win_rate"] >= \
               leaderboard["baseline/always_cooperate"]["win_rate"]

        print("test_baseline_only_tournament_end_to_end PASSED")
        for model, stats in leaderboard.items():
            print(f"  {model}: {stats}")
    finally:
        shutil.rmtree(tmp_dir)


def test_reliability_rate_cannot_exceed_100_percent():
    """Regression test for a real bug found via a live run: reliability_issue_rate
    was computed as retry_or_fallback_events / decisions_made, but failures can
    happen during the MESSAGE phase too (which has far more calls per round
    than the decision phase), so the rate could exceed 1.0. It must now use
    total LLM calls (both phases) as the denominator."""
    import json
    tmp_dir = tempfile.mkdtemp()
    try:
        # Hand-crafted log: 1 agent, 1 round, 3 message-phase calls (2 failed)
        # + 1 decision-phase call (also failed) = 4 total calls, 3 failures.
        # Old buggy code would have divided 3 failures by 1 decision = 300%.
        fake_log = {
            "metadata": {"seat_assignment": {"A": {"provider": "groq", "model": "test-model"}}},
            "seed": 1, "max_rounds": 1, "final_balances": {"A": 10}, "winner": "A",
            "pacts": [],
            "rounds": [{
                "round": 1, "messages": [], "proposals": [],
                "decisions": [{"agent": "A", "action": "HOLD", "target": None, "amount": None, "reasoning": None}],
                "balances_before": {"A": 10}, "balances_after": {"A": 10},
                "alliances": [], "betrayals": [],
                "calls": [{"agent": "A", "phase": "message"}, {"agent": "A", "phase": "message"},
                          {"agent": "A", "phase": "decision"}],
                "parse_failures": [
                    {"agent": "A", "phase": "message", "used_retry": True, "used_fallback": False, "error": "x"},
                    {"agent": "A", "phase": "message", "used_retry": False, "used_fallback": True, "error": "y"},
                    {"agent": "A", "phase": "decision", "used_retry": True, "used_fallback": False, "error": "z"},
                ],
            }],
        }
        with open(os.path.join(tmp_dir, "fake_game.json"), "w") as f:
            json.dump(fake_log, f)

        leaderboard = build_leaderboard(tmp_dir)
        rate = leaderboard["groq/test-model"]["reliability_issue_rate"]
        assert rate <= 1.0, f"reliability rate must never exceed 100%, got {rate}"
        assert rate == 1.0, f"3 failures / 3 total calls should be exactly 1.0, got {rate}"
        print("test_reliability_rate_cannot_exceed_100_percent PASSED")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_baseline_only_tournament_end_to_end()
    test_reliability_rate_cannot_exceed_100_percent()
    print("\nTournament structural tests passed.")
