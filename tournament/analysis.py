"""
Aggregates saved game logs (persistence.py) into the per-model metrics
defined in the Evaluation Spec (Section 3). Deliberately standard-library
only for now - a pandas/plotly notebook can consume this same data later
for richer analysis and charts (Evaluation Spec Section 5), but this gives
you a real leaderboard with zero extra dependencies.
"""
import glob
import os
from collections import defaultdict
from typing import Dict, Any, List

from .persistence import load_game_log


def _model_key(seat_info: Dict[str, Any]) -> str:
    if "model" in seat_info:
        return f"{seat_info['provider']}/{seat_info['model']}"
    return f"baseline/{seat_info['baseline']}"


def build_leaderboard(log_dir: str) -> Dict[str, Dict[str, Any]]:
    stats = defaultdict(lambda: {
        "games_played": 0, "wins": 0, "final_balance_sum": 0,
        "bankruptcies": 0, "betrayals_committed": 0, "betrayals_suffered": 0,
        "alliances_participated": 0, "decisions_made": 0,
        "total_llm_calls": 0,  # BOTH phases - the correct denominator for reliability rate
        "retry_or_fallback_events": 0,
    })

    paths = sorted(glob.glob(os.path.join(log_dir, "*.json")))
    for path in paths:
        data = load_game_log(path)
        seat_assignment = data["metadata"]["seat_assignment"]
        winner_seat = data["winner"]

        for agent_id, seat_info in seat_assignment.items():
            model = _model_key(seat_info)
            s = stats[model]
            s["games_played"] += 1
            s["final_balance_sum"] += data["final_balances"].get(agent_id, 0)
            if agent_id == winner_seat:
                s["wins"] += 1
            if data["final_balances"].get(agent_id, 0) == 0:
                s["bankruptcies"] += 1

        for round_data in data["rounds"]:
            for b in round_data["betrayals"]:
                stats[_model_key(seat_assignment[b["betrayer"]])]["betrayals_committed"] += 1
                stats[_model_key(seat_assignment[b["victim"]])]["betrayals_suffered"] += 1
            for a in round_data["alliances"]:
                for ally in a["allies"]:
                    stats[_model_key(seat_assignment[ally])]["alliances_participated"] += 1
            for d in round_data["decisions"]:
                stats[_model_key(seat_assignment[d["agent"]])]["decisions_made"] += 1
            # round_data.get(...) for backward compatibility with logs saved before
            # the "calls" field existed - those will just show total_llm_calls=0.
            for c in round_data.get("calls", []):
                stats[_model_key(seat_assignment[c["agent"]])]["total_llm_calls"] += 1
            for f in round_data["parse_failures"]:
                stats[_model_key(seat_assignment[f["agent"]])]["retry_or_fallback_events"] += 1

    # Derive rates now that totals are complete.
    leaderboard = {}
    for model, s in stats.items():
        games = max(s["games_played"], 1)
        total_calls = max(s["total_llm_calls"], 1)
        leaderboard[model] = {
            **s,
            "win_rate": round(s["wins"] / games, 3),
            "avg_final_balance": round(s["final_balance_sum"] / games, 2),
            "bankruptcy_rate": round(s["bankruptcies"] / games, 3),
            # Correct denominator: ALL calls (message + decision phase), not just
            # decision-phase actions - using decisions_made here undercounts the
            # denominator and can push this rate above 100%, which happened in
            # testing before this fix.
            "reliability_issue_rate": round(s["retry_or_fallback_events"] / total_calls, 3),
        }
    return leaderboard


def print_leaderboard(log_dir: str) -> None:
    leaderboard = build_leaderboard(log_dir)
    if not leaderboard:
        print(f"No game logs found in {log_dir}")
        return
    header = f"{'MODEL':<35}{'GAMES':>7}{'WIN%':>8}{'AVG BAL':>10}{'BANKRUPT%':>11}{'BETRAY':>8}{'RELIAB ISSUE%':>15}"
    print(header)
    print("-" * len(header))
    for model, s in sorted(leaderboard.items(), key=lambda kv: -kv[1]["win_rate"]):
        print(f"{model:<35}{s['games_played']:>7}{s['win_rate']*100:>7.1f}%"
              f"{s['avg_final_balance']:>10.2f}{s['bankruptcy_rate']*100:>10.1f}%"
              f"{s['betrayals_committed']:>8}{s['reliability_issue_rate']*100:>14.1f}%")
