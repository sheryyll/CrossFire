"""
Runs many games across model/baseline matchups and persists each one, per
Evaluation Spec Section 4 (round-robin tournament, seat rotation to cancel
position bias, fixed settings logged per game).
"""
from typing import List, Dict, Any, Optional
import itertools

from engine import Game
from agents import LLMAgent
from inference import LLMClient
from runner import run_game
from .baseline_agents import BASELINE_AGENT_CLASSES
from .persistence import save_game_log

AGENT_SEAT_NAMES = ["A", "B", "C", "D", "E", "F"]  # extend if you ever need more seats


def _rotate(seq: list, n: int) -> list:
    n = n % len(seq)
    return seq[n:] + seq[:n]


def build_seat_rotations(participants: List[Dict[str, Any]], num_seats: int) -> List[Dict[str, Any]]:
    """`participants` is a list of dicts, each either
    {"kind": "llm", "provider":..., "model":..., "persona":...} or
    {"kind": "baseline", "baseline": "always_defect"}.
    Requires len(participants) == num_seats. Returns one seat_assignment dict
    per rotation (cyclic shift), so across all rotations every participant
    plays every seat position exactly once - this is what cancels out any
    inherent advantage of acting earlier/later in the resolution order."""
    assert len(participants) == num_seats, (
        f"Got {len(participants)} participants but {num_seats} seats - "
        f"for MVP these must match 1:1 (extend later for larger fields)."
    )
    seat_ids = AGENT_SEAT_NAMES[:num_seats]
    rotations = []
    for i in range(num_seats):
        rotated_participants = _rotate(participants, i)
        rotations.append(dict(zip(seat_ids, rotated_participants)))
    return rotations


def _build_agent_for_seat(agent_id: str, participant: Dict[str, Any]):
    if participant["kind"] == "llm":
        client = LLMClient(
            provider=participant["provider"], model=participant["model"],
            temperature=participant.get("temperature", 0.7),
        )
        return LLMAgent(agent_id, participant.get("persona", "a strategic player"), client)
    elif participant["kind"] == "baseline":
        cls = BASELINE_AGENT_CLASSES[participant["baseline"]]
        return cls(agent_id)
    raise ValueError(f"Unknown participant kind: {participant['kind']!r}")


def _seat_assignment_metadata(seat_assignment: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    meta = {}
    for agent_id, participant in seat_assignment.items():
        if participant["kind"] == "llm":
            meta[agent_id] = {"provider": participant["provider"], "model": participant["model"]}
        else:
            meta[agent_id] = {"baseline": participant["baseline"]}
    return meta


def run_one_game(seat_assignment: Dict[str, Dict[str, Any]], game_params: Dict[str, Any],
                  seed: int, message_subrounds: int = 2, verbose: bool = False) -> Game:
    agent_ids = list(seat_assignment.keys())
    game = Game(agent_ids, starting_tokens=game_params.get("starting_tokens", 10),
                max_rounds=game_params.get("max_rounds", 10),
                alliance_bonus=game_params.get("alliance_bonus", 1), seed=seed)

    agents = {}
    for agent_id, participant in seat_assignment.items():
        agent = _build_agent_for_seat(agent_id, participant)
        if hasattr(agent, "bind_game"):
            agent.bind_game(game)
        agents[agent_id] = agent

    run_game(game, agents, message_subrounds=message_subrounds, verbose=verbose)
    return game


def run_tournament(participants: List[Dict[str, Any]], game_params: Optional[Dict[str, Any]] = None,
                    repeats_per_rotation: int = 1, output_dir: str = "logs",
                    message_subrounds: int = 2, verbose: bool = False,
                    base_seed: int = 1000) -> List[str]:
    """Plays every seat rotation of `participants`, `repeats_per_rotation`
    times each, saving one JSON log per game. Returns the list of saved
    file paths. Each repeat uses a different seed (logged), so results are
    both reproducible (given the seed) and not all identical."""
    game_params = game_params or {}
    num_seats = len(participants)
    rotations = build_seat_rotations(participants, num_seats)

    saved_paths = []
    seed_counter = itertools.count(base_seed)
    for rotation_index, seat_assignment in enumerate(rotations):
        for repeat_index in range(repeats_per_rotation):
            seed = next(seed_counter)
            if verbose:
                print(f"\n--- Tournament game: rotation {rotation_index}, "
                      f"repeat {repeat_index}, seed {seed} ---")
            game = run_one_game(seat_assignment, game_params, seed=seed,
                                 message_subrounds=message_subrounds, verbose=verbose)
            metadata = {
                "seat_assignment": _seat_assignment_metadata(seat_assignment),
                "rotation_index": rotation_index,
                "repeat_index": repeat_index,
            }
            path = save_game_log(game, metadata, output_dir)
            saved_paths.append(path)
            if verbose:
                print(f"Saved: {path} | winner: {game.winner()} | "
                      f"final: {game.get_public_balances()}")
    return saved_paths
