"""Wall-unaware heuristic baseline for Taxi-v4.

Logic:
    1. If the passenger is not in the taxi:
         - If the taxi is on the passenger's pickup tile: PICKUP.
         - Otherwise: move toward the pickup tile (greedy by Manhattan distance).
    2. If the passenger is in the taxi:
         - If the taxi is on the destination tile: DROPOFF.
         - Otherwise: move toward the destination tile.

The heuristic does not know about walls, so it sometimes bumps into them
(no penalty, just a wasted step). It still completes most episodes -- but
slowly, which is exactly the bar an RL agent should clear.
"""
from __future__ import annotations

import random

from ..env import LOCATIONS, decode_state
from .base import Agent

# Movement actions and their (drow, dcol) effects.
_MOVE_DELTAS = {
    0: (1, 0),    # south
    1: (-1, 0),   # north
    2: (0, 1),    # east
    3: (0, -1),   # west
}
PICKUP, DROPOFF = 4, 5


class HeuristicAgent(Agent):
    n_actions = 6

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_action(self, state: int, greedy: bool = False) -> int:
        row, col, pass_loc, dest = decode_state(state)

        if pass_loc == 4:  # passenger already in taxi -> head to destination
            target = LOCATIONS[dest]
            if (row, col) == target:
                return DROPOFF
        else:               # head to pickup tile
            target = LOCATIONS[pass_loc]
            if (row, col) == target:
                return PICKUP

        return self._greedy_move(row, col, *target)

    def _greedy_move(self, r: int, c: int, tr: int, tc: int) -> int:
        best_actions: list[int] = []
        best_dist = None
        for action, (dr, dc) in _MOVE_DELTAS.items():
            nr, nc = max(0, min(4, r + dr)), max(0, min(4, c + dc))
            dist = abs(nr - tr) + abs(nc - tc)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_actions = [action]
            elif dist == best_dist:
                best_actions.append(action)
        return self.rng.choice(best_actions)
