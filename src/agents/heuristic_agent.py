"""Hand-coded heuristic baseline.

Flaps when the bird is below (or about to fall below) the gap center.
Combines two signals:
    - dy_idx > center bin: bird already below gap center.
    - vy_idx > center bin: bird has downward velocity (gravity dominating).
Reacting to velocity, not just position, prevents the oscillation that
hurts a pure "flap when below center" rule.
"""
from __future__ import annotations

from typing import Tuple

from .base import Agent


class HeuristicAgent(Agent):
    def __init__(self, dy_bins: int = 15, vy_bins: int = 10) -> None:
        self.dy_center = dy_bins // 2
        self.vy_center = vy_bins // 2

    def select_action(self, state: Tuple, greedy: bool = False) -> int:
        _, dy_idx, vy_idx = state
        below_center = dy_idx >= self.dy_center
        falling = vy_idx >= self.vy_center
        return 1 if (below_center and falling) else 0
