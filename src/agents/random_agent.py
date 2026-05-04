"""Uniform-random baseline."""
from __future__ import annotations

import random
from typing import Tuple

from .base import Agent


class RandomAgent(Agent):
    def __init__(self, n_actions: int = 2, seed: int | None = None) -> None:
        self.n_actions = n_actions
        self.rng = random.Random(seed)

    def select_action(self, state: Tuple, greedy: bool = False) -> int:
        return self.rng.randrange(self.n_actions)
