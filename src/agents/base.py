"""Common interface for all agents."""
from __future__ import annotations

from typing import Any, Tuple


class Agent:
    n_actions: int = 2

    def select_action(self, state: Tuple, greedy: bool = False) -> int:
        raise NotImplementedError

    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        next_action: int | None,
        done: bool,
    ) -> None:
        """Optional learning hook. Baselines can leave this empty."""
        return None

    def end_episode(self) -> None:
        """Called after each episode (for epsilon decay etc.)."""
        return None

    def save(self, path: str) -> None:
        return None

    def load(self, path: str) -> None:
        return None
