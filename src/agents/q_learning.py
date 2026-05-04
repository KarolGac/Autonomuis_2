"""Tabular Q-learning, implemented from scratch.

Update rule (Watkins, 1989):
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
"""
from __future__ import annotations

import pickle
import random
from collections import defaultdict
from typing import Tuple

from .base import Agent


def _init_action_values(n_actions: int, init_value: float):
    # Defined at module scope so pickle can find it.
    def _factory():
        return [init_value] * n_actions
    return _factory


class QLearningAgent(Agent):
    def __init__(
        self,
        n_actions: int = 2,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 5000,
        optimistic_init: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.optimistic_init = optimistic_init
        self._episode = 0
        self.rng = random.Random(seed)
        self.Q: dict = defaultdict(_init_action_values(n_actions, optimistic_init))

    # -- policy ----------------------------------------------------------
    def select_action(self, state: Tuple, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        return self._argmax(self.Q[state])

    # -- learning --------------------------------------------------------
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        next_action: int | None = None,
        done: bool = False,
    ) -> None:
        q_sa = self.Q[state][action]
        target = reward
        if not done:
            target += self.gamma * max(self.Q[next_state])
        self.Q[state][action] = q_sa + self.alpha * (target - q_sa)

    def end_episode(self) -> None:
        self._episode += 1
        frac = min(1.0, self._episode / max(1, self.epsilon_decay_episodes))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # -- helpers ---------------------------------------------------------
    def _argmax(self, values) -> int:
        best_val = values[0]
        best_idx = 0
        for i in range(1, len(values)):
            if values[i] > best_val:
                best_val = values[i]
                best_idx = i
        return best_idx

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "Q": dict(self.Q),
                    "epsilon": self.epsilon,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "n_actions": self.n_actions,
                },
                f,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q = defaultdict(_init_action_values(self.n_actions, self.optimistic_init))
        for k, v in data["Q"].items():
            self.Q[k] = list(v)
        self.epsilon = data.get("epsilon", self.epsilon)
