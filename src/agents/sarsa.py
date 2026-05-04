"""Tabular SARSA, implemented from scratch (on-policy TD control).

Update rule (Rummery & Niranjan, 1994):
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]

The key difference vs Q-learning: the bootstrap uses the action a' that the
behaviour (epsilon-greedy) policy actually selects, not the greedy action.
"""
from __future__ import annotations

from typing import Tuple

from .q_learning import QLearningAgent


class SarsaAgent(QLearningAgent):
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        next_action: int | None = None,
        done: bool = False,
    ) -> None:
        if next_action is None and not done:
            raise ValueError("SARSA requires next_action when not terminal.")
        q_sa = self.Q[state][action]
        target = reward
        if not done:
            target += self.gamma * self.Q[next_state][next_action]
        self.Q[state][action] = q_sa + self.alpha * (target - q_sa)
