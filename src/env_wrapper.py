"""Flappy Bird environment wrapper.

Discretizes the continuous observation from `flappy-bird-gymnasium`
(FlappyBird-v0 with use_lidar=False) into a tuple of small ints so a
tabular agent can index a Q-table.

Observation layout (from flappy-bird-gymnasium docs):
    0:  last pipe horizontal position
    1:  last top pipe vertical position
    2:  last bottom pipe vertical position
    3:  next pipe horizontal position
    4:  next top pipe vertical position
    5:  next bottom pipe vertical position
    6:  next-next pipe horizontal position
    7:  next-next top pipe vertical position
    8:  next-next bottom pipe vertical position
    9:  player vertical position
    10: player vertical velocity
    11: player rotation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np

# Importing this module registers the FlappyBird-v0 env.
import flappy_bird_gymnasium  # noqa: F401

State = Tuple[int, int, int]


@dataclass
class DiscretizationConfig:
    dx_bins: int = 12        # horizontal distance to next pipe
    dy_bins: int = 15        # vertical offset (bird y - gap-center y)
    vy_bins: int = 10        # bird vertical velocity


class FlappyBirdDiscrete:
    """Wrap FlappyBird-v0 and produce discrete state tuples + shaped rewards."""

    def __init__(
        self,
        render_mode: str | None = None,
        disc: DiscretizationConfig | None = None,
        reward_alive: float = 0.0,
        reward_pipe: float = 1.0,
        reward_death: float = -1.0,
        reward_gap_shaping: float = 1.0,
    ) -> None:
        self.env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)
        self.disc = disc or DiscretizationConfig()
        self.reward_alive = reward_alive
        self.reward_pipe = reward_pipe
        self.reward_death = reward_death
        self.reward_gap_shaping = reward_gap_shaping
        self._last_dy: float = 0.0
        self.action_space = self.env.action_space  # Discrete(2)

    # -- public API ------------------------------------------------------
    def reset(self, seed: int | None = None) -> State:
        obs, _ = self.env.reset(seed=seed)
        self._last_dy = self._compute_dy(obs)
        return self._encode(obs)

    def step(self, action: int) -> tuple[State, float, bool, dict]:
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        new_dy = self._compute_dy(obs)
        reward = self._shape_reward(raw_reward, done, new_dy)
        self._last_dy = new_dy
        return self._encode(obs), reward, done, info

    def close(self) -> None:
        self.env.close()

    # -- internals -------------------------------------------------------
    def _shape_reward(self, raw_reward: float, done: bool, new_dy: float) -> float:
        # Heavy death penalty + bigger pipe reward break the local optimum
        # where "never flap" looks better than exploring upward. The shaping
        # term rewards moving closer to the next gap center per step, which
        # gives a dense gradient before the agent ever passes a pipe.
        if done:
            return self.reward_death
        if raw_reward >= 1.0:
            return self.reward_pipe
        shaping = self.reward_gap_shaping * (abs(self._last_dy) - abs(new_dy))
        return self.reward_alive + shaping

    @staticmethod
    def _compute_dy(obs: np.ndarray) -> float:
        gap_center = 0.5 * (float(obs[4]) + float(obs[5]))
        return float(obs[9]) - gap_center

    def _encode(self, obs: np.ndarray) -> State:
        next_pipe_x = float(obs[3])
        bird_vy = float(obs[10])
        dy = self._compute_dy(obs)

        # Ranges from src/inspect_obs.py output.
        dx_idx = self._bin(next_pipe_x, lo=0.0, hi=1.0, n=self.disc.dx_bins)
        dy_idx = self._bin(dy, lo=-0.5, hi=0.5, n=self.disc.dy_bins)
        vy_idx = self._bin(bird_vy, lo=-1.0, hi=1.0, n=self.disc.vy_bins)
        return (dx_idx, dy_idx, vy_idx)

    @staticmethod
    def _bin(value: float, lo: float, hi: float, n: int) -> int:
        if value <= lo:
            return 0
        if value >= hi:
            return n - 1
        return int((value - lo) / (hi - lo) * n)


class FlappyBirdContinuous:
    """Same env + reward shaping as FlappyBirdDiscrete, but yields a compact
    5-feature numpy state suitable for a neural-network Q-function.

    state = [next_pipe_x, dy, bird_vy, bird_y, gap_size]
    """

    STATE_DIM = 5

    def __init__(
        self,
        render_mode: str | None = None,
        reward_alive: float = 0.0,
        reward_pipe: float = 1.0,
        reward_death: float = -1.0,
        reward_gap_shaping: float = 1.0,
    ) -> None:
        self.env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)
        self.reward_alive = reward_alive
        self.reward_pipe = reward_pipe
        self.reward_death = reward_death
        self.reward_gap_shaping = reward_gap_shaping
        self._last_dy: float = 0.0
        self.action_space = self.env.action_space

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _ = self.env.reset(seed=seed)
        self._last_dy = self._compute_dy(obs)
        return self._encode(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        new_dy = self._compute_dy(obs)
        reward = self._shape_reward(raw_reward, done, new_dy)
        self._last_dy = new_dy
        return self._encode(obs), reward, done, info

    def close(self) -> None:
        self.env.close()

    def _shape_reward(self, raw_reward: float, done: bool, new_dy: float) -> float:
        if done:
            return self.reward_death
        if raw_reward >= 1.0:
            return self.reward_pipe
        shaping = self.reward_gap_shaping * (abs(self._last_dy) - abs(new_dy))
        return self.reward_alive + shaping

    @staticmethod
    def _compute_dy(obs: np.ndarray) -> float:
        gap_center = 0.5 * (float(obs[4]) + float(obs[5]))
        return float(obs[9]) - gap_center

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        next_pipe_x = float(obs[3])
        next_top = float(obs[4])
        next_bot = float(obs[5])
        bird_y = float(obs[9])
        bird_vy = float(obs[10])
        dy = bird_y - 0.5 * (next_top + next_bot)
        gap_size = next_bot - next_top
        return np.array(
            [next_pipe_x, dy, bird_vy, bird_y, gap_size], dtype=np.float32
        )
