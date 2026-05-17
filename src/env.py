"""Thin wrapper around Gymnasium's Taxi-v4.

State space (500 discrete cells):
    state = ((taxi_row * 5) + taxi_col) * 5 * 4
            + passenger_location * 4
            + destination
    decoded as (taxi_row, taxi_col, passenger_loc, destination).

    passenger_loc: 0=R, 1=G, 2=Y, 3=B, 4=in_taxi
    destination  : 0=R, 1=G, 2=Y, 3=B

Action space (6):
    0=south, 1=north, 2=east, 3=west, 4=pickup, 5=dropoff

Rewards:
    -1   per step
    -10  for an illegal pickup or dropoff
    +20  for successful delivery (terminal)
"""
from __future__ import annotations

import gymnasium as gym

N_STATES = 500
N_ACTIONS = 6
GRID_SHAPE = (5, 5)
ACTION_NAMES = ("south", "north", "east", "west", "pickup", "dropoff")
LOC_NAMES = ("R", "G", "Y", "B", "in_taxi")
LOCATIONS = ((0, 0), (0, 4), (4, 0), (4, 3))   # R, G, Y, B


class TaxiEnv:
    def __init__(self, render_mode: str | None = None) -> None:
        self.env = gym.make("Taxi-v4", render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed: int | None = None) -> int:
        obs, _ = self.env.reset(seed=seed)
        return int(obs)

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["illegal_action"] = reward <= -10 and not terminated
        info["delivered"] = reward >= 20
        done = bool(terminated or truncated)
        return int(obs), float(reward), done, info

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def decode(self, state: int) -> tuple[int, int, int, int]:
        """Return (taxi_row, taxi_col, passenger_loc, destination)."""
        return tuple(self.env.unwrapped.decode(state))  # type: ignore[return-value]


def decode_state(state: int) -> tuple[int, int, int, int]:
    """Static decode without needing an env instance."""
    dest = state % 4
    state //= 4
    pass_loc = state % 5
    state //= 5
    col = state % 5
    row = state // 5
    return row, col, pass_loc, dest
