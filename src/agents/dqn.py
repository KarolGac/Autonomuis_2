"""Deep Q-Network (Mnih et al., 2015), implemented from scratch in PyTorch.

Components:
    * QNetwork: 2-layer MLP that maps state -> Q-values for each action.
    * ReplayBuffer: ring-buffer of transitions sampled uniformly per update.
    * Target network: a frozen copy of the online net, refreshed every
      `target_update_steps` gradient steps. Stabilises bootstrapped targets.

Update (per minibatch):
    y_j = r_j                                if terminal
        = r_j + gamma * max_a' Q_target(s'_j, a')   otherwise
    L = mean( (Q_online(s_j, a_j) - y_j)^2 )
"""
from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Agent


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        self.buf: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, s, a, r, s2, d) -> None:
        self.buf.append((s, a, r, s2, d))

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.asarray(s2, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


class DQNAgent(Agent):
    def __init__(
        self,
        state_dim: int,
        n_actions: int = 2,
        hidden: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        warmup_steps: int = 1_000,
        target_update_steps: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 50_000,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_steps = target_update_steps
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self._step = 0
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.online = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, seed=seed)
        self.rng = random.Random(seed)

    # -- policy ----------------------------------------------------------
    def select_action(self, state, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return int(self.online(s.unsqueeze(0)).argmax(dim=1).item())

    # -- learning --------------------------------------------------------
    def update(self, state, action, reward, next_state, next_action=None,
               done: bool = False) -> None:
        self.buffer.push(state, action, reward, next_state, float(done))
        self._step += 1

        # Linear epsilon decay over the first epsilon_decay_steps env steps.
        frac = min(1.0, self._step / max(1, self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return

        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s_t = torch.from_numpy(s).to(self.device)
        a_t = torch.from_numpy(a).to(self.device).unsqueeze(1)
        r_t = torch.from_numpy(r).to(self.device)
        s2_t = torch.from_numpy(s2).to(self.device)
        d_t = torch.from_numpy(d).to(self.device)

        q_sa = self.online(s_t).gather(1, a_t).squeeze(1)
        with torch.no_grad():
            max_q_next = self.target(s2_t).max(dim=1).values
            target = r_t + self.gamma * max_q_next * (1.0 - d_t)
        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        if self._step % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

    def end_episode(self) -> None:
        return None

    # -- I/O -------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "step": self._step,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.online.load_state_dict(data["online"])
        self.target.load_state_dict(data["target"])
        self._step = data.get("step", 0)
        self.epsilon = data.get("epsilon", self.epsilon)
