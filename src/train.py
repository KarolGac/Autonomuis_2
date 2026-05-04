"""Train an agent on FlappyBird and log per-episode metrics."""
from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from .agents.dqn import DQNAgent
from .agents.q_learning import QLearningAgent
from .agents.sarsa import SarsaAgent
from .env_wrapper import (
    DiscretizationConfig,
    FlappyBirdContinuous,
    FlappyBirdDiscrete,
)


def build_env(agent_name: str, cfg: dict):
    env_kwargs = cfg.get("env", {})
    if agent_name == "dqn":
        return FlappyBirdContinuous(**env_kwargs)
    disc = DiscretizationConfig(**cfg.get("discretization", {}))
    return FlappyBirdDiscrete(disc=disc, **env_kwargs)


def build_agent(name: str, cfg: dict, env, seed: int):
    agent_cfg = cfg.get("agent", {})
    if name == "qlearning":
        return QLearningAgent(seed=seed, **agent_cfg)
    if name == "sarsa":
        return SarsaAgent(seed=seed, **agent_cfg)
    if name == "dqn":
        return DQNAgent(state_dim=env.STATE_DIM, seed=seed, **agent_cfg)
    raise ValueError(f"Unknown trainable agent: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["qlearning", "sarsa", "dqn"])
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = build_env(args.agent, cfg)
    agent = build_agent(args.agent, cfg, env, seed=args.seed)

    run_name = args.run_name or Path(args.config).stem
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"
    suffix = "model.pt" if args.agent == "dqn" else "qtable.pkl"
    weights_path = out_dir / suffix
    best_weights_path = out_dir / f"best_{suffix}"
    recent_returns: deque = deque(maxlen=200)
    best_avg = float("-inf")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "length", "epsilon"])

        for ep in range(args.episodes):
            state = env.reset(seed=args.seed + ep)
            action = agent.select_action(state)
            ep_return = 0.0
            steps = 0

            for _ in range(args.max_steps):
                next_state, reward, done, _ = env.step(action)
                ep_return += reward
                steps += 1

                if args.agent == "sarsa":
                    next_action = agent.select_action(next_state) if not done else None
                    agent.update(state, action, reward, next_state, next_action, done)
                    action = next_action if not done else 0
                else:
                    agent.update(state, action, reward, next_state, None, done)
                    action = agent.select_action(next_state) if not done else 0

                state = next_state
                if done:
                    break

            agent.end_episode()
            writer.writerow([ep, f"{ep_return:.3f}", steps, f"{agent.epsilon:.4f}"])
            recent_returns.append(ep_return)

            # Save best-so-far whenever the rolling mean over last 200 eps
            # exceeds the previous best (after a short warm-up).
            if len(recent_returns) == recent_returns.maxlen:
                avg = sum(recent_returns) / len(recent_returns)
                if avg > best_avg:
                    best_avg = avg
                    agent.save(str(best_weights_path))

            if (ep + 1) % 500 == 0:
                f.flush()
                avg = sum(recent_returns) / max(1, len(recent_returns))
                print(f"[{run_name}] ep={ep + 1} return={ep_return:.2f} "
                      f"avg200={avg:.2f} best200={best_avg:.2f} eps={agent.epsilon:.3f}")

    agent.save(str(weights_path))
    env.close()
    print(f"Saved weights and log to {out_dir}")


if __name__ == "__main__":
    main()
