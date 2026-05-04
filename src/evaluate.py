"""Evaluate an agent (greedy) over N episodes and print summary stats."""
from __future__ import annotations

import argparse
import statistics

from .agents.dqn import DQNAgent
from .agents.heuristic_agent import HeuristicAgent
from .agents.q_learning import QLearningAgent
from .agents.random_agent import RandomAgent
from .agents.sarsa import SarsaAgent
from .env_wrapper import (
    DiscretizationConfig,
    FlappyBirdContinuous,
    FlappyBirdDiscrete,
)


def build_env(agent_name: str, render: bool):
    render_mode = "human" if render else None
    if agent_name == "dqn":
        return FlappyBirdContinuous(render_mode=render_mode)
    return FlappyBirdDiscrete(disc=DiscretizationConfig(), render_mode=render_mode)


def build_agent(name: str, weights: str | None, env):
    if name == "random":
        return RandomAgent()
    if name == "heuristic":
        return HeuristicAgent(dy_bins=env.disc.dy_bins, vy_bins=env.disc.vy_bins)
    if name in ("qlearning", "sarsa"):
        agent = QLearningAgent() if name == "qlearning" else SarsaAgent()
        if weights:
            agent.load(weights)
        agent.epsilon = 0.0
        return agent
    if name == "dqn":
        agent = DQNAgent(state_dim=env.STATE_DIM)
        if weights:
            agent.load(weights)
        agent.epsilon = 0.0
        return agent
    raise ValueError(f"Unknown agent: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True,
                        choices=["random", "heuristic", "qlearning", "sarsa", "dqn"])
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = build_env(args.agent, args.render)
    agent = build_agent(args.agent, args.weights, env)

    returns: list[float] = []
    pipes: list[int] = []
    lengths: list[int] = []

    for ep in range(args.episodes):
        state = env.reset(seed=args.seed + ep)
        ep_return = 0.0
        n_pipes = 0
        steps = 0
        for _ in range(args.max_steps):
            action = agent.select_action(state, greedy=True)
            state, reward, done, _ = env.step(action)
            ep_return += reward
            if reward >= env.reward_pipe:
                n_pipes += 1
            steps += 1
            if done:
                break
        returns.append(ep_return)
        pipes.append(n_pipes)
        lengths.append(steps)

    env.close()

    def stats(xs: list[float]) -> str:
        return (
            f"mean={statistics.mean(xs):.2f} "
            f"std={statistics.pstdev(xs):.2f} "
            f"min={min(xs):.2f} max={max(xs):.2f}"
        )

    print(f"Agent: {args.agent} ({args.episodes} episodes, greedy)")
    print(f"  Return : {stats(returns)}")
    print(f"  Pipes  : {stats([float(p) for p in pipes])}")
    print(f"  Length : {stats([float(l) for l in lengths])}")


if __name__ == "__main__":
    main()
