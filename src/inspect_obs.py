"""Print min/max of each observation dimension over random play.

Use this to set sensible bin ranges in env_wrapper.py.
"""
import flappy_bird_gymnasium  # noqa: F401  (registers the env)
import gymnasium as gym
import numpy as np


def main(steps: int = 5000) -> None:
    env = gym.make("FlappyBird-v0", use_lidar=False)
    obs, _ = env.reset(seed=0)
    mins = obs.copy()
    maxs = obs.copy()
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        mins = np.minimum(mins, obs)
        maxs = np.maximum(maxs, obs)
        if term or trunc:
            obs, _ = env.reset()
    env.close()

    labels = [
        "last_pipe_x", "last_top_y", "last_bot_y",
        "next_pipe_x", "next_top_y", "next_bot_y",
        "next2_pipe_x", "next2_top_y", "next2_bot_y",
        "bird_y", "bird_vy", "bird_rot",
    ]
    print(f"{'idx':>3}  {'name':<14}  {'min':>10}  {'max':>10}")
    for i, name in enumerate(labels):
        print(f"{i:>3}  {name:<14}  {mins[i]:>10.4f}  {maxs[i]:>10.4f}")


if __name__ == "__main__":
    main()
