"""Smoke tests for the discretized Flappy Bird wrapper."""
from src.env_wrapper import DiscretizationConfig, FlappyBirdDiscrete


def test_reset_returns_state_tuple():
    env = FlappyBirdDiscrete(disc=DiscretizationConfig())
    state = env.reset(seed=0)
    assert isinstance(state, tuple) and len(state) == 4
    assert all(isinstance(x, int) for x in state)
    env.close()


def test_step_returns_shaped_reward():
    env = FlappyBirdDiscrete()
    env.reset(seed=0)
    state, reward, done, _ = env.step(0)
    assert reward in (env.reward_alive, env.reward_pipe, env.reward_death)
    assert isinstance(done, bool)
    assert isinstance(state, tuple)
    env.close()
