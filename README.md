# Flappy Bird Reinforcement Learning

Portfolio 2 — Reinforcement Learning. Tabular Q-learning and SARSA implemented from scratch on the Flappy Bird environment, compared against a random and a heuristic baseline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train an agent:
```bash
python -m src.train --agent qlearning --episodes 20000 --config experiments/qlearning_default.yaml
python -m src.train --agent sarsa     --episodes 20000 --config experiments/sarsa_default.yaml
```

Evaluate a trained agent (greedy, epsilon=0):
```bash
python -m src.evaluate --agent qlearning --weights results/qlearning_default/qtable.pkl --episodes 100
python -m src.evaluate --agent random   --episodes 100
python -m src.evaluate --agent heuristic --episodes 100
```

Plot reward curves across runs:
```bash
python -m src.plot_results --runs results/qlearning_default results/sarsa_default
```

Render a demo of the trained policy:
```bash
python -m src.evaluate --agent qlearning --weights results/qlearning_default/qtable.pkl --episodes 3 --render
```

## Project structure

```
.
├── src/
│   ├── env_wrapper.py        # state discretization + reward shaping
│   ├── agents/
│   │   ├── base.py
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   │   ├── q_learning.py     # tabular Q-learning, implemented from scratch
│   │   └── sarsa.py          # tabular SARSA, implemented from scratch
│   ├── train.py
│   ├── evaluate.py
│   └── plot_results.py
├── experiments/              # YAML configs per hyperparameter sweep
├── results/                  # logs, Q-tables, reward curves
├── report/                   # academic report (LaTeX/markdown)
└── tests/
```

## MDP definition

- **State** (discretized): horizontal distance to next pipe, vertical gap of next pipe, bird y-velocity, bird y-position relative to gap.
- **Actions**: `{0: do nothing, 1: flap}`.
- **Reward**: +0.1 per surviving frame, +1 per pipe passed, −1 on death.

## Reproducibility

All runs accept `--seed`. Hyperparameters live in `experiments/*.yaml`. Per-episode reward, length and epsilon are logged to `results/<run>/log.csv`.
