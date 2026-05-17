# Taxi-v4 RL (Q-learning en SARSA)

Klein RL-project op Gymnasium Taxi-v4. We vergelijken:

- Q-learning (tabulair)
- SARSA (tabulair)
- baselines: random en heuristiek

## Installatie

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell gebruiken
pip install -r requirements.txt
```

## Snel starten

Train:

```bash
python -m src.train --agent qlearning --config experiments/qlearning_default.yaml --episodes 5000
python -m src.train --agent sarsa     --config experiments/sarsa_default.yaml     --episodes 5000
```

Evalueren (greedy, epsilon=0):

```bash
python -m src.evaluate --agent qlearning --weights results/qlearning_default/qtable.npy --episodes 100
python -m src.evaluate --agent sarsa     --weights results/sarsa_default/qtable.npy     --episodes 100
python -m src.evaluate --agent random   --episodes 100
python -m src.evaluate --agent heuristic --episodes 100
```

Plot reward curves:

```bash
python -m src.plot_results --runs results/qlearning_default results/sarsa_default
```

Policy plot:

```bash
python -m src.plot_policy --weights results/qlearning_default/qtable.npy --output results/qlearning_default/policy.png
```

Notebook demo:

```bash
jupyter notebook notebooks/01_project_demo.ipynb
```

## Structuur

```text
.
├── src/
│   ├── env.py                # Taxi-v4 wrapper en state-decoder
│   ├── agents/
│   │   ├── base.py
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   │   ├── q_learning.py     # tabulaire Q-learning, zelf gebouwd
│   │   └── sarsa.py          # tabulaire SARSA, zelf gebouwd
│   ├── train.py              # training
│   ├── evaluate.py           # evaluatie
│   ├── plot_results.py       # training curves
│   └── plot_policy.py        # policy visualisatie
├── experiments/              # YAML configs voor hyperparameters
├── results/                  # logs, Q-tables, plots
├── tests/                    # kleine tests
└── report/                   # verslag
```

## Korte samenvatting

- States: 500 discrete staten (taxi positie, passagier locatie, bestemming).
- Acties (6): 0=zuid, 1=noord, 2=oost, 3=west, 4=pickup, 5=dropoff.
- Rewards: -1 per stap, -10 bij illegale pickup/dropoff, +20 bij succesvolle aflevering.

## Reproduceerbaarheid

Alle scripts accepteren `--seed`. Training schrijft `log.csv` en slaat de Q-table op in `results/<run>/`.
