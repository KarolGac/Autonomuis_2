"""Aggregate a hyperparameter sweep into a single mean +/- std reward curve plot.

Reads `results/<sweep_name>/manifest.yaml` to discover the swept parameter
and its values, then loads each `<param>=<value>_seed=<N>/log.csv`,
groups by parameter value, computes mean +/- std over seeds, and plots one
smoothed curve per value with a shaded uncertainty band.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def _load_returns(run_dir: Path) -> np.ndarray:
    returns: list[float] = []
    with open(run_dir / "log.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            returns.append(float(row["return"]))
    return np.array(returns)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.shape[-1] < window:
        return x
    kernel = np.ones(window) / window
    if x.ndim == 1:
        return np.convolve(x, kernel, mode="valid")
    # 2D: smooth along the last axis (episodes).
    return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"),
                               axis=-1, arr=x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", required=True, type=str)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    with open(sweep_dir / "manifest.yaml") as f:
        manifest = yaml.safe_load(f)
    param = manifest["param"]
    values = manifest["values"]
    seeds = manifest["seeds"]

    plt.figure(figsize=(10, 5))
    for value in values:
        runs = []
        for seed in range(seeds):
            run_dir = sweep_dir / f"{param}={value}_seed={seed}"
            runs.append(_load_returns(run_dir))
        # Stack to (seeds, episodes); some configs may end early -- truncate.
        n_episodes = min(r.shape[0] for r in runs)
        stack = np.stack([r[:n_episodes] for r in runs], axis=0)
        smoothed = _smooth(stack, args.window)
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        x = np.arange(mean.shape[0]) + args.window
        line, = plt.plot(x, mean, label=f"{param}={value}")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=line.get_color())

    plt.xlabel("Episode")
    plt.ylabel(f"Return (rolling mean over {args.window} eps, mean +/- std over {seeds} seeds)")
    plt.title(f"Sweep over {param} ({manifest['agent']}, Taxi-v4)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = args.output or str(sweep_dir / "sweep_curves.png")
    plt.savefig(output, dpi=120)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
