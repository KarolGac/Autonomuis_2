"""Plot smoothed reward curves from one or more training runs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_returns(run_dir: Path) -> np.ndarray:
    log = run_dir / "log.csv"
    returns: list[float] = []
    with open(log) as f:
        reader = csv.DictReader(f)
        for row in reader:
            returns.append(float(row["return"]))
    return np.array(returns)


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--output", type=str, default="results/reward_curves.png")
    args = parser.parse_args()

    plt.figure(figsize=(9, 5))
    for run in args.runs:
        run_dir = Path(run)
        returns = load_returns(run_dir)
        smoothed = smooth(returns, args.window)
        x = np.arange(len(smoothed)) + args.window
        plt.plot(x, smoothed, label=run_dir.name)

    plt.xlabel("Episode")
    plt.ylabel(f"Return (rolling mean, w={args.window})")
    plt.title("Training reward curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=120)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
