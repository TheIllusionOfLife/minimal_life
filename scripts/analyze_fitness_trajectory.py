"""Generation-stratified fitness trajectory analysis.

Loads crossover experiment data and computes per-generation-quartile
mean energy and alive count. Compares evolved vs no-evolution vs crossover
conditions for evidence of directional selection.

Usage:
    uv run python scripts/analyze_fitness_trajectory.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_common import experiment_output_dir, log


def load_condition_data(out_dir: Path, prefix: str, condition: str) -> list[dict]:
    """Load per-condition JSON results."""
    path = out_dir / f"{prefix}{condition}.json"
    if not path.exists():
        log(f"Warning: {path} not found, skipping")
        return []
    with open(path) as f:
        return json.load(f)


def compute_trajectory(results: list[dict]) -> dict:
    """Compute time-series trajectory from experiment results."""
    if not results:
        return {}

    # Aggregate across seeds by step
    step_data: dict[int, list[dict]] = {}
    for result in results:
        for sample in result.get("samples", []):
            step = sample["step"]
            step_data.setdefault(step, []).append(sample)

    trajectory = {}
    for step in sorted(step_data.keys()):
        samples = step_data[step]
        trajectory[step] = {
            "alive_count_mean": float(np.mean([s["alive_count"] for s in samples])),
            "alive_count_sem": float(
                np.std([s["alive_count"] for s in samples]) / np.sqrt(len(samples))
            ),
            "energy_mean": float(np.mean([s["energy_mean"] for s in samples])),
            "energy_sem": float(
                np.std([s["energy_mean"] for s in samples]) / np.sqrt(len(samples))
            ),
            "mean_generation": float(np.mean([s["mean_generation"] for s in samples])),
            "genome_diversity": float(
                np.mean([s.get("genome_diversity", 0) for s in samples])
            ),
            "n_seeds": len(samples),
        }
    return trajectory


def main():
    out_dir = experiment_output_dir()

    conditions = [
        "normal_crossover",
        "normal_no_crossover",
        "neutral_drift",
        "no_evolution",
        "uniform_crossover",
    ]

    log("=== Fitness Trajectory Analysis ===\n")

    all_trajectories = {}
    for cond in conditions:
        data = load_condition_data(out_dir, "crossover_", cond)
        trajectory = compute_trajectory(data)
        all_trajectories[cond] = trajectory

        if trajectory:
            steps = sorted(trajectory.keys())
            first = trajectory[steps[0]]
            last = trajectory[steps[-1]]
            log(f"{cond}:")
            log(f"  Steps: {steps[0]}-{steps[-1]} ({len(steps)} points)")
            log(f"  Alive: {first['alive_count_mean']:.0f} → {last['alive_count_mean']:.0f}")
            log(f"  Energy: {first['energy_mean']:.4f} → {last['energy_mean']:.4f}")
            log(f"  Generation: {first['mean_generation']:.1f} → {last['mean_generation']:.1f}")
            log("")
        else:
            log(f"{cond}: NO DATA\n")

    # Save trajectories for figure generation
    output_path = out_dir / "fitness_trajectories.json"
    with open(output_path, "w") as f:
        json.dump(
            {cond: {str(k): v for k, v in traj.items()} for cond, traj in all_trajectories.items()},
            f,
            indent=2,
        )
    log(f"Saved trajectories to {output_path}")


if __name__ == "__main__":
    main()
