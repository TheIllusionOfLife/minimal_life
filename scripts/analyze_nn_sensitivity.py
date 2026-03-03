"""NN hidden size sensitivity analysis.

Compares population dynamics across {8, 16, 32} hidden sizes to verify
ablation results are not NN-capacity artifacts.

Usage:
    uv run python scripts/analyze_nn_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_common import experiment_output_dir, log


def main():
    out_dir = experiment_output_dir()
    log("=== NN Hidden Size Sensitivity Analysis ===\n")

    hidden_sizes = [8, 16, 32]
    results = {}

    for h in hidden_sizes:
        path = out_dir / f"nn_sweep_hidden_{h}.json"
        if not path.exists():
            log(f"  SKIP: {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)
        alive_counts = [r["final_alive_count"] for r in data]
        results[h] = {
            "mean_alive": float(np.mean(alive_counts)),
            "std_alive": float(np.std(alive_counts)),
            "median_alive": float(np.median(alive_counts)),
            "n": len(alive_counts),
            "alive_counts": alive_counts,
        }
        log(
            f"  hidden={h:3d}: mean_alive={results[h]['mean_alive']:.1f} "
            f"(std={results[h]['std_alive']:.1f}, n={results[h]['n']})"
        )

    if len(results) < 2:
        log("\nInsufficient data for comparison")
        return

    # Kruskal-Wallis test across all hidden sizes
    groups = [results[h]["alive_counts"] for h in sorted(results)]
    if len(groups) >= 2:
        stat, p = stats.kruskal(*groups)
        log(f"\n  Kruskal-Wallis H={stat:.3f}, p={p:.4f}")
        if p < 0.05:
            log("  WARNING: Significant difference across hidden sizes — NN capacity matters!")
        else:
            log("  OK: No significant effect of hidden size on population dynamics")

    # Pairwise comparison: 8 vs 16, 16 vs 32
    for h1, h2 in [(8, 16), (16, 32)]:
        if h1 in results and h2 in results:
            stat, p = stats.mannwhitneyu(
                results[h1]["alive_counts"], results[h2]["alive_counts"],
                alternative="two-sided",
            )
            log(f"  {h1} vs {h2}: U={stat:.0f}, p={p:.4f}")

    # Save
    output = {str(k): {kk: vv for kk, vv in v.items() if kk != "alive_counts"} for k, v in results.items()}
    output_path = out_dir / "nn_sensitivity_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
