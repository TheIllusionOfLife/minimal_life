"""Counter/Toy ablation analysis.

Same statistical analysis as the main ablation (Mann-Whitney U, Cliff's delta,
Holm-Bonferroni) applied to Counter and Toy engine results. Compares effect
size patterns across engine types.

Usage:
    uv run python scripts/analyze_counter_toy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_common import CRITERION_TO_FLAG, experiment_output_dir, log


def cliffs_delta(x: list[float], y: list[float]) -> float:
    """Compute Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    prev_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(p * (n - rank), 1.0)
        adj = max(adj, prev_adj)
        adjusted[orig_idx] = adj
        prev_adj = adj
    return adjusted


def analyze_engine(engine_name: str, prefix: str) -> dict:
    """Analyze ablation results for one engine type."""
    out_dir = experiment_output_dir()
    criteria = list(CRITERION_TO_FLAG.keys())

    # Load normal baseline
    normal_path = out_dir / f"{prefix}normal.json"
    if not normal_path.exists():
        log(f"  SKIP: {normal_path} not found")
        return {}

    with open(normal_path) as f:
        normal_results = json.load(f)
    normal_alive = [r["final_alive_count"] for r in normal_results]

    log(f"\n=== {engine_name} Engine ===")
    log(f"  Normal baseline: mean alive = {np.mean(normal_alive):.1f} (n={len(normal_alive)})")

    results = {}
    p_values = []
    for criterion in criteria:
        cond_path = out_dir / f"{prefix}no_{criterion}.json"
        if not cond_path.exists():
            log(f"  SKIP: {cond_path} not found")
            continue
        with open(cond_path) as f:
            cond_results = json.load(f)
        cond_alive = [r["final_alive_count"] for r in cond_results]

        stat, p = stats.mannwhitneyu(normal_alive, cond_alive, alternative="two-sided")
        delta = cliffs_delta(normal_alive, cond_alive)
        p_values.append(p)

        results[criterion] = {
            "mean_alive": float(np.mean(cond_alive)),
            "mann_whitney_U": float(stat),
            "p_value": float(p),
            "cliffs_delta": float(delta),
        }

    # Holm-Bonferroni correction
    if p_values:
        adjusted = holm_bonferroni(p_values)
        for i, criterion in enumerate(results):
            results[criterion]["p_adjusted"] = adjusted[i]

    # Report
    sorted_results = sorted(
        results.items(), key=lambda x: abs(x[1]["cliffs_delta"]), reverse=True
    )
    for criterion, r in sorted_results:
        sig = "*" if r.get("p_adjusted", 1) < 0.05 else " "
        log(
            f"  {criterion:15s}: alive={r['mean_alive']:6.1f}  "
            f"delta={r['cliffs_delta']:+.3f}  p_adj={r.get('p_adjusted', 1):.4f} {sig}"
        )

    return results


def main():
    log("=== Counter/Toy Ablation Analysis ===")

    all_results = {}
    for engine, prefix in [("Counter", "counter_"), ("Toy", "toy_")]:
        all_results[engine] = analyze_engine(engine, prefix)

    # Save results
    out_dir = experiment_output_dir()
    output_path = out_dir / "counter_toy_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
