"""Growth/reproduction separability analysis.

Quantifies growth's independent effect vs reproduction gating effect
by comparing 4 conditions: normal, no_growth, bypass_maturity, no_growth_bypass.

Usage:
    uv run python scripts/analyze_growth_repro.py
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
    log("=== Growth/Reproduction Separability Analysis ===\n")

    conditions = ["normal", "no_growth", "bypass_maturity", "no_growth_bypass"]
    results = {}

    for cond in conditions:
        path = out_dir / f"growth_repro_{cond}.json"
        if not path.exists():
            log(f"  SKIP: {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)
        alive_counts = [r["final_alive_count"] for r in data]
        results[cond] = {
            "mean_alive": float(np.mean(alive_counts)),
            "std_alive": float(np.std(alive_counts)),
            "n": len(alive_counts),
            "alive_counts": alive_counts,
        }
        log(
            f"  {cond:25s}: mean_alive={results[cond]['mean_alive']:.1f} "
            f"(std={results[cond]['std_alive']:.1f}, n={results[cond]['n']})"
        )

    if len(results) < 2:
        log("\nInsufficient data for comparison")
        return

    log("\n--- Comparisons ---")

    # Key comparisons:
    # 1. normal vs no_growth: full growth effect (gating + independent)
    # 2. normal vs bypass_maturity: maturity gating effect only
    # 3. bypass_maturity vs no_growth_bypass: growth's non-gating effect
    comparisons = [
        ("normal", "no_growth", "Growth effect (total)"),
        ("normal", "bypass_maturity", "Maturity gate effect"),
        ("bypass_maturity", "no_growth_bypass", "Growth independent of gating"),
        ("no_growth", "no_growth_bypass", "Maturity gate without growth"),
    ]

    for c1, c2, desc in comparisons:
        if c1 in results and c2 in results:
            stat, p = stats.mannwhitneyu(
                results[c1]["alive_counts"], results[c2]["alive_counts"],
                alternative="two-sided",
            )
            diff = results[c1]["mean_alive"] - results[c2]["mean_alive"]
            sig = "*" if p < 0.05 else " "
            log(f"  {desc:35s}: diff={diff:+.1f}, U={stat:.0f}, p={p:.4f} {sig}")

    # Save
    output = {
        k: {kk: vv for kk, vv in v.items() if kk != "alive_counts"}
        for k, v in results.items()
    }
    output_path = out_dir / "growth_repro_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
