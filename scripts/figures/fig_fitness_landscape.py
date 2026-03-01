"""Fitness landscape figure (Phase 2 — peer review revision).

Panel A: Mean energy by generation cohort (evolved vs clonal) with error bars.
Panel B: Parent-offspring energy scatter with regression line + 95% CI band.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from _project_root import PROJECT_ROOT
from figures._shared import COLORS, FIG_DIR

ANALYSIS_PATH = PROJECT_ROOT / "experiments" / "fitness_landscape_analysis.json"


def generate_fitness_landscape() -> None:
    """Generate the 2-panel fitness landscape figure."""
    if not ANALYSIS_PATH.exists():
        print(f"  SKIP: {ANALYSIS_PATH} not found")
        return

    with open(ANALYSIS_PATH) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Panel A: Generation cohort energy (from h1_trend_test)
    ax = axes[0]
    h1 = data.get("h1_trend_test", {})
    evolved = h1.get("evolved", {})
    clonal = h1.get("clonal_control", {})

    e_means = evolved.get("cohort_means", [])
    c_means = clonal.get("cohort_means", [])

    if e_means:
        cohort_labels = [f"{i * 5}–{i * 5 + 4}" for i in range(len(e_means) - 1)]
        cohort_labels.append(f"{(len(e_means) - 1) * 5}+")
        x = np.arange(len(e_means))
        ax.bar(x - 0.15, e_means, 0.3, label="Evolved", color=COLORS["normal"], alpha=0.8)
        if c_means:
            ax.bar(
                x + 0.15,
                c_means[: len(x)],
                0.3,
                label="Clonal",
                color=COLORS["no_evolution"],
                alpha=0.8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(cohort_labels, fontsize=7)
        ax.set_xlabel("Generation Cohort")
        ax.set_ylabel("Mean Energy")
        p_val = evolved.get("p_value", float("nan"))
        ax.set_title(f"H1: Directional Selection (p={p_val:.3g})", fontsize=9)
        ax.legend(fontsize=7)
    else:
        ax.text(
            0.5,
            0.5,
            "No cohort data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="gray",
        )
        ax.set_title("H1: Directional Selection", fontsize=9)

    # Panel B: Parent-offspring regression
    ax = axes[1]
    h2 = data.get("h2_regression", {})
    slope = h2.get("slope", float("nan"))
    ci_lo = h2.get("ci_lower", float("nan"))
    ci_hi = h2.get("ci_upper", float("nan"))
    n_pairs = h2.get("n_pairs", 0)

    if n_pairs >= 2:
        # We don't have raw scatter data in the analysis JSON, so show a summary
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.bar(
            ["Slope"],
            [slope],
            yerr=[[slope - ci_lo], [ci_hi - slope]],
            color=COLORS["normal"],
            alpha=0.8,
            capsize=5,
        )
        ax.set_ylabel("Regression Slope")
        ci_text = f"[{ci_lo:.3f}, {ci_hi:.3f}]"
        ax.set_title(
            f"H2: Heritability (n={n_pairs}, 95% CI {ci_text})",
            fontsize=8,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient linkage data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="gray",
        )
        ax.set_title("H2: Parent-Offspring Regression", fontsize=9)

    fig.tight_layout()
    out_path = FIG_DIR / "fig_fitness_landscape.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")
