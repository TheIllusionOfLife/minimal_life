"""Threshold sensitivity figure (Phase 4 — peer review revision).

Heatmap: threshold combo × ablation condition → Δ% showing stability
of effect hierarchy across death threshold variations.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from _project_root import PROJECT_ROOT
from figures._shared import FIG_DIR

ANALYSIS_PATH = (
    PROJECT_ROOT / "experiments" / "threshold_sensitivity_analysis.json"
)


def generate_threshold_sensitivity() -> None:
    """Generate the threshold sensitivity heatmap figure."""
    if not ANALYSIS_PATH.exists():
        print(f"  SKIP: {ANALYSIS_PATH} not found")
        return

    with open(ANALYSIS_PATH) as f:
        data = json.load(f)

    deltas = data.get("deltas_by_combo", {})
    if not deltas:
        print("  SKIP: no delta data")
        return

    combo_keys = sorted(deltas.keys())
    # Collect all conditions across combos
    all_conditions: set[str] = set()
    for d in deltas.values():
        all_conditions.update(d.keys())
    conditions = sorted(all_conditions)

    if not conditions:
        print("  SKIP: no conditions")
        return

    # Build matrix
    matrix = np.full((len(combo_keys), len(conditions)), np.nan)
    for i, combo in enumerate(combo_keys):
        for j, cond in enumerate(conditions):
            val = deltas[combo].get(cond)
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(
        matrix, aspect="auto", cmap="RdYlBu",
        vmin=-100, vmax=0,
    )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(
        [c.replace("no_", "No ").replace("_", " ").title()
         for c in conditions],
        fontsize=7, rotation=30, ha="right",
    )
    ax.set_yticks(range(len(combo_keys)))
    ax.set_yticklabels(combo_keys, fontsize=7)
    ax.set_xlabel("Ablation Condition")
    ax.set_ylabel("Threshold Combo")

    # Add text annotations
    for i in range(len(combo_keys)):
        for j in range(len(conditions)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=6, color="black" if val > -50 else "white")

    # Add rank stability info
    stability = data.get("rank_stability", {})
    stable = stability.get("all_stable", False)
    title = "Threshold Sensitivity: Δ% Ablation Effect"
    if stable:
        title += " (ranks stable)"
    ax.set_title(title, fontsize=9)

    fig.colorbar(im, ax=ax, label="Δ% vs Baseline", shrink=0.8)
    fig.tight_layout()

    out_path = FIG_DIR / "fig_threshold_sensitivity.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")
