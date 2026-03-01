"""Boundary topology figure (Phase 5 — peer review revision).

Side-by-side boxplots: toroidal vs bounded for each ablation condition.
"""

from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
from _project_root import PROJECT_ROOT
from figures._shared import FIG_DIR

# Topology-specific colors (not condition-based)
TOPO_COLORS = {"toroidal": "#0072B2", "bounded": "#D55E00"}  # Okabe-Ito blue/vermillion

ANALYSIS_PATH = PROJECT_ROOT / "experiments" / "boundary_topology_analysis.json"


def generate_boundary_topology() -> None:
    """Generate the boundary topology comparison figure."""
    if not ANALYSIS_PATH.exists():
        print(f"  SKIP: {ANALYSIS_PATH} not found")
        return

    with open(ANALYSIS_PATH) as f:
        data = json.load(f)

    summary = data.get("alive_summary", {})
    if not summary:
        print("  SKIP: no alive_summary data")
        return

    # Get conditions (excluding 'normal' for the comparison)
    all_conditions = set()
    for topo_data in summary.values():
        all_conditions.update(topo_data.keys())
    conditions = sorted(c for c in all_conditions if c != "normal")

    if not conditions:
        print("  SKIP: no ablation conditions")
        return

    deltas = data.get("deltas_by_topology", {})
    rank_info = data.get("rank_comparison", {})

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Panel A: Raw alive counts by topology
    ax = axes[0]
    x = np.arange(len(conditions))
    width = 0.35

    for i, topo in enumerate(["toroidal", "bounded"]):
        topo_data = summary.get(topo, {})
        means = [topo_data.get(c, {}).get("mean", 0) for c in conditions]
        ax.bar(
            x + i * width - width / 2,
            means,
            width,
            label=topo.title(),
            color=TOPO_COLORS[topo],
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("no_", "No ").replace("_", " ").title() for c in conditions],
        fontsize=7,
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Mean Final Alive Count")
    ax.set_title("Alive Counts by Topology", fontsize=9)
    ax.legend(fontsize=7)

    # Panel B: Delta% comparison
    ax = axes[1]
    toro_deltas = deltas.get("toroidal", {})
    bounded_deltas = deltas.get("bounded", {})

    toro_vals = [toro_deltas.get(c, 0) for c in conditions]
    bounded_vals = [bounded_deltas.get(c, 0) for c in conditions]

    ax.bar(
        x - width / 2, toro_vals, width, label="Toroidal", color=TOPO_COLORS["toroidal"], alpha=0.8
    )
    ax.bar(
        x + width / 2, bounded_vals, width, label="Bounded", color=TOPO_COLORS["bounded"], alpha=0.8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("no_", "No ").replace("_", " ").title() for c in conditions],
        fontsize=7,
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Δ% vs Baseline")
    rho = rank_info.get("spearman_rho", float("nan"))
    match_str = "match" if rank_info.get("ranks_match", False) else "differ"
    rho_str = f"{rho:.2f}" if not math.isnan(rho) else "N/A"
    ax.set_title(f"Effect Hierarchy (ρ={rho_str}, ranks {match_str})", fontsize=8)
    ax.legend(fontsize=7)

    fig.tight_layout()
    out_path = FIG_DIR / "fig_boundary_topology.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")
