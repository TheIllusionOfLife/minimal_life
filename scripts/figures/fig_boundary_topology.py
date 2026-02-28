"""Boundary topology figure (Phase 5 — peer review revision).

Side-by-side boxplots: toroidal vs bounded for each ablation condition.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from _project_root import PROJECT_ROOT
from figures._shared import COLORS, FIG_DIR

ANALYSIS_PATH = (
    PROJECT_ROOT / "experiments" / "boundary_topology_analysis.json"
)


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

    for i, (topo, color_key) in enumerate([
        ("toroidal", "normal"),
        ("bounded", "no_evolution"),
    ]):
        topo_data = summary.get(topo, {})
        means = [topo_data.get(c, {}).get("mean", 0) for c in conditions]
        ax.bar(x + i * width - width / 2, means, width,
               label=topo.title(), color=COLORS.get(color_key, "#888888"),
               alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("no_", "No ").replace("_", " ").title() for c in conditions],
        fontsize=7, rotation=20, ha="right",
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

    ax.bar(x - width / 2, toro_vals, width, label="Toroidal",
           color=COLORS.get("normal", "#888888"), alpha=0.8)
    ax.bar(x + width / 2, bounded_vals, width, label="Bounded",
           color=COLORS.get("no_evolution", "#888888"), alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("no_", "No ").replace("_", " ").title() for c in conditions],
        fontsize=7, rotation=20, ha="right",
    )
    ax.set_ylabel("Δ% vs Baseline")
    rho = rank_info.get("spearman_rho", float("nan"))
    match_str = "match" if rank_info.get("ranks_match", False) else "differ"
    ax.set_title(f"Effect Hierarchy (ρ={rho:.2f}, ranks {match_str})", fontsize=8)
    ax.legend(fontsize=7)

    fig.tight_layout()
    out_path = FIG_DIR / "fig_boundary_topology.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")
