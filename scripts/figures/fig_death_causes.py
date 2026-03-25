"""Figure: Death cause distribution across ablation conditions.

Stacked bar chart showing proportion of deaths by cause
(BoundaryCollapse / EnergyDepletion / AgeLimit) per condition.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from _shared import COLORS, FIG_DIR, LABELS, PROJECT_ROOT


def generate_death_causes() -> None:
    """Stacked bar chart of death causes per ablation condition."""
    exp_dir = PROJECT_ROOT / "experiments"

    conditions = list(COLORS.keys())
    cause_names = ["boundary_collapse", "energy_depletion", "age_limit"]
    cause_colors = ["#56B4E9", "#E69F00", "#009E73"]  # Okabe-Ito subset
    cause_labels = ["Boundary Collapse", "Energy Depletion", "Age Limit"]

    # Aggregate death counts per condition
    totals = {cond: {c: 0 for c in cause_names} for cond in conditions}
    found_any = False

    for cond in conditions:
        path = exp_dir / f"death_causes_{cond}.json"
        if not path.exists():
            continue
        found_any = True
        with open(path) as f:
            results = json.load(f)
        for result in results:
            summary = result.get("summary", result)
            totals[cond]["boundary_collapse"] += summary.get("deaths_boundary", 0)
            totals[cond]["energy_depletion"] += summary.get("deaths_energy", 0)
            totals[cond]["age_limit"] += summary.get("deaths_age", 0)

    if not found_any:
        print("  SKIP: no death_causes_*.json files found")
        return

    # Normalize to proportions
    props = {}
    for cond in conditions:
        total = sum(totals[cond].values())
        if total > 0:
            props[cond] = {c: totals[cond][c] / total for c in cause_names}
        else:
            props[cond] = {c: 0.0 for c in cause_names}

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(conditions))
    width = 0.6

    bottoms = np.zeros(len(conditions))
    for cause, color, label in zip(cause_names, cause_colors, cause_labels, strict=True):
        values = [props[cond][cause] for cond in conditions]
        ax.bar(
            x, values, width, bottom=bottoms,
            color=color, label=label, edgecolor="white", linewidth=0.5,
        )
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(c, c) for c in conditions], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Proportion of Deaths")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = FIG_DIR / "fig_death_causes.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    generate_death_causes()
