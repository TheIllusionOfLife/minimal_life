"""Figure 2: Population dynamics time-series."""

from collections import defaultdict

import numpy as np
from figures._shared import *

# Markers and line styles for color accessibility (n8)
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def generate_timeseries(data: list[dict]) -> None:
    """Figure 2: Population dynamics time-series with confidence bands."""
    # Group by (condition, step) → list of alive_count values
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in data:
        key = (row["condition"], int(row["step"]))
        groups[key].append(row["alive_count"])

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    for idx, condition in enumerate(CONDITION_ORDER):
        steps = sorted({s for (c, s) in groups if c == condition})
        means = []
        sems = []
        for step in steps:
            vals = groups[(condition, step)]
            arr = np.array(vals)
            means.append(arr.mean())
            sems.append(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) >= 2 else 0.0)

        means = np.array(means)
        sems = np.array(sems)
        color = COLORS[condition]
        lw = 1.5 if condition == "normal" else 0.8
        ls = "-" if condition == "normal" else _LINESTYLES[idx % len(_LINESTYLES)]
        marker = None if condition == "normal" else _MARKERS[idx % len(_MARKERS)]
        # Place markers every 10th data point for readability
        markevery = max(1, len(steps) // 10) if marker else None
        ax.plot(
            steps,
            means,
            color=color,
            linewidth=lw,
            linestyle=ls,
            marker=marker,
            markevery=markevery,
            markersize=3,
            label=LABELS[condition],
            zorder=10 if condition == "normal" else 5,
        )
        ax.fill_between(steps, means - sems, means + sems, color=color, alpha=0.15, zorder=2)

    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean Alive Count ($n$=30)")
    ax.set_xlim(0, 2000)
    ax.set_ylim(bottom=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=4,
        framealpha=0.9,
        edgecolor="0.8",
        fontsize=5,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_timeseries.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_timeseries.pdf'}")
