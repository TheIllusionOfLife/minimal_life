"""Figure: Fitness trajectory — evolved vs no-evolution vs crossover.

Line plot showing generation vs mean fitness (energy/alive count)
with SEM bands across conditions.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from _shared import FIG_DIR, PROJECT_ROOT

CONDITION_STYLES = {
    "normal_crossover": {"color": "#0072B2", "ls": "-", "label": "Crossover (segment)"},
    "normal_no_crossover": {"color": "#000000", "ls": "-", "label": "No Crossover"},
    "neutral_drift": {"color": "#D55E00", "ls": "--", "label": "Neutral Drift"},
    "no_evolution": {"color": "#CC79A7", "ls": ":", "label": "No Evolution"},
    "uniform_crossover": {"color": "#009E73", "ls": "-.", "label": "Crossover (uniform)"},
}


def generate_fitness_trajectory() -> None:
    """Line plot of fitness trajectories across evolution conditions."""
    traj_path = PROJECT_ROOT / "experiments" / "fitness_trajectories.json"
    if not traj_path.exists():
        print(f"  SKIP: {traj_path} not found")
        return

    with open(traj_path) as f:
        all_traj = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))

    # Panel A: Alive count over time
    ax = axes[0]
    for cond, style in CONDITION_STYLES.items():
        traj = all_traj.get(cond, {})
        if not traj:
            continue
        steps = sorted(traj.keys(), key=int)
        x = [int(s) for s in steps]
        y = [traj[s]["alive_count_mean"] for s in steps]
        sem = [traj[s]["alive_count_sem"] for s in steps]
        ax.plot(x, y, color=style["color"], ls=style["ls"], lw=1.5, label=style["label"])
        ax.fill_between(x, np.array(y) - np.array(sem), np.array(y) + np.array(sem),
                        alpha=0.15, color=style["color"])
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean Alive Count")
    ax.set_title("(A) Population Dynamics", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Energy over time
    ax = axes[1]
    for cond, style in CONDITION_STYLES.items():
        traj = all_traj.get(cond, {})
        if not traj:
            continue
        steps = sorted(traj.keys(), key=int)
        x = [int(s) for s in steps]
        y = [traj[s]["energy_mean"] for s in steps]
        sem = [traj[s]["energy_sem"] for s in steps]
        ax.plot(x, y, color=style["color"], ls=style["ls"], lw=1.5, label=style["label"])
        ax.fill_between(x, np.array(y) - np.array(sem), np.array(y) + np.array(sem),
                        alpha=0.15, color=style["color"])
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean Energy")
    ax.set_title("(B) Energy Trajectory", fontsize=9)
    ax.legend(loc="best", fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = FIG_DIR / "fig_fitness_trajectory.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    generate_fitness_trajectory()
