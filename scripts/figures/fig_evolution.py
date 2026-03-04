"""Figure 4: Evolution strengthening — long run + env shift."""

from collections import defaultdict

import numpy as np
from figures._shared import *


def generate_evolution() -> None:
    """Figure 4: Evolution strengthening — long run + env shift."""
    exp_dir = PROJECT_ROOT / "experiments"

    conditions = {
        "long_normal": ("Normal", "#000000"),
        "long_no_evolution": ("No Evolution", "#CC79A7"),
        "shift_normal": ("Normal", "#000000"),
        "shift_no_evolution": ("No Evolution", "#CC79A7"),
    }

    # Load primary time-series from dedicated evolution experiment outputs.
    cond_data: dict[str, dict[int, list[float]]] = {}
    missing_primary = False
    for cond in conditions:
        path = exp_dir / f"evolution_{cond}.json"
        if not path.exists():
            missing_primary = True
            break
        results = load_json(path)
        step_vals: dict[int, list[float]] = defaultdict(list)
        for r in results:
            for s in r["samples"]:
                step_vals[s["step"]].append(s["alive_count"])
        cond_data[cond] = step_vals

    panel_titles = ("Long run (10,000 steps)", "Environmental shift at step 2,500")
    shift_line = 2500
    # Fallback for environments where long-run files are not present.
    if missing_primary:
        traj_path = exp_dir / "fitness_trajectories.json"
        stress_on_path = exp_dir / "ecology_stress_cyclic_stress.json"
        stress_off_path = exp_dir / "ecology_stress_cyclic_stress_no_evolution.json"
        if not traj_path.exists() or not stress_on_path.exists() or not stress_off_path.exists():
            print("  SKIP: evolution fallback inputs not found")
            return
        with open(traj_path, encoding="utf-8") as f:
            traj = json.load(f)

        def _from_aggregate(series: dict[str, dict]) -> dict[int, list[float]]:
            out: dict[int, list[float]] = defaultdict(list)
            for step_str, payload in series.items():
                out[int(step_str)].append(float(payload.get("alive_count_mean", 0.0)))
            return out

        def _from_raw_runs(path: Path) -> dict[int, list[float]]:
            out: dict[int, list[float]] = defaultdict(list)
            for run in load_json(path):
                for s in run.get("samples", []):
                    out[int(s["step"])].append(float(s["alive_count"]))
            return out

        cond_data = {
            "long_normal": _from_aggregate(traj.get("normal_no_crossover", {})),
            "long_no_evolution": _from_aggregate(traj.get("no_evolution", {})),
            "shift_normal": _from_raw_runs(stress_on_path),
            "shift_no_evolution": _from_raw_runs(stress_off_path),
        }
        panel_titles = ("Long run proxy (fitness trajectories)", "Cyclic stress comparison")
        shift_line = 1000

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.0), sharex=False)

    # Top: Long run (10K steps)
    ax = axes[0]
    for cond in ["long_normal", "long_no_evolution"]:
        label, color = conditions[cond]
        steps = sorted(cond_data[cond].keys())
        means = [np.mean(cond_data[cond][s]) for s in steps]
        sems = [
            np.std(cond_data[cond][s], ddof=1) / np.sqrt(len(cond_data[cond][s]))
            if len(cond_data[cond][s]) > 1
            else 0.0
            for s in steps
        ]
        means, sems = np.array(means), np.array(sems)
        ls = "-" if "normal" in cond and "no_" not in cond else "--"
        ax.plot(steps, means, color=color, linestyle=ls, label=label)
        ax.fill_between(steps, means - sems, means + sems, color=color, alpha=0.15)
    ax.set_ylabel("Alive Count")
    ax.set_title(panel_titles[0], fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bottom: Shift run (5K steps)
    ax = axes[1]
    for cond in ["shift_normal", "shift_no_evolution"]:
        label, color = conditions[cond]
        steps = sorted(cond_data[cond].keys())
        means = [np.mean(cond_data[cond][s]) for s in steps]
        sems = [
            np.std(cond_data[cond][s], ddof=1) / np.sqrt(len(cond_data[cond][s]))
            if len(cond_data[cond][s]) > 1
            else 0.0
            for s in steps
        ]
        means, sems = np.array(means), np.array(sems)
        ls = "-" if "normal" in cond and "no_" not in cond else "--"
        ax.plot(steps, means, color=color, linestyle=ls, label=label)
        ax.fill_between(steps, means - sems, means + sems, color=color, alpha=0.15)
    ax.axvline(x=shift_line, color="#888888", linestyle=":", linewidth=0.8, label="Stress mark")
    ax.set_xlabel("Step")
    ax.set_ylabel("Alive Count")
    ax.set_title(panel_titles[1], fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_evolution.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_evolution.pdf'}")
