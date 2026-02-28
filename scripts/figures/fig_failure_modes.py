"""Failure mode analysis figure (Phase 1 — peer review revision).

3-panel figure showing normalized time-series for the "essential triad"
(metabolism, response, reproduction) with ±1 SEM bands and vertical
markers at median "first break" step per metric.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from _project_root import PROJECT_ROOT
from figures._shared import FIG_DIR

ANALYSIS_PATH = PROJECT_ROOT / "experiments" / "failure_mode_analysis.json"

# Triad panels: criterion → display title
TRIAD_PANELS = {
    "metabolism": "Metabolism Ablation",
    "response": "Response Ablation",
    "reproduction": "Reproduction Ablation",
}

METRIC_STYLES = {
    "energy_mean": {"color": "#E69F00", "label": "Energy", "marker": "o"},
    "boundary_mean": {"color": "#56B4E9", "label": "Boundary", "marker": "s"},
    "internal_state_mean_0": {"color": "#009E73", "label": "Internal State", "marker": "^"},
    "waste_mean": {"color": "#D55E00", "label": "Waste", "marker": "D"},
}

DATA_DIR = PROJECT_ROOT / "experiments" / "ablation_sweep"


def _load_metric_series(
    runs: list[dict],
    metric: str,
) -> tuple[list[int], np.ndarray]:
    """Load per-seed metric series and return (steps, 2D array[seeds, steps])."""
    all_series = []
    steps = None
    for run in runs:
        series = []
        if steps is None:
            steps = [s["step"] for s in run.get("samples", [])]
        for sample in run.get("samples", []):
            if metric == "internal_state_mean_0":
                val = sample.get("internal_state_mean", [0.0])[0]
            else:
                val = sample.get(metric, 0.0)
            series.append(float(val))
        all_series.append(series)
    if not all_series:
        return [], np.array([])
    min_len = min(len(s) for s in all_series)
    arr = np.array([s[:min_len] for s in all_series])
    return (steps or [])[:min_len], arr


def _load_condition_runs(condition: str, n_seeds: int = 20) -> list[dict]:
    """Load per-seed runs for a drop condition."""
    runs = []
    for i in range(n_seeds):
        path = DATA_DIR / f"drop_{condition}_seed{i}.json"
        if path.exists():
            with open(path) as f:
                runs.append(json.load(f))
    return runs


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize each seed's series to [0, 1]."""
    mins = arr.min(axis=1, keepdims=True)
    maxs = arr.max(axis=1, keepdims=True)
    rng = maxs - mins
    rng[rng < 1e-12] = 1.0
    return (arr - mins) / rng


def generate_failure_modes() -> None:
    """Generate the 3-panel failure mode figure."""
    # Load analysis results for median break steps
    analysis = {}
    if ANALYSIS_PATH.exists():
        with open(ANALYSIS_PATH) as f:
            analysis = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)

    for ax, (condition, title) in zip(axes, TRIAD_PANELS.items(), strict=True):
        runs = _load_condition_runs(condition)
        if not runs:
            ax.set_title(title)
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="gray",
            )
            continue

        for metric, style in METRIC_STYLES.items():
            steps, arr = _load_metric_series(runs, metric)
            if arr.size == 0:
                continue
            normed = _normalize(arr)
            mean = normed.mean(axis=0)
            sem = normed.std(axis=0) / np.sqrt(normed.shape[0])

            ax.plot(
                steps,
                mean,
                color=style["color"],
                label=style["label"],
                linewidth=1.2,
                marker=style["marker"],
                markersize=2,
                markevery=max(1, len(steps) // 8),
            )
            ax.fill_between(steps, mean - sem, mean + sem, color=style["color"], alpha=0.15)

        # Vertical markers at median break steps
        cond_analysis = analysis.get(condition, {})
        median_breaks = cond_analysis.get("median_break_steps", {})
        for metric, style in METRIC_STYLES.items():
            step = median_breaks.get(metric)
            if step is not None:
                ax.axvline(step, color=style["color"], linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Step")
        if ax is axes[0]:
            ax.set_ylabel("Normalized Value")

    axes[-1].legend(loc="upper right", fontsize=6, framealpha=0.8)
    fig.tight_layout()

    out_path = FIG_DIR / "fig_failure_modes.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")
