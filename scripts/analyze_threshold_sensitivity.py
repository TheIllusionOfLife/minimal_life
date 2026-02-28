"""Threshold sensitivity analysis (Phase 4 — peer review revision).

Varies death thresholds and checks whether the rank ordering of ablation
effects remains stable (Reviewer A Q3).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------------

def _rank(values: list[float]) -> list[float]:
    """Assign ranks to values (1-based, average ties)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient.

    Returns nan for fewer than 2 data points.
    """
    n = min(len(x), len(y))
    if n < 2:
        return float("nan")

    rx = _rank(x[:n])
    ry = _rank(y[:n])

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x < 1e-12 or den_y < 1e-12:
        return float("nan")

    return num / (den_x * den_y)


# ---------------------------------------------------------------------------
# Delta% computation
# ---------------------------------------------------------------------------

def compute_delta_percent(
    baseline_alive: list[int],
    ablation_alive: list[int],
) -> float:
    """Compute Δ% = (mean_ablation - mean_baseline) / mean_baseline * 100."""
    if not baseline_alive or not ablation_alive:
        return float("nan")
    mean_base = sum(baseline_alive) / len(baseline_alive)
    mean_abl = sum(ablation_alive) / len(ablation_alive)
    if mean_base == 0:
        return 0.0
    return (mean_abl - mean_base) / mean_base * 100.0


# ---------------------------------------------------------------------------
# Rank stability assessment
# ---------------------------------------------------------------------------

def assess_rank_stability(
    deltas_by_combo: dict[str, dict[str, float]],
    reference_key: str | None = None,
) -> dict:
    """Assess rank stability of ablation effects across threshold combos.

    Computes Spearman rank correlation between the reference combo's
    delta ranking and each other combo's ranking.
    """
    if not deltas_by_combo:
        return {"spearman_correlations": {}, "all_stable": False}

    keys = list(deltas_by_combo.keys())
    if reference_key is None:
        reference_key = keys[0]

    ref_deltas = deltas_by_combo[reference_key]
    conditions = sorted(ref_deltas.keys())

    if len(conditions) < 2:
        return {"spearman_correlations": {}, "all_stable": True}

    ref_values = [ref_deltas.get(c, 0.0) for c in conditions]

    correlations = {}
    for key in keys:
        if key == reference_key:
            continue
        other_deltas = deltas_by_combo[key]
        other_values = [other_deltas.get(c, 0.0) for c in conditions]
        correlations[key] = spearman_rank_correlation(ref_values, other_values)

    all_stable = all(
        not math.isnan(r) and r >= 0.8 for r in correlations.values()
    )

    return {
        "reference_key": reference_key,
        "spearman_correlations": correlations,
        "all_stable": all_stable,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_threshold_sensitivity(
    data_dir: Path,
    boundary_thresholds: list[float] | None = None,
    energy_thresholds: list[float] | None = None,
    conditions: list[str] | None = None,
    n_seeds: int = 15,
) -> dict:
    """Run the threshold sensitivity analysis pipeline.

    Expects files: thresh_bt{bt}_et{et}_{cond}_seed{i}.json
    """
    if boundary_thresholds is None:
        boundary_thresholds = [0.05, 0.1, 0.15, 0.2]
    if energy_thresholds is None:
        energy_thresholds = [0.0, 0.05, 0.1]
    if conditions is None:
        conditions = [
            "normal",
            "no_metabolism",
            "no_reproduction",
            "no_response",
        ]

    deltas_by_combo: dict[str, dict[str, float]] = {}

    for bt in boundary_thresholds:
        for et in energy_thresholds:
            combo_key = f"bt{bt}_et{et}"

            # Load baseline (normal condition) for this threshold combo
            baseline_alive = []
            for seed in range(n_seeds):
                fname = f"thresh_bt{bt}_et{et}_normal_seed{seed}.json"
                path = data_dir / fname
                if path.exists():
                    with open(path) as f:
                        run = json.load(f)
                    baseline_alive.append(run.get("final_alive_count", 0))

            if not baseline_alive:
                continue

            combo_deltas: dict[str, float] = {}
            for cond in conditions:
                if cond == "normal":
                    continue
                abl_alive = []
                for seed in range(n_seeds):
                    fname = f"thresh_bt{bt}_et{et}_{cond}_seed{seed}.json"
                    path = data_dir / fname
                    if path.exists():
                        with open(path) as f:
                            run = json.load(f)
                        abl_alive.append(run.get("final_alive_count", 0))

                if abl_alive:
                    combo_deltas[cond] = compute_delta_percent(
                        baseline_alive, abl_alive,
                    )

            if combo_deltas:
                deltas_by_combo[combo_key] = combo_deltas

    # Find the default threshold combo as reference
    default_key = "bt0.1_et0.0"
    if default_key not in deltas_by_combo and deltas_by_combo:
        default_key = next(iter(deltas_by_combo))

    rank_stability = assess_rank_stability(deltas_by_combo, default_key)

    return {
        "deltas_by_combo": deltas_by_combo,
        "rank_stability": rank_stability,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threshold sensitivity analysis.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "threshold_sweep"
        ),
    )
    parser.add_argument("--n-seeds", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "threshold_sensitivity_analysis.json"
        )

    result = analyze_threshold_sensitivity(
        data_dir=args.data_dir,
        n_seeds=args.n_seeds,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
