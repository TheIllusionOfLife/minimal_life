"""Failure mode analysis for criterion-ablation experiments (Phase 1).

Computes per-step z-scores against a full-criteria baseline, detects
"first break" (first step where z < −2 sustained for ≥3 consecutive
samples), and derives cascade ordering via majority-vote across seeds.

Directly answers Reviewers A (Q3: failure mechanisms) and C (Q3: death cause
per ablation).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Z-score computation
# ---------------------------------------------------------------------------

def compute_z_scores(
    baseline_series: list[float],
    test_series: list[float],
) -> list[float]:
    """Compute per-step z-scores of *test_series* against *baseline_series*.

    Both lists must have the same length (one value per sample step).  The
    baseline mean and std are computed across the *entire* baseline series,
    then each test value is compared to that distribution.

    When baseline std is zero (constant baseline), any deviation from the
    mean yields z = ±(value − mean) / epsilon, clamped to avoid inf.
    """
    n = len(baseline_series)
    if n == 0:
        return []

    mean = sum(baseline_series) / n
    variance = sum((v - mean) ** 2 for v in baseline_series) / n
    std = math.sqrt(variance)

    # Guard against zero std: use a small epsilon so deviations still produce
    # finite z-scores proportional to the deviation magnitude.
    if std < 1e-12:
        std = 1e-6

    return [(test_series[i] - mean) / std for i in range(min(n, len(test_series)))]


# ---------------------------------------------------------------------------
# First-break detection
# ---------------------------------------------------------------------------

def detect_first_break(
    z_scores: list[float],
    steps: list[int],
    threshold: float = -2.0,
    sustained: int = 3,
) -> int | None:
    """Return the step at which z drops below *threshold* for *sustained*
    consecutive samples, or ``None`` if no sustained break is found.
    """
    run_length = 0
    run_start_step = None

    for i, z in enumerate(z_scores):
        if z < threshold:
            if run_length == 0:
                run_start_step = steps[i]
            run_length += 1
            if run_length >= sustained:
                return run_start_step
        else:
            run_length = 0
            run_start_step = None

    return None


# ---------------------------------------------------------------------------
# Cascade ordering
# ---------------------------------------------------------------------------

TRACKED_METRICS = ("energy_mean", "boundary_mean", "internal_state_mean_0", "waste_mean")


def detect_cascade_order(
    break_points: dict[str, int | None],
) -> list[tuple[str, int]]:
    """Sort metrics by their break step (ascending).  Metrics with no break
    are omitted.  Ties are preserved (both appear at the same step).
    """
    items = [(m, s) for m, s in break_points.items() if s is not None]
    items.sort(key=lambda x: (x[1], x[0]))
    return items


def majority_vote_cascade(
    per_seed_orders: list[list[tuple[str, int]]],
) -> list[str]:
    """Determine consensus cascade order across seeds via majority vote.

    The "first breaker" is the metric that appears first most often across
    seeds.  Ties in voting are broken alphabetically.
    """
    first_metric_counter: Counter[str] = Counter()
    for order in per_seed_orders:
        if order:
            first_step = order[0][1]
            # All metrics tied at the first step count as "first"
            for metric, step in order:
                if step == first_step:
                    first_metric_counter[metric] += 1
                else:
                    break

    if not first_metric_counter:
        return []

    # Build ordered list: most-common first, tie-break alphabetically
    ranked = sorted(first_metric_counter.items(), key=lambda x: (-x[1], x[0]))

    # For a complete ordering, also collect metrics that broke but weren't first
    all_metrics: Counter[str] = Counter()
    for order in per_seed_orders:
        for metric, _step in order:
            all_metrics[metric] += 1

    result = [m for m, _ in ranked]
    for metric, _ in sorted(all_metrics.items(), key=lambda x: (-x[1], x[0])):
        if metric not in result:
            result.append(metric)

    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_metric_series(
    runs: list[dict],
    metric: str,
) -> list[list[float]]:
    """Extract per-step values for *metric* from each seed's run.

    Returns a list of lists: outer = seeds, inner = steps.
    """
    all_series = []
    for run in runs:
        series = []
        for sample in run.get("samples", []):
            if metric == "internal_state_mean_0":
                val = sample.get("internal_state_mean", [0.0])[0]
            else:
                val = sample.get(metric, 0.0)
            series.append(float(val))
        all_series.append(series)
    return all_series


def _mean_across_seeds(per_seed_series: list[list[float]]) -> list[float]:
    """Compute mean at each step across seeds."""
    if not per_seed_series:
        return []
    n_steps = min(len(s) for s in per_seed_series)
    result = []
    for i in range(n_steps):
        vals = [s[i] for s in per_seed_series if i < len(s)]
        result.append(sum(vals) / len(vals) if vals else 0.0)
    return result


def _extract_steps(runs: list[dict]) -> list[int]:
    """Extract step numbers from the first run's samples."""
    if not runs or not runs[0].get("samples"):
        return []
    return [s["step"] for s in runs[0]["samples"]]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_failure_modes(
    data_dir: Path,
    conditions: list[str] | None = None,
    n_seeds: int = 20,
) -> dict:
    """Run the full failure-mode analysis pipeline.

    Args:
        data_dir: Directory containing per-seed JSON files
            (``full_seed{i}.json``, ``drop_{criterion}_seed{i}.json``).
        conditions: List of criterion names to analyze (default: all 7).
        n_seeds: Number of seeds per condition.

    Returns:
        Dict with per-condition cascade analysis results.
    """
    if conditions is None:
        conditions = [
            "metabolism", "boundary", "homeostasis", "response",
            "reproduction", "evolution", "growth",
        ]

    # Load baseline
    baseline_runs = []
    for i in range(n_seeds):
        path = data_dir / f"full_seed{i}.json"
        if not path.exists():
            continue
        with open(path) as f:
            baseline_runs.append(json.load(f))

    if not baseline_runs:
        return {"error": "no baseline data found"}

    # Compute baseline mean series per metric
    baseline_means: dict[str, list[float]] = {}
    for metric in TRACKED_METRICS:
        per_seed = _extract_metric_series(baseline_runs, metric)
        baseline_means[metric] = _mean_across_seeds(per_seed)

    results: dict = {}

    for condition in conditions:
        cond_runs = []
        for i in range(n_seeds):
            path = data_dir / f"drop_{condition}_seed{i}.json"
            if not path.exists():
                continue
            with open(path) as f:
                cond_runs.append(json.load(f))

        if not cond_runs:
            results[condition] = {"error": "no data found"}
            continue

        per_seed_results = []
        per_seed_cascades = []

        for seed_idx, run in enumerate(cond_runs):
            seed_break_points: dict[str, int | None] = {}

            for metric in TRACKED_METRICS:
                seed_series = _extract_metric_series([run], metric)[0]
                z_scores = compute_z_scores(baseline_means[metric], seed_series)
                seed_steps = _extract_steps([run])
                break_step = detect_first_break(z_scores, seed_steps)
                seed_break_points[metric] = break_step

            cascade = detect_cascade_order(seed_break_points)
            per_seed_results.append({
                "seed": seed_idx,
                "break_points": {k: v for k, v in seed_break_points.items()},
                "cascade": [[m, s] for m, s in cascade],
            })
            per_seed_cascades.append(cascade)

        consensus = majority_vote_cascade(per_seed_cascades)

        # Compute median break step per metric across seeds
        median_breaks: dict[str, int | None] = {}
        for metric in TRACKED_METRICS:
            break_steps = [
                r["break_points"][metric]
                for r in per_seed_results
                if r["break_points"].get(metric) is not None
            ]
            if break_steps:
                break_steps.sort()
                mid = len(break_steps) // 2
                median_breaks[metric] = break_steps[mid]
            else:
                median_breaks[metric] = None

        results[condition] = {
            "cascade_order": consensus,
            "median_break_steps": median_breaks,
            "per_seed": per_seed_results,
        }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Failure mode analysis for criterion-ablation experiments.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "experiments" / "ablation_sweep",
        help="Directory containing per-seed JSON files",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Criteria to analyze (default: all 7)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="Number of seeds per condition",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: experiments/failure_mode_analysis.json)",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "failure_mode_analysis.json"
        )

    result = analyze_failure_modes(
        data_dir=args.data_dir,
        conditions=args.conditions,
        n_seeds=args.n_seeds,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
