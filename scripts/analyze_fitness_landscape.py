"""Fitness landscape analysis (Phase 2 — peer review revision).

Tests two pre-specified hypotheses about evolution:
- H1: Directional selection (Jonckheere-Terpstra trend test)
- H2: Heritability (parent-offspring energy regression)

Directly strengthens the evolution criterion (Reviewer A Q1, M1).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Jonckheere-Terpstra trend test
# ---------------------------------------------------------------------------


def jonckheere_terpstra_test(
    groups: list[list[float]],
) -> tuple[float, float]:
    """Non-parametric test for ordered alternatives (one-sided increasing).

    Counts concordant pairs across ordered groups.  Uses normal
    approximation for p-value when groups are large enough.

    Returns (J statistic, p-value).  Returns (0, nan) for degenerate cases.
    """
    k = len(groups)
    if k < 2:
        return (0.0, float("nan"))

    # Count concordant pairs
    j_stat = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            for xi in groups[i]:
                for xj in groups[j]:
                    if xj > xi:
                        j_stat += 1.0
                    elif xj == xi:
                        j_stat += 0.5

    # Expected value and variance under null
    n_total = sum(len(g) for g in groups)
    ns = [len(g) for g in groups]

    e_j = (n_total * n_total - sum(ni * ni for ni in ns)) / 4.0

    # Variance formula with tie correction.
    # Ties within groups are counted and subtracted from the variance.
    a = n_total * (n_total - 1) * (2 * n_total + 5)
    b = sum(ni * (ni - 1) * (2 * ni + 5) for ni in ns)
    # Tie correction: count runs of identical values within each group
    tie_term = 0.0
    for g in groups:
        counts = Counter(g)
        for t in counts.values():
            if t > 1:
                tie_term += t * (t - 1) * (2 * t + 5)
    var_j = (a - b - tie_term) / 72.0

    if var_j <= 0:
        return (j_stat, float("nan"))

    z = (j_stat - e_j) / math.sqrt(var_j)

    # One-sided p-value (upper tail: increasing trend)
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return (j_stat, p)


# ---------------------------------------------------------------------------
# Parent-offspring regression
# ---------------------------------------------------------------------------


def parent_offspring_regression(
    parent_energies: list[float],
    offspring_energies: list[float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Compute parent-offspring regression slope with bootstrap 95% CI.

    Returns dict with slope, ci_lower, ci_upper, n_pairs.
    """
    n = min(len(parent_energies), len(offspring_energies))
    if n < 2:
        return {
            "slope": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_pairs": n,
        }

    def _slope(x: list[float], y: list[float]) -> float:
        m = len(x)
        mean_x = sum(x) / m
        mean_y = sum(y) / m
        ss_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(m))
        ss_xx = sum((x[i] - mean_x) ** 2 for i in range(m))
        if ss_xx < 1e-12:
            return 0.0
        return ss_xy / ss_xx

    px = parent_energies[:n]
    oy = offspring_energies[:n]
    observed_slope = _slope(px, oy)

    rng = random.Random(seed)
    boot_slopes = []
    indices = list(range(n))
    for _ in range(n_bootstrap):
        sample = rng.choices(indices, k=n)
        bx = [px[i] for i in sample]
        by = [oy[i] for i in sample]
        boot_slopes.append(_slope(bx, by))

    boot_slopes.sort()
    lo_idx = max(0, math.floor(0.025 * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, math.ceil(0.975 * n_bootstrap) - 1)

    return {
        "slope": observed_slope,
        "ci_lower": boot_slopes[lo_idx],
        "ci_upper": boot_slopes[hi_idx],
        "n_pairs": n,
    }


# ---------------------------------------------------------------------------
# Cohort segmentation
# ---------------------------------------------------------------------------


def segment_by_generation_cohort(
    organisms: list[dict],
    bins: list[int] | None = None,
) -> list[list[float]]:
    """Segment organisms into generation cohorts and return energy lists.

    Default bins: [0, 5, 10, 15] meaning cohorts [0-4], [5-9], [10-14], [15+].
    """
    if bins is None:
        bins = [0, 5, 10, 15]

    cohorts: list[list[float]] = [[] for _ in range(len(bins))]

    for org in organisms:
        gen = org.get("generation", 0)
        energy = org.get("energy", 0.0)
        # Reverse search finds the highest bin where gen >= bins[i].
        # With bins starting at 0, gen >= 0 always matches at minimum.
        for i in range(len(bins) - 1, -1, -1):
            if gen >= bins[i]:
                cohorts[i].append(energy)
                break

    return cohorts


# ---------------------------------------------------------------------------
# Lineage linkage
# ---------------------------------------------------------------------------


def link_parent_offspring_energies(
    snapshot_frames: list[dict],
    lineage_events: list[dict],
) -> tuple[list[float], list[float]]:
    """Link parent and offspring energies using lineage events and snapshots.

    For each lineage event, find the parent and child in the closest
    snapshot frame and extract their energy values.

    Returns (parent_energies, offspring_energies).
    """
    # Build stable_id → energy map from all snapshot frames.
    # NOTE: Uses last-seen energy for each organism (latest snapshot).
    # Birth-time energy is not available in snapshot data; this introduces
    # measurement noise that may attenuate the slope estimate (conservative).
    id_to_energy: dict[int, float] = {}
    for frame in snapshot_frames:
        for org in frame.get("organisms", []):
            sid = org.get("stable_id")
            if sid is not None:
                id_to_energy[sid] = org.get("energy", 0.0)

    parent_energies = []
    offspring_energies = []

    for event in lineage_events:
        parent_id = event.get("parent_stable_id")
        child_id = event.get("child_stable_id")
        if parent_id in id_to_energy and child_id in id_to_energy:
            parent_energies.append(id_to_energy[parent_id])
            offspring_energies.append(id_to_energy[child_id])

    return parent_energies, offspring_energies


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1 = sum(group1) / n1
    m2 = sum(group2) / n2
    var1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (m1 - m2) / pooled_std


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_fitness_landscape(
    data_dir: Path,
    n_seeds: int = 30,
) -> dict:
    """Run the fitness landscape analysis pipeline.

    Expects files: fitness_normal_seed{i}.json, fitness_no_evolution_seed{i}.json
    """
    evolved_runs = []
    clonal_runs = []

    for i in range(n_seeds):
        epath = data_dir / f"fitness_normal_seed{i}.json"
        if epath.exists():
            with open(epath) as f:
                evolved_runs.append(json.load(f))
        cpath = data_dir / f"fitness_no_evolution_seed{i}.json"
        if cpath.exists():
            with open(cpath) as f:
                clonal_runs.append(json.load(f))

    if not evolved_runs:
        return {"error": "no evolved data found"}

    # H1: Directional selection — cohort-level energy trend
    all_evolved_orgs = []
    for run in evolved_runs:
        for frame in run.get("organism_snapshots", []):
            all_evolved_orgs.extend(frame.get("organisms", []))

    cohorts = segment_by_generation_cohort(all_evolved_orgs)
    non_empty_cohorts = [c for c in cohorts if c]
    jt_stat, jt_p = jonckheere_terpstra_test(non_empty_cohorts)

    # Clonal control: should show no trend
    all_clonal_orgs = []
    for run in clonal_runs:
        for frame in run.get("organism_snapshots", []):
            all_clonal_orgs.extend(frame.get("organisms", []))

    clonal_cohorts = segment_by_generation_cohort(all_clonal_orgs)
    non_empty_clonal = [c for c in clonal_cohorts if c]
    clonal_jt_stat, clonal_jt_p = jonckheere_terpstra_test(non_empty_clonal)

    # H2: Heritability — parent-offspring regression
    all_parents = []
    all_offspring = []
    for run in evolved_runs:
        snapshots = run.get("organism_snapshots", [])
        lineage = run.get("lineage_events", [])
        p, o = link_parent_offspring_energies(snapshots, lineage)
        all_parents.extend(p)
        all_offspring.extend(o)

    regression = parent_offspring_regression(all_parents, all_offspring)

    # Effect size: evolved vs clonal final snapshot energies
    evolved_final = []
    for run in evolved_runs:
        snaps = run.get("organism_snapshots", [])
        if snaps:
            evolved_final.extend(o.get("energy", 0.0) for o in snaps[-1].get("organisms", []))
    clonal_final = []
    for run in clonal_runs:
        snaps = run.get("organism_snapshots", [])
        if snaps:
            clonal_final.extend(o.get("energy", 0.0) for o in snaps[-1].get("organisms", []))

    d = cohens_d(evolved_final, clonal_final)

    alpha = 0.05
    return {
        "alpha": alpha,
        "h1_reject": not math.isnan(jt_p) and jt_p < alpha,
        "h2_reject": (not math.isnan(regression["ci_lower"]) and regression["ci_lower"] > 0),
        "h1_trend_test": {
            "evolved": {
                "jt_statistic": jt_stat,
                "p_value": jt_p,
                "n_cohorts": len(non_empty_cohorts),
                "cohort_sizes": [len(c) for c in non_empty_cohorts],
                "cohort_means": [sum(c) / len(c) if c else 0.0 for c in non_empty_cohorts],
            },
            "clonal_control": {
                "jt_statistic": clonal_jt_stat,
                "p_value": clonal_jt_p,
                "n_cohorts": len(non_empty_clonal),
                "cohort_sizes": [len(c) for c in non_empty_clonal],
                "cohort_means": [sum(c) / len(c) if c else 0.0 for c in non_empty_clonal],
            },
        },
        "h2_regression": regression,
        "effect_size": {
            "cohens_d": d,
            "n_evolved": len(evolved_final),
            "n_clonal": len(clonal_final),
        },
        "n_evolved_seeds": len(evolved_runs),
        "n_clonal_seeds": len(clonal_runs),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fitness landscape analysis for evolution criterion.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "experiments",
        help="Directory containing niche result JSON files",
    )
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "fitness_landscape_analysis.json"
        )

    result = analyze_fitness_landscape(
        data_dir=args.data_dir,
        n_seeds=args.n_seeds,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
