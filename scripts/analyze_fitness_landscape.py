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
from pathlib import Path

import numpy as np
from analyses.results.statistics import jonckheere_terpstra

# ---------------------------------------------------------------------------
# Jonckheere-Terpstra trend test (delegates to shared implementation)
# ---------------------------------------------------------------------------


def jonckheere_terpstra_test(
    groups: list[list[float]],
) -> tuple[float, float]:
    """Non-parametric test for ordered alternatives (one-sided increasing).

    Returns (J statistic, p-value).  Returns (0, nan) for degenerate cases.
    The J statistic counts concordant pairs where later-group values exceed
    earlier-group values (increasing convention).
    """
    if len(groups) < 2:
        return (0.0, float("nan"))

    np_groups = [np.array(g) for g in groups]
    stat_dec, p_two = jonckheere_terpstra(np_groups)

    # Shared implementation uses "earlier > later" (decreasing) convention.
    # Convert to "later > earlier" (increasing) convention:
    #   J_inc + J_dec = total_pairs  (ties get 0.5 credit in both)
    k = len(np_groups)
    total_pairs = sum(
        len(np_groups[i]) * len(np_groups[j]) for i in range(k) for j in range(i + 1, k)
    )
    stat_inc = total_pairs - stat_dec

    # Two-sided p is symmetric; for one-sided increasing:
    # If stat_inc > E[J] (= total_pairs/2), trend is increasing → p = p_two/2
    # Otherwise, trend is not increasing → p = 1 - p_two/2
    e_j = total_pairs / 2.0
    if stat_inc >= e_j:
        p_one = p_two / 2.0
    else:
        p_one = 1.0 - p_two / 2.0

    return (float(stat_inc), float(p_one))


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
    seed_start: int = 100,
) -> dict:
    """Run the fitness landscape analysis pipeline.

    Expects files: fitness_normal_seed{i}.json, fitness_no_evolution_seed{i}.json
    """
    evolved_runs = []
    clonal_runs = []

    for i in range(seed_start, seed_start + n_seeds):
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
    per_seed_parents: list[list[float]] = []
    per_seed_offspring: list[list[float]] = []
    for run in evolved_runs:
        snapshots = run.get("organism_snapshots", [])
        lineage = run.get("lineage_events", [])
        p, o = link_parent_offspring_energies(snapshots, lineage)
        all_parents.extend(p)
        all_offspring.extend(o)
        per_seed_parents.append(p)
        per_seed_offspring.append(o)

    regression = parent_offspring_regression(all_parents, all_offspring)
    clustered = parent_offspring_regression_clustered(per_seed_parents, per_seed_offspring)

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
        "h2_regression_clustered": clustered,
        "effect_size": {
            "cohens_d": d,
            "n_evolved": len(evolved_final),
            "n_clonal": len(clonal_final),
        },
        "n_evolved_seeds": len(evolved_runs),
        "n_clonal_seeds": len(clonal_runs),
    }


# ---------------------------------------------------------------------------
# Cluster-robust bootstrap CI for heritability
# ---------------------------------------------------------------------------


def parent_offspring_regression_clustered(
    per_seed_parents: list[list[float]],
    per_seed_offspring: list[list[float]],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Seed-level cluster-robust bootstrap CI for the heritability slope.

    Standard pair-level bootstrap (in parent_offspring_regression) ignores
    within-seed correlation, producing anti-conservative CIs when seeds
    are the true independent unit.  This function resamples at the seed level:
    resample n_seeds seeds with replacement, pool all pairs within the
    selected seeds, then compute the OLS slope.

    Args:
        per_seed_parents:   List of parent-energy lists, one list per seed.
        per_seed_offspring: Matching offspring-energy lists.
        n_bootstrap:        Number of seed-level bootstrap replicates.
        seed:               RNG seed for reproducibility.

    Returns dict with slope, naive_ci (from pair-level), cluster_ci, n_seeds,
    n_pairs.
    """
    n_seeds = len(per_seed_parents)
    if n_seeds < 2:
        return {
            "slope": float("nan"),
            "cluster_ci_lower": float("nan"),
            "cluster_ci_upper": float("nan"),
            "n_seeds": n_seeds,
            "n_pairs": 0,
        }

    # Pool all pairs for the observed slope
    all_px: list[float] = []
    all_oy: list[float] = []
    for px, oy in zip(per_seed_parents, per_seed_offspring, strict=True):
        n = min(len(px), len(oy))
        all_px.extend(px[:n])
        all_oy.extend(oy[:n])

    n_pairs = len(all_px)

    def _slope(x: list[float], y: list[float]) -> float:
        m = len(x)
        if m < 2:
            return float("nan")
        mean_x = sum(x) / m
        mean_y = sum(y) / m
        ss_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(m))
        ss_xx = sum((x[i] - mean_x) ** 2 for i in range(m))
        if ss_xx < 1e-12:
            return float("nan")
        return ss_xy / ss_xx

    observed_slope = _slope(all_px, all_oy)

    # Seed-level bootstrap
    rng = random.Random(seed)
    seed_indices = list(range(n_seeds))
    boot_slopes: list[float] = []
    for _ in range(n_bootstrap):
        chosen = rng.choices(seed_indices, k=n_seeds)
        bx: list[float] = []
        by: list[float] = []
        for idx in chosen:
            px = per_seed_parents[idx]
            oy = per_seed_offspring[idx]
            n = min(len(px), len(oy))
            bx.extend(px[:n])
            by.extend(oy[:n])
        s = _slope(bx, by)
        if not math.isnan(s):
            boot_slopes.append(s)

    boot_slopes.sort()
    nb = len(boot_slopes)
    lo_idx = max(0, math.floor(0.025 * nb))
    hi_idx = min(nb - 1, math.ceil(0.975 * nb) - 1)

    return {
        "slope": observed_slope,
        "cluster_ci_lower": boot_slopes[lo_idx] if boot_slopes else float("nan"),
        "cluster_ci_upper": boot_slopes[hi_idx] if boot_slopes else float("nan"),
        "n_seeds": n_seeds,
        "n_pairs": n_pairs,
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
    parser.add_argument("--seed-start", type=int, default=100)
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
        seed_start=args.seed_start,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
