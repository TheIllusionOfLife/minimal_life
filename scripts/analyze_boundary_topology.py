"""Analyze boundary topology experiment results (Phase 5 — peer review revision).

Compares ablation effect hierarchies between toroidal and bounded world
topologies to test whether findings generalize (Reviewer C Q4).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy import stats


def mann_whitney_comparison(
    a: list[float], b: list[float],
) -> dict:
    """Run Mann-Whitney U test comparing two groups.

    Returns dict with U statistic, p-value, and sample sizes.
    """
    if len(a) < 2 or len(b) < 2:
        return {"U": float("nan"), "p_value": float("nan"), "n_a": len(a), "n_b": len(b)}

    u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(u_stat), "p_value": float(p_val), "n_a": len(a), "n_b": len(b)}


def compute_topology_deltas(
    data: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, float]]:
    """Compute Δ% for each ablation condition relative to its topology's baseline.

    Args:
        data: {topology: {condition: [alive_counts]}}

    Returns:
        {topology: {condition: delta_percent}} (excluding 'normal' baseline).
    """
    result: dict[str, dict[str, float]] = {}
    for topo, conditions in data.items():
        baseline_counts = conditions.get("normal", [])
        baseline_mean = sum(baseline_counts) / len(baseline_counts) if baseline_counts else 0.0
        deltas: dict[str, float] = {}
        for cond, counts in conditions.items():
            if cond == "normal":
                continue
            cond_mean = sum(counts) / len(counts) if counts else 0.0
            if baseline_mean == 0:
                deltas[cond] = float("nan")
            else:
                deltas[cond] = ((cond_mean - baseline_mean) / baseline_mean) * 100.0
        result[topo] = deltas
    return result


def compare_rank_ordering(
    deltas_a: dict[str, float],
    deltas_b: dict[str, float],
) -> dict:
    """Compare rank ordering of ablation effects between two topologies.

    Uses Spearman rank correlation on the shared conditions.

    Returns:
        Dict with spearman_rho, p_value, ranks_match (rho >= 0.9), conditions.
    """
    shared = sorted(set(deltas_a.keys()) & set(deltas_b.keys()))
    # Filter out NaN values
    shared = [c for c in shared if not math.isnan(deltas_a[c]) and not math.isnan(deltas_b[c])]

    if len(shared) < 3:
        return {
            "spearman_rho": float("nan"),
            "p_value": float("nan"),
            "ranks_match": False,
            "n_conditions": len(shared),
        }

    vals_a = [deltas_a[c] for c in shared]
    vals_b = [deltas_b[c] for c in shared]

    rho, p_val = stats.spearmanr(vals_a, vals_b)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_val),
        "ranks_match": bool(rho >= 0.9),
        "n_conditions": len(shared),
        "conditions": shared,
    }


def analyze_boundary_topology(
    data_dir: Path | None = None,
    n_seeds: int = 30,
    output_path: Path | None = None,
) -> dict:
    """Run the full boundary topology analysis pipeline.

    Reads experiment JSON files, computes deltas, compares rank orderings,
    and runs Mann-Whitney U tests per condition.

    Args:
        data_dir: Directory containing boundary_{topo}_{cond}_seed{N}.json files.
        n_seeds: Number of seeds expected per condition.
        output_path: Where to write the analysis JSON (optional).

    Returns:
        Analysis results dict.
    """
    if data_dir is None:
        from _project_root import PROJECT_ROOT

        data_dir = PROJECT_ROOT / "experiments" / "boundary_sweep"

    topologies = ["toroidal", "bounded"]

    # Collect alive counts: {topology: {condition: [alive_counts]}}
    alive_data: dict[str, dict[str, list[float]]] = {}
    conditions_seen: set[str] = set()

    for topo in topologies:
        topo_data: dict[str, list[float]] = {}
        # Discover conditions from files
        for f in sorted(data_dir.glob(f"boundary_{topo}_*_seed*.json")):
            parts = f.stem.split("_")
            # boundary_{topo}_{cond}_seed{N}
            # Find 'seed' part index
            seed_idx = next(i for i, p in enumerate(parts) if p.startswith("seed"))
            cond = "_".join(parts[2:seed_idx])
            conditions_seen.add(cond)

            with open(f) as fh:
                result = json.load(fh)
            alive = result.get("final_alive_count", 0)
            topo_data.setdefault(cond, []).append(float(alive))
        alive_data[topo] = topo_data

    # Compute deltas
    deltas = compute_topology_deltas(alive_data)

    # Compare rank ordering between topologies
    rank_comparison = {}
    if "toroidal" in deltas and "bounded" in deltas:
        rank_comparison = compare_rank_ordering(deltas["toroidal"], deltas["bounded"])

    # Mann-Whitney U per condition (toroidal vs bounded)
    mw_results: dict[str, dict] = {}
    for cond in sorted(conditions_seen):
        toro_counts = alive_data.get("toroidal", {}).get(cond, [])
        bounded_counts = alive_data.get("bounded", {}).get(cond, [])
        mw_results[cond] = mann_whitney_comparison(toro_counts, bounded_counts)

    analysis = {
        "deltas_by_topology": deltas,
        "rank_comparison": rank_comparison,
        "mann_whitney_by_condition": mw_results,
        "alive_summary": {
            topo: {
                cond: {
                    "mean": sum(counts) / len(counts) if counts else 0,
                    "n": len(counts),
                }
                for cond, counts in conds.items()
            }
            for topo, conds in alive_data.items()
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)

    return analysis


if __name__ == "__main__":
    from _project_root import PROJECT_ROOT

    result = analyze_boundary_topology(
        output_path=PROJECT_ROOT / "experiments" / "boundary_topology_analysis.json",
    )
    print(json.dumps(result, indent=2))
