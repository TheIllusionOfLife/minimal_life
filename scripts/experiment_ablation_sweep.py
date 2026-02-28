"""Full criterion-ablation sweep for Phase 1 reduction analysis.

Generates all single (7) and pairwise (21) criterion dropout conditions,
plus the full-system baseline and all-off negative control.

Total: 30 conditions × 20 seeds = 600 runs.
Each run: 2000 steps, sample_every=20 → 100 StepMetrics samples.

Output: experiments/ablation_sweep/{condition}_seed{seed}.json

Usage:
    # Dry run — print all conditions without running simulation:
    uv run python scripts/experiment_ablation_sweep.py --dry-run

    # Quick validation (2 conditions × 2 seeds):
    uv run python scripts/experiment_ablation_sweep.py \
        --dry-run --conditions full drop_metabolism --seeds 0 1

    # Full sweep (~20 min on M2 Pro):
    uv run python scripts/experiment_ablation_sweep.py

Verification:
    python -c "
    import json
    d = json.load(open('experiments/ablation_sweep/full_seed0.json'))
    print(len(d['samples']), 'samples')
    "
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import minimal_life
import numpy as np
from experiment_common import CRITERION_TO_FLAG, log, make_config_dict

# ── constants ─────────────────────────────────────────────────────────────────

CRITERIA = [
    "metabolism",
    "boundary",
    "homeostasis",
    "response",
    "reproduction",
    "evolution",
    "growth",
]

SEEDS = list(range(20))  # discovery seeds (calibration set)
STEPS = 2000
SAMPLE_EVERY = 20  # finer sampling than existing experiments (which use 50)
OUT_SUBDIR = "ablation_sweep"


# ── condition generation ──────────────────────────────────────────────────────


def ablation_conditions(
    include_triples: bool = False,
) -> list[tuple[str, list[str]]]:
    """Return ordered list of (condition_name, disabled_criteria) pairs.

    Default: 30 conditions (1 all_off + 1 full + 7 single + 21 pairwise).
    With include_triples=True: +35 triple ablations (65 total).

    The disabled_criteria list contains short criterion names (e.g. "metabolism"),
    which map to SimConfig flags via CRITERION_TO_FLAG.
    """
    result: list[tuple[str, list[str]]] = []
    result.append(("all_off", list(CRITERIA)))  # negative-control floor
    result.append(("full", []))  # all-criteria ceiling
    for c in CRITERIA:
        result.append((f"drop_{c}", [c]))
    for c1, c2 in combinations(CRITERIA, 2):
        result.append((f"drop_{c1}_{c2}", [c1, c2]))
    if include_triples:
        for c1, c2, c3 in combinations(CRITERIA, 3):
            result.append((f"drop_{c1}_{c2}_{c3}", [c1, c2, c3]))
    return result


# ── config construction ───────────────────────────────────────────────────────


def make_ablation_config(disabled: list[str], seed: int) -> dict:
    """Build a config dict with specified criteria disabled.

    Args:
        disabled: List of short criterion names to disable (e.g. ["metabolism"]).
        seed: Random seed for this run.

    Returns:
        Config dict ready for json.dumps + minimal_life.run_experiment_json.
    """
    overrides = {CRITERION_TO_FLAG[c]: False for c in disabled}
    return make_config_dict(seed, overrides)


# ── criterion presence matrix (shared with analysis scripts) ──────────────────


def build_criterion_presence_matrix(
    condition_names: list[str],
    criteria: list[str],
) -> np.ndarray:
    """Build binary matrix (n_conditions × n_criteria) encoding enabled criteria.

    Entry [i, j] = 1 if criterion j is enabled in condition i, else 0.

    'full'    → all 1s.
    'all_off' → all 0s.
    'drop_c'  → 0 at criterion c's index, 1 elsewhere.
    'drop_c1_c2' → 0 at both indices, 1 elsewhere.
    """
    import numpy as np

    n_cond = len(condition_names)
    n_crit = len(criteria)
    presence = np.ones((n_cond, n_crit), dtype=float)
    # Hoisted: build once, reuse for every drop_ condition.
    criteria_to_idx = {c: j for j, c in enumerate(criteria)}

    for i, name in enumerate(condition_names):
        if name == "all_off":
            presence[i, :] = 0.0
        elif name == "full":
            presence[i, :] = 1.0
        elif name.startswith("drop_"):
            # All criterion names are single words (no underscores), so a simple
            # split unambiguously recovers the disabled criteria.
            tokens = name[len("drop_") :].split("_")
            for token in tokens:
                j = criteria_to_idx.get(token)
                if j is not None:
                    presence[i, j] = 0.0
                else:
                    warnings.warn(
                        f"build_criterion_presence_matrix: unrecognized token "
                        f"'{token}' in condition '{name}' (index {i}); skipped.",
                        stacklevel=2,
                    )
    return presence


# ── performance metrics (shared with analyze_criterion_reduction.py) ──────────


def compute_performance_metrics(
    samples: list[dict],
    total_steps: int,
) -> dict[str, float]:
    """Compute scalar performance metrics from one run's StepMetrics list.

    Args:
        samples: Ordered list of StepMetrics dicts for one seed.
        total_steps: Simulation step count (for normalization).

    Returns:
        Dict with keys: alive_auc, energy_stability, boundary_integrity,
        homeostasis_quality, reproduction_rate, genome_diversity_late,
        spatial_cohesion.
    """
    import numpy as np

    try:
        _trapz = np.trapezoid  # type: ignore[attr-defined]  # NumPy ≥ 2.0
    except AttributeError:
        _trapz = np.trapz  # type: ignore[attr-defined]  # NumPy < 2.0

    zero = {
        "alive_auc": 0.0,
        "energy_stability": 0.0,
        "boundary_integrity": 0.0,
        "homeostasis_quality": 0.0,
        "reproduction_rate": 0.0,
        "genome_diversity_late": 0.0,
        "spatial_cohesion": 0.0,
    }
    if not samples:
        return zero

    steps_arr = [s["step"] for s in samples]
    alive_arr = [s["alive_count"] for s in samples]
    alive_auc = float(_trapz(alive_arr, steps_arr) / max(total_steps, 1))

    n_late = min(50, len(samples))
    late = samples[-n_late:]

    energy_stability = float(sum(s["energy_mean"] for s in late) / n_late)
    boundary_integrity = float(sum(s["boundary_mean"] for s in late) / n_late)

    # internal_state_std is [f32; 4] in Rust → 4-element list in JSON
    ist_means = [float(sum(s.get("internal_state_std", [0.0, 0.0, 0.0, 0.0])) / 4.0) for s in late]
    mean_ist = sum(ist_means) / max(len(ist_means), 1)
    homeostasis_quality = 1.0 / (mean_ist + 1e-6)

    total_births = sum(s["birth_count"] for s in samples)
    reproduction_rate = float(total_births) / max(total_steps, 1)

    genome_diversity_late = float(sum(s.get("genome_diversity", 0.0) for s in late) / n_late)
    spatial_cohesion = float(sum(s.get("spatial_cohesion_mean", 0.0) for s in late) / n_late)

    return {
        "alive_auc": alive_auc,
        "energy_stability": energy_stability,
        "boundary_integrity": boundary_integrity,
        "homeostasis_quality": homeostasis_quality,
        "reproduction_rate": reproduction_rate,
        "genome_diversity_late": genome_diversity_late,
        "spatial_cohesion": spatial_cohesion,
    }


# ── sweep runner ──────────────────────────────────────────────────────────────


def run_ablation_sweep(
    conditions: list[tuple[str, list[str]]],
    seeds: list[int],
    dry_run: bool,
    out_dir: Path,
) -> None:
    """Run specified conditions and save per-seed JSON files.

    Cached runs (output file already exists) are skipped automatically,
    so the sweep is safe to resume after interruption.

    Args:
        conditions: List of (condition_name, disabled_criteria) pairs.
        seeds: List of seed integers to run.
        dry_run: If True, print plan without running simulation.
        out_dir: Root experiments directory. Output goes to out_dir/ablation_sweep/.
    """
    sweep_dir = out_dir / OUT_SUBDIR
    sweep_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(conditions) * len(seeds)
    log(f"Ablation sweep: {len(conditions)} conditions × {len(seeds)} seeds = {total_runs} runs")
    log(f"Output directory: {sweep_dir}")

    if dry_run:
        log("DRY RUN — no simulation executed")
        log("")
        for cond_name, disabled in conditions:
            log(f"  {cond_name}: disable={disabled}")
        return

    total_start = time.perf_counter()
    run_count = 0
    skip_count = 0

    for cond_name, disabled in conditions:
        log(f"--- Condition: {cond_name} (disable={disabled}) ---")
        cond_start = time.perf_counter()

        for seed in seeds:
            out_path = sweep_dir / f"{cond_name}_seed{seed}.json"
            if out_path.exists():
                log(f"  seed={seed:3d}  [cached — skipped]")
                skip_count += 1
                run_count += 1
                continue

            cfg = make_ablation_config(disabled, seed)
            t0 = time.perf_counter()
            result_json = minimal_life.run_experiment_json(json.dumps(cfg), STEPS, SAMPLE_EVERY)
            result = json.loads(result_json)
            elapsed = time.perf_counter() - t0

            with open(out_path, "w") as f:
                json.dump(result, f)

            log(
                f"  seed={seed:3d}  alive={result['final_alive_count']:4d}"
                f"  samples={len(result.get('samples', []))}  {elapsed:.2f}s"
            )
            run_count += 1

        log(f"  Condition time: {time.perf_counter() - cond_start:.1f}s")
        log("")

    elapsed_total = time.perf_counter() - total_start
    log(f"Done: {run_count} runs ({skip_count} cached) in {elapsed_total:.1f}s")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full criterion-ablation sweep (30 conditions × 20 seeds)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print condition plan without running simulation.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        metavar="COND",
        help="Subset of condition names to run (e.g. full drop_metabolism).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        metavar="SEED",
        help="Subset of seed integers to run (default: 0–19).",
    )
    parser.add_argument(
        "--triples",
        action="store_true",
        help="Also generate triple ablations (35 additional conditions).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    log(f"Minimal Life v{minimal_life.version()}")
    log(f"Steps={STEPS}, sample_every={SAMPLE_EVERY}")
    log("")

    all_conditions = ablation_conditions(include_triples=args.triples)

    if args.conditions is not None:
        cond_names_set = set(args.conditions)
        all_conditions = [(n, d) for n, d in all_conditions if n in cond_names_set]
        missing = cond_names_set - {n for n, _ in all_conditions}
        if missing:
            log(f"WARNING: unknown condition names: {missing}")

    seeds = args.seeds if args.seeds is not None else SEEDS

    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    out_dir.mkdir(exist_ok=True)

    run_ablation_sweep(all_conditions, seeds, dry_run=args.dry_run, out_dir=out_dir)


if __name__ == "__main__":
    main()
