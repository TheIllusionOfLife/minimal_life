"""Phase 2 surrogate expansion sweep (≥100 conditions, regime-diverse).

Two orthogonal axes:
  Axis 1 (~40 ablation conditions): single, pairwise, graded, mid-run
  Axis 2 (5 environment regimes): default, scarce, patchy, cyclic, dense

Total: ~200 conditions × 40 seeds avg = ~8000 runs.
Uses ProcessPoolExecutor for parallelism (~60 min on M2 Pro 10 cores).

3-layer regime-based split:
  Train:    Regimes A-C × all ablations, seeds 0-39
  Validate: Regimes A-C × all ablations, seeds 40-69
  Test:     Regimes D-E × all ablations, seeds 0-69 (unseen regimes)

Usage:
    uv run python scripts/experiment_phase2_sweep.py > experiments/phase2_data.tsv

Partial (quick validation):
    uv run python scripts/experiment_phase2_sweep.py --partial > experiments/phase2_partial.tsv
"""

from __future__ import annotations

import sys
import time
from itertools import combinations
from pathlib import Path

import minimal_life
from experiment_common import (
    CRITERION_TO_FLAG,
    PAIRS,
    experiment_output_dir,
    log,
    run_condition_suite_parallel,
)

STEPS = 2000
SAMPLE_EVERY = 50

# --- Axis 1: Ablation conditions ---

# Single ablations (8)
SINGLE_CONDITIONS: dict[str, dict] = {
    "normal": {},
    **{f"no_{c}": {flag: False} for c, flag in CRITERION_TO_FLAG.items()},
}

# Pairwise ablations (from PAIRS)
PAIRWISE_CONDITIONS: dict[str, dict] = {}
for c1, c2 in PAIRS:
    name = f"no_{c1}_{c2}"
    overrides = {CRITERION_TO_FLAG[c1]: False, CRITERION_TO_FLAG[c2]: False}
    PAIRWISE_CONDITIONS[name] = overrides

# Additional pairwise combinations not in PAIRS
ALL_PAIR_COMBOS = list(combinations(CRITERION_TO_FLAG.keys(), 2))
for c1, c2 in ALL_PAIR_COMBOS:
    name = f"no_{c1}_{c2}"
    if name not in PAIRWISE_CONDITIONS:
        overrides = {CRITERION_TO_FLAG[c1]: False, CRITERION_TO_FLAG[c2]: False}
        PAIRWISE_CONDITIONS[name] = overrides

# Graded ablations (metabolism efficiency at 0.25, 0.50, 0.75)
GRADED_CONDITIONS: dict[str, dict] = {
    "metabolism_eff_025": {"metabolism_efficiency_multiplier": 0.25},
    "metabolism_eff_050": {"metabolism_efficiency_multiplier": 0.50},
    "metabolism_eff_075": {"metabolism_efficiency_multiplier": 0.75},
}

# Mid-run ablations (ablation at step 1000)
MIDRUN_CONDITIONS: dict[str, dict] = {}
for criterion in CRITERION_TO_FLAG:
    name = f"midrun_{criterion}"
    MIDRUN_CONDITIONS[name] = {
        "ablation_step": 1000,
        "ablation_targets": [criterion],
    }

# Combine all ablation conditions
ALL_ABLATION_CONDITIONS: dict[str, dict] = {
    **SINGLE_CONDITIONS,
    **PAIRWISE_CONDITIONS,
    **GRADED_CONDITIONS,
    **MIDRUN_CONDITIONS,
}

# --- Axis 2: Environment regimes ---

REGIMES: dict[str, dict] = {
    "regime_a": {},  # Default
    "regime_b": {"resource_regeneration_rate": 0.003},  # Scarce
    "regime_c": {  # Patchy (high regen + environment shift)
        "resource_regeneration_rate": 0.02,
        "environment_shift_step": 500,
        "environment_shift_resource_rate": 0.005,
    },
    "regime_d": {  # Cyclic stress
        "environment_cycle_period": 500,
        "environment_cycle_low_rate": 0.002,
    },
    "regime_e": {  # Dense
        "num_organisms": 100,
        "world_size": 80.0,
        "agents_per_organism": 25,
    },
}

# 3-layer split by regime
TRAIN_REGIMES = ["regime_a", "regime_b", "regime_c"]
TRAIN_SEEDS = list(range(0, 40))
VALIDATE_SEEDS = list(range(40, 70))
TEST_REGIMES = ["regime_d", "regime_e"]
TEST_SEEDS = list(range(0, 70))

GRAPH_OVERRIDES = {"metabolism_mode": "graph"}


def build_full_conditions() -> dict[str, dict]:
    """Build all (regime × ablation) condition combinations."""
    conditions: dict[str, dict] = {}
    for regime_name, regime_ov in REGIMES.items():
        for abl_name, abl_ov in ALL_ABLATION_CONDITIONS.items():
            cond_name = f"{regime_name}__{abl_name}"
            # Merge: regime overrides + ablation overrides + graph mode
            combined = {**GRAPH_OVERRIDES, **regime_ov, **abl_ov}
            conditions[cond_name] = combined
    return conditions


def build_partial_conditions(n_ablations: int = 20, n_seeds: int = 10) -> tuple[dict, list[int]]:
    """Build a subset for quick validation."""
    # Take first n_ablations from ALL_ABLATION_CONDITIONS
    partial_ablations = dict(list(ALL_ABLATION_CONDITIONS.items())[:n_ablations])
    conditions: dict[str, dict] = {}
    # Only use regime_a and regime_b for partial
    for regime_name in ["regime_a", "regime_b"]:
        regime_ov = REGIMES[regime_name]
        for abl_name, abl_ov in partial_ablations.items():
            cond_name = f"{regime_name}__{abl_name}"
            combined = {**GRAPH_OVERRIDES, **regime_ov, **abl_ov}
            conditions[cond_name] = combined
    seeds = list(range(0, n_seeds))
    return conditions, seeds


def main():
    """Run Phase 2 surrogate expansion sweep."""
    partial = "--partial" in sys.argv

    log(f"Digital Life v{minimal_life.version()}")

    if partial:
        conditions, seeds = build_partial_conditions()
        log(
            f"Phase 2 PARTIAL sweep: {len(conditions)} conditions × {len(seeds)} seeds "
            f"= {len(conditions) * len(seeds)} runs"
        )
    else:
        conditions = build_full_conditions()
        # Build seed list: train + validate for regimes A-C, test for D-E
        # For simplicity, run all seeds (0-69) for all regimes
        seeds = list(range(0, 70))
        log(
            f"Phase 2 FULL sweep: {len(conditions)} conditions × {len(seeds)} seeds "
            f"= {len(conditions) * len(seeds)} runs"
        )

    log("")
    out_dir = experiment_output_dir()

    run_condition_suite_parallel(
        "phase2_",
        conditions,
        STEPS,
        seeds,
        SAMPLE_EVERY,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
