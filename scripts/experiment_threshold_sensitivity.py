"""Run threshold sensitivity experiments (Phase 4 — peer review revision).

Varies death_boundary_threshold and death_energy_threshold across
4 ablation conditions to test rank stability (Reviewer A Q3).

4 boundary × 3 energy × 4 conditions × 15 seeds = 720 runs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_common import (
    CRITERION_TO_FLAG,
    experiment_output_dir,
    log,
    run_single,
)

BOUNDARY_THRESHOLDS = [0.05, 0.1, 0.15, 0.2]
ENERGY_THRESHOLDS = [0.0, 0.05, 0.1]
CONDITIONS = {
    "normal": {},
    "no_metabolism": {CRITERION_TO_FLAG["metabolism"]: False},
    "no_reproduction": {CRITERION_TO_FLAG["reproduction"]: False},
    "no_response": {CRITERION_TO_FLAG["response"]: False},
}
SEEDS = list(range(100, 115))
STEPS = 2000
SAMPLE_EVERY = 50


def run_threshold_sensitivity() -> None:
    out_dir = experiment_output_dir() / "threshold_sweep"
    out_dir.mkdir(exist_ok=True)

    total_runs = len(BOUNDARY_THRESHOLDS) * len(ENERGY_THRESHOLDS) * len(CONDITIONS) * len(SEEDS)
    log(f"Total runs: {total_runs}")
    total_start = time.perf_counter()
    completed = 0

    for bt in BOUNDARY_THRESHOLDS:
        for et in ENERGY_THRESHOLDS:
            combo = f"bt{bt}_et{et}"
            log(f"--- Threshold combo: {combo} ---")

            for cond_name, overrides in CONDITIONS.items():
                cond_overrides = {
                    **overrides,
                    "death_boundary_threshold": bt,
                    "death_energy_threshold": et,
                }

                for seed in SEEDS:
                    t0 = time.perf_counter()
                    result = run_single(
                        seed,
                        cond_overrides,
                        steps=STEPS,
                        sample_every=SAMPLE_EVERY,
                    )
                    elapsed = time.perf_counter() - t0
                    completed += 1

                    fname = f"thresh_{combo}_{cond_name}_seed{seed}.json"
                    with open(out_dir / fname, "w") as f:
                        json.dump(result, f, indent=2)

                    alive = result.get("final_alive_count", 0)
                    log(
                        f"  [{completed}/{total_runs}]"
                        f" {combo} {cond_name} seed={seed}"
                        f" alive={alive:4d} {elapsed:.2f}s"
                    )

    log(f"Total time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    run_threshold_sensitivity()
