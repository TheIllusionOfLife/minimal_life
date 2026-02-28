"""Run threshold sensitivity experiments (Phase 4 — peer review revision).

Varies death_boundary_threshold and death_energy_threshold across
4 ablation conditions to test rank stability (Reviewer A Q3).

4 boundary × 3 energy × 4 conditions × 15 seeds = 720 runs.

Parallelised with ProcessPoolExecutor (max_workers=6).
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
MAX_WORKERS = 6


def _run_one(
    bt: float, et: float, cond_name: str, overrides: dict, seed: int, out_dir: str
) -> dict:
    """Run a single threshold sensitivity experiment (designed for subprocess)."""
    cond_overrides = {
        **overrides,
        "death_boundary_threshold": bt,
        "death_energy_threshold": et,
    }
    t0 = time.perf_counter()
    result = run_single(seed, cond_overrides, steps=STEPS, sample_every=SAMPLE_EVERY)
    elapsed = time.perf_counter() - t0

    combo = f"bt{bt}_et{et}"
    fname = f"thresh_{combo}_{cond_name}_seed{seed}.json"
    with open(Path(out_dir) / fname, "w") as f:
        json.dump(result, f, indent=2)

    return {
        "combo": combo,
        "cond": cond_name,
        "seed": seed,
        "alive": result.get("final_alive_count", 0),
        "elapsed": elapsed,
    }


def run_threshold_sensitivity() -> None:
    out_dir = experiment_output_dir() / "threshold_sweep"
    out_dir.mkdir(exist_ok=True)

    tasks = [
        (bt, et, cond_name, overrides, seed)
        for bt in BOUNDARY_THRESHOLDS
        for et in ENERGY_THRESHOLDS
        for cond_name, overrides in CONDITIONS.items()
        for seed in SEEDS
    ]
    total = len(tasks)
    log(f"Threshold sensitivity: {total} runs with {MAX_WORKERS} workers")
    total_start = time.perf_counter()
    completed = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_run_one, bt, et, cond, ov, seed, str(out_dir)): (cond, seed)
            for bt, et, cond, ov, seed in tasks
        }
        for future in as_completed(futures):
            info = future.result()
            completed += 1
            log(
                f"  [{completed}/{total}] {info['combo']} {info['cond']}"
                f" seed={info['seed']} alive={info['alive']:4d}"
                f" {info['elapsed']:.2f}s"
            )

    log(f"Total time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    run_threshold_sensitivity()
