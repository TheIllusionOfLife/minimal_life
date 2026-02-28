"""Run fitness landscape experiments (Phase 2 — peer review revision).

Seeds 100–129 (n=30), snapshots every 1,000 steps over 10,000 steps.
2 conditions: normal (evolution on) vs no_evolution (clonal control).
Uses graph metabolism mode.

Parallelised with ProcessPoolExecutor (max_workers=6).
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import minimal_life
from experiment_common import (
    experiment_output_dir,
    log,
    make_config,
)

SEEDS = list(range(100, 130))
STEPS = 10_000
SAMPLE_EVERY = 100
SNAPSHOT_STEPS_JSON = json.dumps(list(range(1000, STEPS + 1, 1000)))
MAX_WORKERS = 6

CONDITIONS = {
    "normal": {"metabolism_mode": "graph"},
    "no_evolution": {"metabolism_mode": "graph", "enable_evolution": False},
}


def _run_one(cond_name: str, overrides: dict, seed: int, out_dir: str) -> dict:
    """Run a single fitness landscape experiment (designed for subprocess)."""
    t0 = time.perf_counter()
    config_json = make_config(seed, overrides)
    result_json = minimal_life.run_niche_experiment_json(
        config_json, STEPS, SAMPLE_EVERY, SNAPSHOT_STEPS_JSON
    )
    result = json.loads(result_json)
    elapsed = time.perf_counter() - t0

    fname = f"fitness_{cond_name}_seed{seed}.json"
    with open(Path(out_dir) / fname, "w") as f:
        json.dump(result, f, indent=2)

    return {
        "cond": cond_name,
        "seed": seed,
        "alive": result.get("final_alive_count", 0),
        "snaps": len(result.get("organism_snapshots", [])),
        "elapsed": elapsed,
    }


def run_fitness_landscape() -> None:
    out_dir = experiment_output_dir()
    tasks = [
        (cond_name, overrides, seed)
        for cond_name, overrides in CONDITIONS.items()
        for seed in SEEDS
    ]
    total = len(tasks)
    log(f"Fitness landscape: {total} runs with {MAX_WORKERS} workers")
    total_start = time.perf_counter()
    completed = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_run_one, cond, ov, seed, str(out_dir)): (cond, seed)
            for cond, ov, seed in tasks
        }
        for future in as_completed(futures):
            info = future.result()
            completed += 1
            log(
                f"  [{completed}/{total}] {info['cond']} seed={info['seed']:3d}"
                f"  alive={info['alive']:4d}  snaps={info['snaps']}"
                f"  {info['elapsed']:.2f}s"
            )

    log(f"Total time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    run_fitness_landscape()
