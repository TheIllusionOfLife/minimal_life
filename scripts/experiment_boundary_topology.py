"""Run boundary topology experiments (Phase 5 — peer review revision).

Compares toroidal vs bounded world topologies across 4 ablation conditions
to test generality of findings (Reviewer C Q4).

2 topologies × 4 conditions × 30 seeds = 240 runs.

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

TOPOLOGIES = {
    "toroidal": {"world_topology": "toroidal"},
    "bounded": {"world_topology": "bounded"},
}
CONDITIONS = {
    "normal": {},
    "no_metabolism": {CRITERION_TO_FLAG["metabolism"]: False},
    "no_reproduction": {CRITERION_TO_FLAG["reproduction"]: False},
    "no_response": {CRITERION_TO_FLAG["response"]: False},
}
SEEDS = list(range(100, 130))
STEPS = 2000
SAMPLE_EVERY = 50
MAX_WORKERS = 6


def _run_one(
    topo_name: str,
    topo_overrides: dict,
    cond_name: str,
    cond_overrides: dict,
    seed: int,
    out_dir: str,
) -> dict:
    """Run a single boundary topology experiment (designed for subprocess)."""
    combined = {**topo_overrides, **cond_overrides}
    t0 = time.perf_counter()
    result = run_single(seed, combined, steps=STEPS, sample_every=SAMPLE_EVERY)
    elapsed = time.perf_counter() - t0

    fname = f"boundary_{topo_name}_{cond_name}_seed{seed}.json"
    with open(Path(out_dir) / fname, "w") as f:
        json.dump(result, f, indent=2)

    return {
        "topo": topo_name,
        "cond": cond_name,
        "seed": seed,
        "alive": result.get("final_alive_count", 0),
        "elapsed": elapsed,
    }


def run_boundary_topology() -> None:
    out_dir = experiment_output_dir() / "boundary_sweep"
    out_dir.mkdir(exist_ok=True)

    tasks = [
        (topo_name, topo_ov, cond_name, cond_ov, seed)
        for topo_name, topo_ov in TOPOLOGIES.items()
        for cond_name, cond_ov in CONDITIONS.items()
        for seed in SEEDS
    ]
    total = len(tasks)
    log(f"Boundary topology: {total} runs with {MAX_WORKERS} workers")
    total_start = time.perf_counter()
    completed = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_run_one, tn, to, cn, co, seed, str(out_dir)): (tn, cn, seed)
            for tn, to, cn, co, seed in tasks
        }
        for future in as_completed(futures):
            info = future.result()
            completed += 1
            log(
                f"  [{completed}/{total}] {info['topo']} {info['cond']}"
                f" seed={info['seed']} alive={info['alive']:4d}"
                f" {info['elapsed']:.2f}s"
            )

    log(f"Total time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    run_boundary_topology()
