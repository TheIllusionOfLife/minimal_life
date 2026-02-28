"""Run boundary topology experiments (Phase 5 — peer review revision).

Compares toroidal vs bounded world topologies across 4 ablation conditions
to test generality of findings (Reviewer C Q4).

2 topologies × 4 conditions × 30 seeds = 240 runs.
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


def run_boundary_topology() -> None:
    out_dir = experiment_output_dir() / "boundary_sweep"
    out_dir.mkdir(exist_ok=True)

    total_runs = len(TOPOLOGIES) * len(CONDITIONS) * len(SEEDS)
    log(f"Total runs: {total_runs}")
    total_start = time.perf_counter()
    completed = 0

    for topo_name, topo_overrides in TOPOLOGIES.items():
        log(f"=== Topology: {topo_name} ===")

        for cond_name, cond_overrides in CONDITIONS.items():
            combined = {**topo_overrides, **cond_overrides}
            log(f"--- Condition: {cond_name} ---")

            for seed in SEEDS:
                t0 = time.perf_counter()
                result = run_single(
                    seed,
                    combined,
                    steps=STEPS,
                    sample_every=SAMPLE_EVERY,
                )
                elapsed = time.perf_counter() - t0
                completed += 1

                fname = f"boundary_{topo_name}_{cond_name}_seed{seed}.json"
                with open(out_dir / fname, "w") as f:
                    json.dump(result, f, indent=2)

                alive = result.get("final_alive_count", 0)
                log(
                    f"  [{completed}/{total_runs}]"
                    f" {topo_name} {cond_name} seed={seed}"
                    f" alive={alive:4d} {elapsed:.2f}s"
                )

    log(f"Total time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    run_boundary_topology()
