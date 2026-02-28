"""Run fitness landscape experiments (Phase 2 — peer review revision).

Seeds 100–129 (n=30), snapshots every 1,000 steps over 10,000 steps.
2 conditions: normal (evolution on) vs no_evolution (clonal control).
Uses graph metabolism mode.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import minimal_life
from experiment_common import (
    experiment_output_dir,
    log,
    make_config,
)


def run_fitness_landscape() -> None:
    out_dir = experiment_output_dir()
    seeds = list(range(100, 130))
    steps = 10_000
    sample_every = 100
    snapshot_steps = list(range(1000, steps + 1, 1000))

    conditions = {
        "normal": {"metabolism_mode": "graph"},
        "no_evolution": {"metabolism_mode": "graph", "enable_evolution": False},
    }

    total_start = time.perf_counter()
    for cond_name, overrides in conditions.items():
        log(f"--- Condition: {cond_name} ---")
        cond_start = time.perf_counter()

        for seed in seeds:
            t0 = time.perf_counter()
            config_json = make_config(seed, overrides)
            result_json = minimal_life.run_niche_experiment_json(
                config_json,
                steps,
                sample_every,
                snapshot_steps,
            )
            result = json.loads(result_json)
            elapsed = time.perf_counter() - t0

            fname = f"fitness_{cond_name}_seed{seed}.json"
            with open(out_dir / fname, "w") as f:
                json.dump(result, f, indent=2)

            alive = result.get("final_alive_count", 0)
            n_snaps = len(result.get("organism_snapshots", []))
            log(f"  seed={seed:3d}  alive={alive:4d}  snaps={n_snaps}  {elapsed:.2f}s")

        log(f"  Condition time: {time.perf_counter() - cond_start:.1f}s\n")

    log(f"Total time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    run_fitness_landscape()
