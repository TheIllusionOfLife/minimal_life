"""Crossover and evolution evidence experiment.

5 conditions to distinguish adaptation from neutral drift:
1. normal_crossover — evolution + segment-wise crossover
2. normal_no_crossover — evolution, no crossover (baseline)
3. neutral_drift — random parent selection (neutral-drift control)
4. no_evolution — hard ablation (copy genome without mutation)
5. uniform_crossover — evolution + uniform crossover (sensitivity check)

Long runs (10,000 steps) with seeds 100-129.

Usage:
    uv run python scripts/experiment_crossover.py > experiments/crossover_data.tsv
"""

import minimal_life
from experiment_common import log, run_condition_suite

STEPS = 10_000
SAMPLE_EVERY = 100
SEEDS = list(range(100, 130))

GRAPH_OVERRIDES = {"metabolism_mode": "graph"}

CONDITIONS = {
    "normal_crossover": {
        "enable_crossover": True,
        "crossover_mode": "segment_wise",
    },
    "normal_no_crossover": {
        "enable_crossover": False,
    },
    "neutral_drift": {
        "random_parent_selection": True,
        "enable_crossover": False,
    },
    "no_evolution": {
        "enable_evolution": False,
    },
    "uniform_crossover": {
        "enable_crossover": True,
        "crossover_mode": "uniform",
    },
}


def main():
    """Run crossover/evolution experiment: 5 conditions x 30 seeds x 10k steps."""
    log(f"Digital Life v{minimal_life.version()}")
    log(
        f"Crossover experiment: {STEPS} steps, sample every {SAMPLE_EVERY}, "
        f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})"
    )
    log("")
    run_condition_suite(
        "crossover_",
        CONDITIONS,
        STEPS,
        SEEDS,
        SAMPLE_EVERY,
        extra_overrides=GRAPH_OVERRIDES,
    )


if __name__ == "__main__":
    main()
