"""Cause-of-death analysis experiment.

Runs standard 8-condition ablation sweep with death cause tracking.
Seeds 100-129, 2000 steps, sample_every=50.

Usage:
    uv run python scripts/experiment_death_causes.py > experiments/death_causes_data.tsv
"""

import minimal_life
from experiment_common import CONDITIONS, log, run_condition_suite

STEPS = 2000
SAMPLE_EVERY = 50
SEEDS = list(range(100, 130))

GRAPH_OVERRIDES = {"metabolism_mode": "graph"}


def main():
    """Run death cause analysis: 8 conditions x 30 seeds."""
    log(f"Digital Life v{minimal_life.version()}")
    log(
        f"Death cause experiment: {STEPS} steps, sample every {SAMPLE_EVERY}, "
        f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})"
    )
    log("")
    run_condition_suite(
        "death_causes_",
        CONDITIONS,
        STEPS,
        SEEDS,
        SAMPLE_EVERY,
        extra_overrides=GRAPH_OVERRIDES,
    )


if __name__ == "__main__":
    main()
