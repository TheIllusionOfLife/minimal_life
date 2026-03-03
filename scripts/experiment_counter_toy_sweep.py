"""Counter and Toy metabolism full ablation sweep.

Runs the 8-condition ablation for Counter and Toy metabolism engines
to compare ablation effect patterns across engine types.
Seeds 100-129, 2000 steps.

Usage:
    uv run python scripts/experiment_counter_toy_sweep.py > experiments/counter_toy_data.tsv
"""

import minimal_life
from experiment_common import CONDITIONS, log, run_condition_suite

STEPS = 2000
SAMPLE_EVERY = 50
SEEDS = list(range(100, 130))


def main():
    """Run ablation sweep for Counter and Toy engines."""
    log(f"Digital Life v{minimal_life.version()}")
    log(
        f"Counter/Toy sweep: {STEPS} steps, sample every {SAMPLE_EVERY}, "
        f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})"
    )

    log("\n=== Counter Engine ===")
    run_condition_suite(
        "counter_",
        CONDITIONS,
        STEPS,
        SEEDS,
        SAMPLE_EVERY,
        extra_overrides={"metabolism_mode": "counter"},
    )

    log("\n=== Toy Engine ===")
    run_condition_suite(
        "toy_",
        CONDITIONS,
        STEPS,
        SEEDS,
        SAMPLE_EVERY,
        extra_overrides={"metabolism_mode": "toy"},
    )


if __name__ == "__main__":
    main()
