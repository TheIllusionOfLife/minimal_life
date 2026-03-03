"""NN hidden layer size sweep.

Tests whether ablation results are sensitive to NN capacity.
Hidden sizes {8, 16, 32}, normal condition only.
Seeds 100-129, 2000 steps.

Usage:
    uv run python scripts/experiment_nn_sweep.py > experiments/nn_sweep_data.tsv
"""

import minimal_life
from experiment_common import log, run_condition_suite

STEPS = 2000
SAMPLE_EVERY = 50
SEEDS = list(range(100, 130))

GRAPH_OVERRIDES = {"metabolism_mode": "graph"}

CONDITIONS = {
    "hidden_8": {"nn_hidden_size": 8},
    "hidden_16": {"nn_hidden_size": 16},
    "hidden_32": {"nn_hidden_size": 32},
}


def main():
    """Run NN hidden size sweep: 3 conditions x 30 seeds."""
    log(f"Digital Life v{minimal_life.version()}")
    log(
        f"NN sweep: {STEPS} steps, sample every {SAMPLE_EVERY}, "
        f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})"
    )
    log("")
    run_condition_suite(
        "nn_sweep_",
        CONDITIONS,
        STEPS,
        SEEDS,
        SAMPLE_EVERY,
        extra_overrides=GRAPH_OVERRIDES,
    )


if __name__ == "__main__":
    main()
