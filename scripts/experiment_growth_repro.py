"""Growth/reproduction separability experiment.

Tests whether growth and reproduction are independently functional
or if growth's effect is primarily through its gating of reproduction.

4 conditions:
1. normal — growth + reproduction both enabled
2. no_growth — growth disabled, reproduction enabled
3. bypass_maturity — growth enabled but maturity gate bypassed
4. no_growth_bypass — growth disabled + maturity bypassed

Seeds 100-129, 2000 steps.

Usage:
    uv run python scripts/experiment_growth_repro.py > experiments/growth_repro_data.tsv
"""

import minimal_life
from experiment_common import log, run_condition_suite

STEPS = 2000
SAMPLE_EVERY = 50
SEEDS = list(range(100, 130))

GRAPH_OVERRIDES = {"metabolism_mode": "graph"}

CONDITIONS = {
    "normal": {},
    "no_growth": {"enable_growth": False},
    "bypass_maturity": {"reproduction_bypass_maturity": True},
    "no_growth_bypass": {
        "enable_growth": False,
        "reproduction_bypass_maturity": True,
    },
}


def main():
    """Run growth/repro separability: 4 conditions x 30 seeds."""
    log(f"Digital Life v{minimal_life.version()}")
    log(
        f"Growth/repro experiment: {STEPS} steps, sample every {SAMPLE_EVERY}, "
        f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})"
    )
    log("")
    run_condition_suite(
        "growth_repro_",
        CONDITIONS,
        STEPS,
        SEEDS,
        SAMPLE_EVERY,
        extra_overrides=GRAPH_OVERRIDES,
    )


if __name__ == "__main__":
    main()
