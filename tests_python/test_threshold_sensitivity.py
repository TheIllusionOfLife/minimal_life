"""Tests for threshold sensitivity analysis (Phase 4 — peer review revision).

Tests Spearman rank correlation computation and the analysis pipeline
for verifying ablation effect hierarchy stability across threshold combos.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_sample(step: int, alive_count: int = 50) -> dict:
    return {
        "step": step,
        "energy_mean": 0.5,
        "waste_mean": 0.1,
        "boundary_mean": 0.8,
        "internal_state_mean": [0.5, 0.5, 0.5, 0.5],
        "alive_count": alive_count,
        "resource_total": 100.0,
        "birth_count": 0,
        "death_count": 0,
        "population_size": alive_count,
        "mean_generation": 0.0,
        "mean_genome_drift": 0.0,
        "agent_id_exhaustion_events": 0,
        "energy_std": 0.01,
        "waste_std": 0.01,
        "boundary_std": 0.01,
        "internal_state_std": [0.01, 0.01, 0.01, 0.01],
        "mean_age": 100.0,
        "genome_diversity": 0.0,
        "max_generation": 0,
        "maturity_mean": 1.0,
        "spatial_cohesion_mean": 5.0,
    }


def _make_run(final_alive: int = 50) -> dict:
    samples = [_make_sample(step=s, alive_count=final_alive) for s in range(20, 2001, 20)]
    return {
        "schema_version": 1,
        "steps": 2000,
        "sample_every": 20,
        "final_alive_count": final_alive,
        "samples": samples,
        "lifespans": [],
        "total_reproduction_events": 0,
        "lineage_events": [],
    }


# ---------------------------------------------------------------------------
# Tests: Spearman rank correlation
# ---------------------------------------------------------------------------


class TestSpearmanRank:
    """Test Spearman rank correlation computation."""

    def test_perfect_correlation(self):
        from analyze_threshold_sensitivity import spearman_rank_correlation

        x = [1.0, 2.0, 3.0, 4.0]
        y = [10.0, 20.0, 30.0, 40.0]
        rho = spearman_rank_correlation(x, y)
        assert abs(rho - 1.0) < 1e-6

    def test_perfect_anticorrelation(self):
        from analyze_threshold_sensitivity import spearman_rank_correlation

        x = [1.0, 2.0, 3.0, 4.0]
        y = [40.0, 30.0, 20.0, 10.0]
        rho = spearman_rank_correlation(x, y)
        assert abs(rho - (-1.0)) < 1e-6

    def test_no_correlation(self):
        from analyze_threshold_sensitivity import spearman_rank_correlation

        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 1.0, 5.0, 2.0, 4.0]
        rho = spearman_rank_correlation(x, y)
        assert abs(rho) < 0.5

    def test_too_few_values(self):
        from analyze_threshold_sensitivity import spearman_rank_correlation

        rho = spearman_rank_correlation([1.0], [2.0])
        assert math.isnan(rho)


# ---------------------------------------------------------------------------
# Tests: Delta computation
# ---------------------------------------------------------------------------


class TestDeltaComputation:
    """Test Δ% computation for ablation vs baseline."""

    def test_delta_percent(self):
        from analyze_threshold_sensitivity import compute_delta_percent

        baseline_alive = [50, 48, 52, 49, 51]
        ablation_alive = [25, 24, 26, 25, 24]
        delta = compute_delta_percent(baseline_alive, ablation_alive)
        # Mean baseline = 50, mean ablation = 24.8, delta = (24.8-50)/50*100 = -50.4%
        assert delta < -40.0

    def test_zero_baseline(self):
        from analyze_threshold_sensitivity import compute_delta_percent

        delta = compute_delta_percent([0, 0, 0], [0, 0, 0])
        assert math.isnan(delta)


# ---------------------------------------------------------------------------
# Tests: Rank stability
# ---------------------------------------------------------------------------


class TestRankStability:
    """Test rank stability assessment across threshold combos."""

    def test_stable_ranking(self):
        from analyze_threshold_sensitivity import assess_rank_stability

        # All threshold combos produce the same ranking
        deltas_by_combo = {
            "bt0.05_et0.0": {"no_metabolism": -80, "no_reproduction": -60, "no_response": -40},
            "bt0.1_et0.0": {"no_metabolism": -75, "no_reproduction": -55, "no_response": -35},
            "bt0.15_et0.0": {"no_metabolism": -70, "no_reproduction": -50, "no_response": -30},
        }
        result = assess_rank_stability(deltas_by_combo, reference_key="bt0.1_et0.0")
        # All should have rho ≈ 1.0 since ordering is consistent
        for rho in result["spearman_correlations"].values():
            assert rho > 0.9

    def test_unstable_ranking(self):
        from analyze_threshold_sensitivity import assess_rank_stability

        deltas_by_combo = {
            "ref": {"a": -80, "b": -60, "c": -40},
            "alt": {"a": -40, "b": -80, "c": -60},  # different ordering
        }
        result = assess_rank_stability(deltas_by_combo, reference_key="ref")
        # The alt combo has a different ranking, rho should be < 1
        assert result["spearman_correlations"]["alt"] < 0.9


# ---------------------------------------------------------------------------
# Tests: End-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration test with synthetic threshold sweep data."""

    def test_full_pipeline(self, tmp_path: Path):
        from analyze_threshold_sensitivity import analyze_threshold_sensitivity

        data_dir = tmp_path / "threshold_sweep"
        data_dir.mkdir()

        # Create synthetic results for 2 threshold combos × 2 conditions
        for bt in ["0.05", "0.1"]:
            for et in ["0.0", "0.05"]:
                for cond, alive in [("normal", 50), ("no_metabolism", 10)]:
                    for seed in range(3):
                        fname = f"thresh_bt{bt}_et{et}_{cond}_seed{seed}.json"
                        run = _make_run(final_alive=alive)
                        (data_dir / fname).write_text(json.dumps(run))

        result = analyze_threshold_sensitivity(
            data_dir=data_dir,
            boundary_thresholds=[0.05, 0.1],
            energy_thresholds=[0.0, 0.05],
            conditions=["normal", "no_metabolism"],
            n_seeds=3,
        )
        assert "deltas_by_combo" in result
        assert "rank_stability" in result
