"""Tests for failure mode analysis (Phase 1 — peer review revision).

Tests the z-score-based "first break" detection algorithm with synthetic
time-series having known break points.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers to create synthetic experiment data matching the RunSummary schema
# ---------------------------------------------------------------------------


def _make_sample(
    step: int,
    energy: float,
    boundary: float,
    internal_state_0: float,
    waste: float,
    alive_count: int = 50,
) -> dict:
    """Create a single sample dict matching Rust RunSummary schema."""
    return {
        "step": step,
        "energy_mean": energy,
        "boundary_mean": boundary,
        "internal_state_mean": [internal_state_0, 0.5, 0.5, 0.5],
        "waste_mean": waste,
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


def _make_run(samples: list[dict], final_alive: int = 50) -> dict:
    """Create a minimal RunSummary dict."""
    return {
        "schema_version": 1,
        "steps": samples[-1]["step"] if samples else 0,
        "sample_every": 20,
        "final_alive_count": final_alive,
        "samples": samples,
        "lifespans": [],
        "total_reproduction_events": 0,
        "lineage_events": [],
    }


def _make_stable_baseline(
    n_seeds: int = 20, n_steps: int = 100, sample_every: int = 20
) -> list[dict]:
    """Create n_seeds baseline runs with stable metric values."""
    runs = []
    for _seed in range(n_seeds):
        samples = []
        for step in range(sample_every, n_steps + 1, sample_every):
            samples.append(
                _make_sample(
                    step=step,
                    energy=0.5,
                    boundary=0.8,
                    internal_state_0=0.5,
                    waste=0.1,
                )
            )
        runs.append(_make_run(samples))
    return runs


def _make_drop_condition(
    n_seeds: int = 20,
    n_steps: int = 100,
    sample_every: int = 20,
    break_metric: str = "energy_mean",
    break_step: int = 60,
    break_value: float = 0.0,
) -> list[dict]:
    """Create runs where one metric drops at break_step."""
    metric_defaults = {
        "energy_mean": 0.5,
        "boundary_mean": 0.8,
        "internal_state_0": 0.5,
        "waste_mean": 0.1,
    }
    runs = []
    for _seed in range(n_seeds):
        samples = []
        for step in range(sample_every, n_steps + 1, sample_every):
            vals = dict(metric_defaults)
            if step >= break_step:
                if break_metric == "internal_state_0":
                    vals["internal_state_0"] = break_value
                else:
                    vals[break_metric] = break_value
            samples.append(
                _make_sample(
                    step=step,
                    energy=vals["energy_mean"],
                    boundary=vals["boundary_mean"],
                    internal_state_0=vals["internal_state_0"],
                    waste=vals["waste_mean"],
                )
            )
        runs.append(_make_run(samples))
    return runs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeZScores:
    """Test z-score computation against baseline."""

    def test_stable_series_has_zero_zscore(self):
        from analyze_failure_modes import compute_z_scores

        baseline_series = [0.5] * 10
        test_series = [0.5] * 10
        z_scores = compute_z_scores(baseline_series, test_series)
        for z in z_scores:
            assert abs(z) < 1e-6, f"Expected ~0, got {z}"

    def test_dropped_series_has_negative_zscore(self):
        from analyze_failure_modes import compute_z_scores

        baseline_series = [0.5] * 10
        test_series = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        z_scores = compute_z_scores(baseline_series, test_series)
        # First 3 should be near 0, rest should be strongly negative
        for z in z_scores[:3]:
            assert abs(z) < 1e-6
        for z in z_scores[3:]:
            assert z < -2.0

    def test_constant_baseline_zero_std_handled(self):
        """When baseline std is 0, z-score should still be computable."""
        from analyze_failure_modes import compute_z_scores

        baseline_series = [0.5] * 10
        test_series = [0.5] * 5 + [0.3] * 5
        z_scores = compute_z_scores(baseline_series, test_series)
        # Should not produce inf or nan
        for z in z_scores:
            assert math.isfinite(z)


class TestFirstBreakDetection:
    """Test the sustained-break detection algorithm."""

    def test_detects_known_break_point(self):
        from analyze_failure_modes import detect_first_break

        # z-scores: first 3 are 0, then -3 for the rest (sustained)
        z_scores = [0.0, 0.0, 0.0, -3.0, -3.0, -3.0, -3.0, -3.0]
        steps = [20, 40, 60, 80, 100, 120, 140, 160]
        result = detect_first_break(z_scores, steps, threshold=-2.0, sustained=3)
        assert result == 80  # First step where z < -2 sustained for 3 samples

    def test_no_break_returns_none(self):
        from analyze_failure_modes import detect_first_break

        z_scores = [0.0, -0.5, 0.0, -1.0, 0.0]
        steps = [20, 40, 60, 80, 100]
        result = detect_first_break(z_scores, steps, threshold=-2.0, sustained=3)
        assert result is None

    def test_transient_dip_not_detected(self):
        """A dip of only 2 consecutive samples should not trigger sustained=3."""
        from analyze_failure_modes import detect_first_break

        z_scores = [0.0, -3.0, -3.0, 0.0, 0.0]
        steps = [20, 40, 60, 80, 100]
        result = detect_first_break(z_scores, steps, threshold=-2.0, sustained=3)
        assert result is None

    def test_break_at_end_of_series(self):
        from analyze_failure_modes import detect_first_break

        z_scores = [0.0, 0.0, -3.0, -3.0, -3.0]
        steps = [20, 40, 60, 80, 100]
        result = detect_first_break(z_scores, steps, threshold=-2.0, sustained=3)
        assert result == 60


class TestSimultaneousBreaks:
    """Test handling of simultaneous breaks across metrics."""

    def test_simultaneous_breaks_both_reported(self):
        from analyze_failure_modes import detect_cascade_order

        # Both energy and boundary break at step 80
        break_points = {
            "energy_mean": 80,
            "boundary_mean": 80,
            "internal_state_mean_0": None,
            "waste_mean": None,
        }
        order = detect_cascade_order(break_points)
        # Both should appear first (tied)
        first_step = order[0][1]
        assert first_step == 80
        first_metrics = [m for m, s in order if s == first_step]
        assert "energy_mean" in first_metrics
        assert "boundary_mean" in first_metrics

    def test_cascade_order_sorted(self):
        from analyze_failure_modes import detect_cascade_order

        break_points = {
            "energy_mean": 100,
            "boundary_mean": 60,
            "internal_state_mean_0": 80,
            "waste_mean": None,
        }
        order = detect_cascade_order(break_points)
        # Should be sorted by step: boundary(60) < internal(80) < energy(100)
        assert order == [
            ("boundary_mean", 60),
            ("internal_state_mean_0", 80),
            ("energy_mean", 100),
        ]


class TestMajorityVote:
    """Test majority-vote cascade ordering across seeds."""

    def test_majority_selects_most_common_first_metric(self):
        from analyze_failure_modes import majority_vote_cascade

        # 3 seeds: energy breaks first in 2, boundary in 1
        per_seed_orders = [
            [("energy_mean", 60), ("boundary_mean", 80)],
            [("energy_mean", 40), ("boundary_mean", 100)],
            [("boundary_mean", 50), ("energy_mean", 70)],
        ]
        result = majority_vote_cascade(per_seed_orders)
        assert result[0] == "energy_mean"  # majority says energy first


class TestAllZeroBaseline:
    """Edge case: all-zero baseline should not crash."""

    def test_zero_baseline_no_crash(self):
        from analyze_failure_modes import compute_z_scores

        baseline_series = [0.0] * 10
        test_series = [0.0] * 10
        z_scores = compute_z_scores(baseline_series, test_series)
        for z in z_scores:
            assert math.isfinite(z)


class TestEndToEndAnalysis:
    """Integration test using synthetic data files."""

    def test_full_pipeline(self, tmp_path: Path):
        from analyze_failure_modes import analyze_failure_modes

        # Create synthetic data
        baseline = _make_stable_baseline(n_seeds=5, n_steps=200, sample_every=20)
        drop_metab = _make_drop_condition(
            n_seeds=5,
            n_steps=200,
            sample_every=20,
            break_metric="energy_mean",
            break_step=80,
        )

        data_dir = tmp_path / "ablation_sweep"
        data_dir.mkdir()
        for i, run in enumerate(baseline):
            (data_dir / f"full_seed{i}.json").write_text(json.dumps(run))
        for i, run in enumerate(drop_metab):
            (data_dir / f"drop_metabolism_seed{i}.json").write_text(json.dumps(run))

        result = analyze_failure_modes(
            data_dir=data_dir,
            conditions=["metabolism"],
            n_seeds=5,
        )

        assert "metabolism" in result
        metab = result["metabolism"]
        assert "cascade_order" in metab
        assert "per_seed" in metab
        # Energy should break first for metabolism ablation
        if metab["cascade_order"]:
            assert metab["cascade_order"][0] == "energy_mean"
