"""Tests for the Phase 1 criterion-reduction pipeline.

All tests are hermetic — they use synthetic data and do not read from disk.
The scripts being tested expose pure functions that can be imported independently
of the minimal_life Rust extension.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

# scripts/ is on sys.path via pyproject.toml: pythonpath = [".", "python", "scripts"]
from analyze_criterion_reduction import stability_selection
from analyze_surrogates import FEATURE_NAMES, extract_run_features
from experiment_ablation_sweep import (
    CRITERIA,
    ablation_conditions,
    build_criterion_presence_matrix,
    compute_performance_metrics,
)


class TestAblationConditions:
    """Verify the condition generator produces the correct 30 conditions."""

    def _all_cond_dict(self) -> dict[str, list[str]]:
        return dict(ablation_conditions())

    def test_total_count(self):
        conds = list(ablation_conditions())
        assert len(conds) == 30  # 1 all_off + 1 full + 7 single + 21 pairwise

    def test_all_off_disables_all_criteria(self):
        conds = self._all_cond_dict()
        assert set(conds["all_off"]) == set(CRITERIA)

    def test_full_disables_nothing(self):
        conds = self._all_cond_dict()
        assert conds["full"] == []

    def test_single_ablations_count(self):
        conds = self._all_cond_dict()
        # Single ablations: "drop_{c}" with exactly one underscore
        singles = [k for k in conds if k.startswith("drop_") and k.count("_") == 1]
        assert len(singles) == 7

    def test_pairwise_ablations_count(self):
        conds = self._all_cond_dict()
        # Pairwise: "drop_{c1}_{c2}" — criteria are single words, so 2 underscores
        pairs = [k for k in conds if k.startswith("drop_") and k.count("_") == 2]
        assert len(pairs) == 21

    def test_condition_names_unique(self):
        names = [n for n, _ in ablation_conditions()]
        assert len(names) == len(set(names))

    def test_single_ablation_disables_one(self):
        conds = self._all_cond_dict()
        for c in CRITERIA:
            assert conds[f"drop_{c}"] == [c]

    def test_pairwise_ablation_disables_two(self):
        conds = self._all_cond_dict()
        for c1, c2 in combinations(CRITERIA, 2):
            key = f"drop_{c1}_{c2}"
            assert key in conds
            assert set(conds[key]) == {c1, c2}

    def test_criteria_list_length(self):
        assert len(CRITERIA) == 7


# ── compute_performance_metrics ──────────────────────────────────────────────


def _make_sample(
    step: int,
    alive: int = 10,
    energy: float = 1.0,
    waste: float = 0.5,
    boundary: float = 0.8,
    births: int = 0,
    deaths: int = 0,
    genome_div: float = 0.5,
    spatial: float = 0.1,
    maturity: float = 0.7,
    internal_std: list[float] | None = None,
    resource: float = 100.0,
) -> dict:
    return {
        "step": step,
        "alive_count": alive,
        "energy_mean": energy,
        "waste_mean": waste,
        "boundary_mean": boundary,
        "birth_count": births,
        "death_count": deaths,
        "genome_diversity": genome_div,
        "spatial_cohesion_mean": spatial,
        "maturity_mean": maturity,
        "internal_state_std": internal_std if internal_std is not None else [0.1, 0.1, 0.1, 0.1],
        "resource_total": resource,
    }


class TestComputePerformanceMetrics:
    def test_empty_samples_returns_zero_dict(self):
        result = compute_performance_metrics([], 1000)
        assert isinstance(result, dict)
        assert result["alive_auc"] == pytest.approx(0.0)

    def test_all_metric_keys_present(self):
        samples = [_make_sample(0), _make_sample(100)]
        result = compute_performance_metrics(samples, 100)
        expected_keys = {
            "alive_auc",
            "energy_stability",
            "boundary_integrity",
            "homeostasis_quality",
            "reproduction_rate",
            "genome_diversity_late",
            "spatial_cohesion",
        }
        assert set(result.keys()) == expected_keys

    def test_alive_auc_constant_population(self):
        # Constant 10 alive over 0→100 steps: AUC = 10*100 / 100 = 10.0
        samples = [_make_sample(0, alive=10), _make_sample(100, alive=10)]
        result = compute_performance_metrics(samples, 100)
        assert result["alive_auc"] == pytest.approx(10.0)

    def test_alive_auc_zero_population(self):
        samples = [_make_sample(0, alive=0), _make_sample(100, alive=0)]
        result = compute_performance_metrics(samples, 100)
        assert result["alive_auc"] == pytest.approx(0.0)

    def test_reproduction_rate_calculation(self):
        # 2 samples, births=[5, 5] → total=10, steps=100 → rate=0.1
        samples = [_make_sample(0, births=5), _make_sample(100, births=5)]
        result = compute_performance_metrics(samples, 100)
        assert result["reproduction_rate"] == pytest.approx(0.1)

    def test_homeostasis_quality_inversely_proportional_to_std(self):
        # Higher internal_state_std → lower homeostasis_quality
        high_std = [_make_sample(0, internal_std=[1.0, 1.0, 1.0, 1.0])]
        low_std = [_make_sample(0, internal_std=[0.01, 0.01, 0.01, 0.01])]
        q_high = compute_performance_metrics(high_std, 100)["homeostasis_quality"]
        q_low = compute_performance_metrics(low_std, 100)["homeostasis_quality"]
        assert q_high < q_low

    def test_energy_stability_is_late_phase_mean(self):
        # Build 60 samples; last 50 should dominate
        samples = [_make_sample(i * 20, energy=float(i)) for i in range(60)]
        result = compute_performance_metrics(samples, 60 * 20)
        # Last 50 samples: i=10..59 → energies 10..59, mean=34.5
        assert result["energy_stability"] == pytest.approx(34.5, abs=0.5)

    def test_values_are_finite(self):
        samples = [_make_sample(i * 20) for i in range(20)]
        result = compute_performance_metrics(samples, 20 * 20)
        for k, v in result.items():
            assert np.isfinite(v), f"Non-finite value for {k}: {v}"


# ── build_criterion_presence_matrix ──────────────────────────────────────────


class TestBuildCriterionPresenceMatrix:
    def test_full_row_all_ones(self):
        matrix = build_criterion_presence_matrix(["full"], CRITERIA)
        assert matrix.shape == (1, len(CRITERIA))
        assert np.all(matrix[0] == 1.0)

    def test_all_off_row_all_zeros(self):
        matrix = build_criterion_presence_matrix(["all_off"], CRITERIA)
        assert np.all(matrix[0] == 0.0)

    def test_single_drop_zeroes_correct_index(self):
        for c in CRITERIA:
            matrix = build_criterion_presence_matrix([f"drop_{c}"], CRITERIA)
            c_idx = CRITERIA.index(c)
            assert matrix[0, c_idx] == 0.0, f"Expected 0 at index {c_idx} for drop_{c}"
            for j, other in enumerate(CRITERIA):
                if other != c:
                    assert matrix[0, j] == 1.0, f"Expected 1 at index {j} for drop_{c}"

    def test_pairwise_drop_zeroes_two_indices(self):
        c1, c2 = CRITERIA[0], CRITERIA[1]  # metabolism, boundary
        matrix = build_criterion_presence_matrix([f"drop_{c1}_{c2}"], CRITERIA)
        i1, i2 = CRITERIA.index(c1), CRITERIA.index(c2)
        assert matrix[0, i1] == 0.0
        assert matrix[0, i2] == 0.0
        for j, c in enumerate(CRITERIA):
            if c not in (c1, c2):
                assert matrix[0, j] == 1.0

    def test_matrix_shape_matches_inputs(self):
        all_names = [n for n, _ in ablation_conditions()]
        matrix = build_criterion_presence_matrix(all_names, CRITERIA)
        assert matrix.shape == (30, 7)

    def test_full_row_is_ones_in_30_condition_matrix(self):
        all_names = [n for n, _ in ablation_conditions()]
        matrix = build_criterion_presence_matrix(all_names, CRITERIA)
        full_idx = all_names.index("full")
        assert np.all(matrix[full_idx] == 1.0)

    def test_all_off_row_is_zeros_in_30_condition_matrix(self):
        all_names = [n for n, _ in ablation_conditions()]
        matrix = build_criterion_presence_matrix(all_names, CRITERIA)
        all_off_idx = all_names.index("all_off")
        assert np.all(matrix[all_off_idx] == 0.0)

    def test_output_dtype_float(self):
        matrix = build_criterion_presence_matrix(["full"], CRITERIA)
        assert matrix.dtype == float or np.issubdtype(matrix.dtype, np.floating)

    def test_all_21_pairwise_conditions_parse_correctly(self):
        """Split-based parser must correctly zero exactly two columns per pair."""
        all_names = [n for n, _ in ablation_conditions()]
        matrix = build_criterion_presence_matrix(all_names, CRITERIA)
        for name, disabled in ablation_conditions():
            if not (name.startswith("drop_") and len(disabled) == 2):
                continue
            i = all_names.index(name)
            for j, c in enumerate(CRITERIA):
                if c in disabled:
                    assert matrix[i, j] == 0.0, f"{name}: expected 0 for {c}"
                else:
                    assert matrix[i, j] == 1.0, f"{name}: expected 1 for {c}"

    def test_unknown_token_in_condition_name_is_ignored(self):
        """Unknown tokens (not valid criterion names) are silently skipped."""
        # "invalid" is not a criterion name; only "metabolism" should be disabled
        matrix = build_criterion_presence_matrix(["drop_invalid_metabolism"], CRITERIA)
        assert matrix[0, CRITERIA.index("metabolism")] == 0.0
        for j, c in enumerate(CRITERIA):
            if c != "metabolism":
                assert matrix[0, j] == 1.0

    def test_response_reproduction_parsing_unambiguous(self):
        """Verify 'response' and 'reproduction' are not confused by the parser."""
        matrix = build_criterion_presence_matrix(["drop_response_reproduction"], CRITERIA)
        assert matrix[0, CRITERIA.index("response")] == 0.0
        assert matrix[0, CRITERIA.index("reproduction")] == 0.0
        assert matrix[0, CRITERIA.index("metabolism")] == 1.0


# ── analyze_criterion_reduction: stability_selection ─────────────────────────


class TestStabilitySelection:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 5))
        y = rng.standard_normal(40)
        freqs = stability_selection(X, y, n_bootstraps=30)
        assert freqs.shape == (5,)

    def test_frequencies_in_unit_interval(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 4))
        y = rng.standard_normal(40)
        freqs = stability_selection(X, y, n_bootstraps=30)
        assert np.all(freqs >= 0.0)
        assert np.all(freqs <= 1.0)

    def test_strong_predictor_ranked_first(self):
        # X[:,0] has very strong signal; X[:,1] is pure noise
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 2))
        y = X[:, 0] * 5.0 + rng.standard_normal(60) * 0.05
        freqs = stability_selection(X, y, n_bootstraps=100)
        # The informative feature should be selected more often than the noise
        assert freqs[0] > freqs[1]

    def test_elasticnet_variant_runs(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((30, 3))
        y = rng.standard_normal(30)
        freqs = stability_selection(X, y, n_bootstraps=20, model="elasticnet")
        assert freqs.shape == (3,)

    def test_raises_on_all_failures(self):
        # NaN targets cause LassoCV.fit to raise ValueError on every bootstrap
        # iteration (sklearn check_X_y rejects NaN).  stability_selection must
        # propagate this as RuntimeError when n_success == 0.
        X = np.random.default_rng(0).standard_normal((20, 3))
        y = np.full(20, np.nan)
        with pytest.raises(RuntimeError, match="All .* bootstrap iterations failed"):
            stability_selection(X, y, n_bootstraps=10)


# ── analyze_surrogates: extract_run_features ─────────────────────────────────


def _make_run_data(n_samples: int = 50, total_reproduction_events: int = 10) -> dict:
    """Synthetic run data with incrementing energy_mean (for slope/autocorr)."""
    samples = []
    for i in range(n_samples):
        samples.append(
            {
                "step": i * 20,
                "alive_count": 10 + (i % 5),
                "energy_mean": 1.0 + 0.01 * i,  # gently rising
                "waste_mean": 0.5 - 0.002 * i,  # gently falling
                "boundary_mean": 0.8 + 0.001 * i,
                "birth_count": 1 if i % 5 == 0 else 0,
                "death_count": 0,
                "genome_diversity": 0.3 + 0.005 * i,
                "spatial_cohesion_mean": 0.1,
                "maturity_mean": 0.7,
                "internal_state_std": [0.1, 0.2, 0.1, 0.15],
                "resource_total": 100.0,
            }
        )
    return {
        "samples": samples,
        "total_reproduction_events": total_reproduction_events,
        "final_alive_count": 12,
    }


class TestExtractRunFeatures:
    def test_returns_all_feature_names(self):
        run = _make_run_data()
        features = extract_run_features(run, total_steps=2000)
        for fname in FEATURE_NAMES:
            assert fname in features, f"Missing feature: {fname}"

    def test_feature_count_matches_names(self):
        run = _make_run_data()
        features = extract_run_features(run, total_steps=2000)
        assert len(features) == len(FEATURE_NAMES)

    def test_alive_auc_positive(self):
        run = _make_run_data()
        features = extract_run_features(run, total_steps=2000)
        assert features["alive_auc"] > 0.0

    def test_birth_rate_positive_when_births_occur(self):
        run = _make_run_data(n_samples=10)
        features = extract_run_features(run, total_steps=180)
        assert features["birth_rate"] > 0.0

    def test_death_rate_zero_when_no_deaths(self):
        run = _make_run_data()
        features = extract_run_features(run, total_steps=2000)
        assert features["death_rate"] == pytest.approx(0.0)

    def test_energy_autocorr_in_valid_range(self):
        run = _make_run_data()
        features = extract_run_features(run, total_steps=2000)
        assert -1.0 <= features["energy_autocorr"] <= 1.0

    def test_waste_slope_negative_for_declining_waste(self):
        # waste_mean is set up to decrease over time
        run = _make_run_data()
        features = extract_run_features(run, total_steps=2000)
        assert features["waste_slope"] < 0.0

    def test_reproduction_events_from_run_total(self):
        run = _make_run_data(total_reproduction_events=20)
        features = extract_run_features(run, total_steps=1000)
        assert features["reproduction_events"] == pytest.approx(20.0 / 1000.0)

    def test_all_values_finite(self):
        run = _make_run_data()
        features = extract_run_features(run, total_steps=2000)
        for k, v in features.items():
            assert np.isfinite(v), f"Non-finite feature: {k} = {v}"

    def test_empty_samples_returns_zeros(self):
        run = {"samples": [], "total_reproduction_events": 0, "final_alive_count": 0}
        features = extract_run_features(run, total_steps=2000)
        for fname in FEATURE_NAMES:
            assert fname in features
            assert np.isfinite(features[fname])
