"""Tests for boundary topology analysis (Phase 5 — peer review revision).

Tests Mann-Whitney U comparison, delta% by topology, rank ordering
assessment, and the end-to-end analysis pipeline with synthetic data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Tests: Mann-Whitney U wrapper
# ---------------------------------------------------------------------------


class TestMannWhitneyComparison:
    """Test topology comparison via Mann-Whitney U."""

    def test_different_distributions(self):
        from analyze_boundary_topology import mann_whitney_comparison

        a = [10, 20, 30, 40, 50]
        b = [60, 70, 80, 90, 100]
        result = mann_whitney_comparison(a, b)
        assert result["p_value"] < 0.05
        assert result["n_a"] == 5
        assert result["n_b"] == 5

    def test_same_distribution(self):
        from analyze_boundary_topology import mann_whitney_comparison

        a = [50, 51, 52, 53, 54]
        b = [50, 51, 52, 53, 54]
        result = mann_whitney_comparison(a, b)
        assert result["p_value"] > 0.05

    def test_insufficient_data(self):
        from analyze_boundary_topology import mann_whitney_comparison

        result = mann_whitney_comparison([1], [])
        assert math.isnan(result["p_value"])


# ---------------------------------------------------------------------------
# Tests: Delta% by topology
# ---------------------------------------------------------------------------


class TestDeltaByTopology:
    """Test delta% computation per topology."""

    def test_basic_delta(self):
        from analyze_boundary_topology import compute_topology_deltas

        # baseline=100, ablation=50 → Δ% = -50
        data = {
            "toroidal": {"normal": [100, 100], "no_metabolism": [50, 50]},
            "bounded": {"normal": [80, 80], "no_metabolism": [20, 20]},
        }
        result = compute_topology_deltas(data)
        assert "toroidal" in result
        assert "bounded" in result
        assert result["toroidal"]["no_metabolism"] == -50.0
        assert result["bounded"]["no_metabolism"] == -75.0

    def test_zero_baseline_returns_nan(self):
        from analyze_boundary_topology import compute_topology_deltas

        data = {
            "toroidal": {"normal": [0, 0], "no_metabolism": [50, 50]},
        }
        result = compute_topology_deltas(data)
        assert math.isnan(result["toroidal"]["no_metabolism"])


# ---------------------------------------------------------------------------
# Tests: Rank ordering comparison
# ---------------------------------------------------------------------------


class TestRankComparison:
    """Test rank ordering stability between topologies."""

    def test_same_ordering(self):
        from analyze_boundary_topology import compare_rank_ordering

        deltas_toro = {"no_metabolism": -80.0, "no_response": -40.0, "no_reproduction": -20.0}
        deltas_bounded = {"no_metabolism": -70.0, "no_response": -30.0, "no_reproduction": -10.0}
        result = compare_rank_ordering(deltas_toro, deltas_bounded)
        assert result["spearman_rho"] == 1.0
        assert result["ranks_match"]

    def test_different_ordering(self):
        from analyze_boundary_topology import compare_rank_ordering

        deltas_toro = {"no_metabolism": -80.0, "no_response": -40.0, "no_reproduction": -20.0}
        deltas_bounded = {"no_metabolism": -10.0, "no_response": -70.0, "no_reproduction": -90.0}
        result = compare_rank_ordering(deltas_toro, deltas_bounded)
        assert result["spearman_rho"] < 1.0
        assert not result["ranks_match"]

    def test_too_few_conditions(self):
        from analyze_boundary_topology import compare_rank_ordering

        result = compare_rank_ordering({"a": -10.0}, {"a": -20.0})
        assert math.isnan(result["spearman_rho"])


# ---------------------------------------------------------------------------
# Tests: End-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration test with synthetic boundary topology data."""

    def test_full_pipeline(self, tmp_path: Path):
        from analyze_boundary_topology import analyze_boundary_topology

        data_dir = tmp_path / "boundary_sweep"
        data_dir.mkdir()

        conditions = ["normal", "no_metabolism", "no_reproduction", "no_response"]
        topologies = ["toroidal", "bounded"]

        for topo in topologies:
            for cond in conditions:
                for seed in range(100, 105):
                    # Bounded gets fewer alive in ablations
                    if cond == "normal":
                        alive = 80 if topo == "toroidal" else 60
                    elif cond == "no_metabolism":
                        alive = 5 if topo == "toroidal" else 3
                    elif cond == "no_reproduction":
                        alive = 40 if topo == "toroidal" else 30
                    else:
                        alive = 30 if topo == "toroidal" else 20

                    result = {
                        "final_alive_count": alive,
                        "samples": [{"step": 2000, "alive_count": alive}],
                    }
                    fname = f"boundary_{topo}_{cond}_seed{seed}.json"
                    (data_dir / fname).write_text(json.dumps(result))

        result = analyze_boundary_topology(data_dir=data_dir, n_seeds=5)
        assert "deltas_by_topology" in result
        assert "rank_comparison" in result
        assert "mann_whitney_by_condition" in result
        assert result["rank_comparison"]["ranks_match"]
