"""Tests for fitness landscape analysis (Phase 2 — peer review revision).

Tests Jonckheere-Terpstra trend test, parent-offspring regression,
and the end-to-end analysis pipeline with synthetic data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_organism_snapshot(
    stable_id: int,
    generation: int,
    energy: float,
    age_steps: int = 100,
) -> dict:
    return {
        "stable_id": stable_id,
        "generation": generation,
        "age_steps": age_steps,
        "energy": energy,
        "waste": 0.1,
        "boundary_integrity": 0.8,
        "maturity": 1.0,
        "center_x": 50.0,
        "center_y": 50.0,
        "n_agents": 10,
    }


def _make_snapshot_frame(step: int, organisms: list[dict]) -> dict:
    return {"step": step, "organisms": organisms}


def _make_lineage_event(
    step: int,
    parent_stable_id: int,
    child_stable_id: int,
    generation: int,
) -> dict:
    return {
        "step": step,
        "parent_stable_id": parent_stable_id,
        "child_stable_id": child_stable_id,
        "generation": generation,
    }


def _make_niche_result(
    organism_snapshots: list[dict],
    lineage_events: list[dict],
) -> dict:
    """Minimal RunSummary for niche experiment."""
    return {
        "schema_version": 1,
        "steps": 10000,
        "sample_every": 100,
        "final_alive_count": 50,
        "samples": [],
        "lifespans": [],
        "total_reproduction_events": len(lineage_events),
        "lineage_events": lineage_events,
        "organism_snapshots": organism_snapshots,
    }


# ---------------------------------------------------------------------------
# Tests: Jonckheere-Terpstra
# ---------------------------------------------------------------------------


class TestJonckheereTerpstra:
    """Test the Jonckheere-Terpstra trend test implementation."""

    def test_increasing_trend_detected(self):
        from analyze_fitness_landscape import jonckheere_terpstra_test

        # Monotonically increasing groups
        groups = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        stat, p = jonckheere_terpstra_test(groups)
        assert stat > 0
        assert p < 0.05

    def test_no_trend_not_significant(self):
        from analyze_fitness_landscape import jonckheere_terpstra_test

        # All groups same
        groups = [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]
        stat, p = jonckheere_terpstra_test(groups)
        assert p > 0.05

    def test_single_group_returns_nan(self):
        from analyze_fitness_landscape import jonckheere_terpstra_test

        groups = [[1.0, 2.0, 3.0]]
        stat, p = jonckheere_terpstra_test(groups)
        assert math.isnan(p) or p > 0.05  # degenerate case

    def test_empty_groups_handled(self):
        from analyze_fitness_landscape import jonckheere_terpstra_test

        groups: list[list[float]] = []
        stat, p = jonckheere_terpstra_test(groups)
        assert math.isnan(p)


# ---------------------------------------------------------------------------
# Tests: Parent-offspring regression
# ---------------------------------------------------------------------------


class TestParentOffspringRegression:
    """Test parent-offspring energy regression and bootstrap CI."""

    def test_positive_heritability(self):
        from analyze_fitness_landscape import parent_offspring_regression

        # Perfect positive correlation
        parent_energies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        offspring_energies = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        result = parent_offspring_regression(
            parent_energies,
            offspring_energies,
            n_bootstrap=100,
        )
        assert result["slope"] > 0
        assert result["ci_lower"] > 0  # CI should exclude zero

    def test_no_correlation(self):
        from analyze_fitness_landscape import parent_offspring_regression

        # Random-like, no correlation
        parent_energies = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
        offspring_energies = [0.8, 0.2, 0.7, 0.3, 0.9, 0.1]
        result = parent_offspring_regression(
            parent_energies,
            offspring_energies,
            n_bootstrap=100,
        )
        # CI should include zero (or slope close to zero)
        assert result["ci_lower"] < 0 or abs(result["slope"]) < 0.3

    def test_insufficient_data_handled(self):
        from analyze_fitness_landscape import parent_offspring_regression

        result = parent_offspring_regression([0.5], [0.5], n_bootstrap=100)
        assert math.isnan(result["slope"]) or result["n_pairs"] < 2


# ---------------------------------------------------------------------------
# Tests: Cohort segmentation
# ---------------------------------------------------------------------------


class TestCohortSegmentation:
    """Test organism segmentation by generation cohort."""

    def test_correct_binning(self):
        from analyze_fitness_landscape import segment_by_generation_cohort

        organisms = [
            {"generation": 0, "energy": 0.5},
            {"generation": 3, "energy": 0.6},
            {"generation": 7, "energy": 0.7},
            {"generation": 12, "energy": 0.8},
            {"generation": 18, "energy": 0.9},
        ]
        cohorts = segment_by_generation_cohort(organisms)
        # Default bins: 0-4, 5-9, 10-14, 15+
        assert len(cohorts) == 4
        assert len(cohorts[0]) == 2  # gen 0, 3
        assert len(cohorts[1]) == 1  # gen 7
        assert len(cohorts[2]) == 1  # gen 12
        assert len(cohorts[3]) == 1  # gen 18


# ---------------------------------------------------------------------------
# Tests: Lineage linkage
# ---------------------------------------------------------------------------


class TestLineageLinkage:
    """Test parent-offspring linkage from lineage_events + snapshots."""

    def test_links_parent_offspring_energies(self):
        from analyze_fitness_landscape import link_parent_offspring_energies

        snapshots = [
            _make_snapshot_frame(
                5000,
                [
                    _make_organism_snapshot(0, 0, energy=0.5),
                    _make_organism_snapshot(1, 1, energy=0.6),
                ],
            ),
        ]
        lineage = [
            _make_lineage_event(step=3000, parent_stable_id=0, child_stable_id=1, generation=1),
        ]
        parents, offspring = link_parent_offspring_energies(
            snapshots,
            lineage,
        )
        assert len(parents) == 1
        assert parents[0] == 0.5
        assert offspring[0] == 0.6

    def test_no_matches_returns_empty(self):
        from analyze_fitness_landscape import link_parent_offspring_energies

        snapshots = [
            _make_snapshot_frame(
                5000,
                [
                    _make_organism_snapshot(0, 0, energy=0.5),
                ],
            ),
        ]
        lineage = [
            _make_lineage_event(step=3000, parent_stable_id=0, child_stable_id=99, generation=1),
        ]
        parents, offspring = link_parent_offspring_energies(
            snapshots,
            lineage,
        )
        # child 99 not in snapshots, so no linkage
        assert len(parents) == 0


# ---------------------------------------------------------------------------
# Tests: End-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration test with synthetic niche result files."""

    def test_full_pipeline(self, tmp_path: Path):
        from analyze_fitness_landscape import analyze_fitness_landscape

        # Create synthetic evolved data with increasing energy
        snapshots = [
            _make_snapshot_frame(
                2000,
                [
                    _make_organism_snapshot(i, gen, energy=0.3 + gen * 0.05)
                    for i, gen in enumerate([0, 0, 1, 2])
                ],
            ),
            _make_snapshot_frame(
                5000,
                [
                    _make_organism_snapshot(i, gen, energy=0.4 + gen * 0.05)
                    for i, gen in enumerate([0, 1, 2, 5])
                ],
            ),
        ]
        lineage = [
            _make_lineage_event(1000, 0, 2, 1),
            _make_lineage_event(1500, 0, 3, 1),
            _make_lineage_event(2500, 2, 4, 2),
        ]

        data_dir = tmp_path / "fitness"
        data_dir.mkdir()

        for seed in range(5):
            result = _make_niche_result(snapshots, lineage)
            (data_dir / f"fitness_normal_seed{seed}.json").write_text(
                json.dumps(result),
            )
            # Clonal control: flat energy, no lineage
            flat_snaps = [
                _make_snapshot_frame(
                    2000, [_make_organism_snapshot(i, 0, energy=0.4) for i in range(4)]
                ),
                _make_snapshot_frame(
                    5000, [_make_organism_snapshot(i, 0, energy=0.4) for i in range(4)]
                ),
            ]
            control = _make_niche_result(flat_snaps, [])
            (data_dir / f"fitness_no_evolution_seed{seed}.json").write_text(
                json.dumps(control),
            )

        result = analyze_fitness_landscape(data_dir=data_dir, n_seeds=5, seed_start=0)
        assert "h1_trend_test" in result
        assert "h2_regression" in result
        assert "effect_size" in result
