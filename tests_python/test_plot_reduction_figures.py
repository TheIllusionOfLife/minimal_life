"""Tests for plot_reduction_figures.py

Each plot_figN function is exercised with synthetic minimal data.
Tests are hermetic — output goes to pytest's tmp_path, no disk pollution.
PNG dimensions are validated by reading the IHDR chunk directly (no PIL).
"""

from __future__ import annotations

import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_reduction_figures import (  # noqa: E402
    plot_criterion_pareto,
    plot_criterion_stability,
    plot_pca_biplot,
    plot_stability_heatmap,
    plot_surrogate_pareto,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CRITERIA = [
    "metabolism",
    "boundary",
    "homeostasis",
    "response",
    "reproduction",
    "evolution",
    "growth",
]
_METRICS = [
    "alive_auc",
    "energy_stability",
    "boundary_integrity",
    "homeostasis_quality",
    "reproduction_rate",
    "genome_diversity_late",
    "spatial_cohesion",
]


def _png_width(path: Path) -> int:
    """Read PNG width from IHDR without external libraries."""
    with open(path, "rb") as f:
        f.read(8)  # PNG magic
        f.read(4)  # IHDR chunk length
        f.read(4)  # "IHDR"
        return struct.unpack(">I", f.read(4))[0]


def _make_crit_data() -> dict:
    rng = np.random.default_rng(42)
    n_cond = 5

    return {
        "criteria": _CRITERIA,
        "performance_metrics": _METRICS,
        "conditions_used": [
            "all_off",
            "full",
            "drop_metabolism",
            "drop_boundary",
            "drop_homeostasis",
        ],
        "performance_matrix": rng.random((n_cond, 7)).tolist(),
        "mean_stability_lasso": {c: float(rng.uniform(0.2, 0.8)) for c in _CRITERIA},
        "mean_stability_enet": {c: float(rng.uniform(0.3, 0.9)) for c in _CRITERIA},
        "pareto_curve": [
            {
                "k": k,
                "added_feature": _CRITERIA[k - 1],
                "r2_mean": -0.5 + 0.08 * k,
                "r2_ci_lo": -1.0 + 0.05 * k,
                "r2_ci_hi": 0.1 + 0.05 * k,
            }
            for k in range(1, 8)
        ],
        "stability_scores_enet": {m: rng.random(7).tolist() for m in _METRICS},
    }


def _make_surr_data() -> dict:
    _FEATURES = [
        "homeostasis_var",
        "boundary_stability",
        "energy_autocorr",
        "spatial_cohesion_late",
        "maturity_late",
        "genome_diversity_late",
    ]
    return {
        "pareto_curve": [
            {
                "k": k,
                "added_feature": _FEATURES[k - 1],
                "r2_mean": -0.01 + 0.05 * k,
                "r2_ci_lo": -0.05 + 0.03 * k,
                "r2_ci_hi": 0.04 + 0.06 * k,
            }
            for k in range(1, 7)
        ],
        "held_out_r2": {
            "alive_auc": 0.219,
            "regulation_score": 0.668,
            "interdependence_score": 0.372,
            "adaptation_score": 0.129,
        },
    }


def _render(fig: plt.Figure, path: Path, dpi: int = 72) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1 — Criterion stability bar chart
# ---------------------------------------------------------------------------


class TestFig1CriterionStability:
    def test_renders_without_error(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        plot_criterion_stability(ax, _make_crit_data())
        out = tmp_path / "fig1.png"
        _render(fig, out)
        assert out.exists()
        assert _png_width(out) > 200

    def test_threshold_line_present(self, tmp_path: Path) -> None:
        """The axes must contain a horizontal line (the threshold dashed line)."""
        fig, ax = plt.subplots()
        plot_criterion_stability(ax, _make_crit_data())
        h_lines = ax.get_lines()
        assert len(h_lines) >= 1, "Expected at least one horizontal line for threshold"
        plt.close(fig)

    def test_bar_count_matches_criteria(self) -> None:
        fig, ax = plt.subplots()
        data = _make_crit_data()
        plot_criterion_stability(ax, data)
        n_criteria = len(data["mean_stability_enet"])
        # ax.bar() returns one BarContainer per call; we call it twice (LASSO + Enet).
        # Each container holds n_criteria individual bars.
        assert len(ax.containers) == 2
        assert len(ax.containers[0]) == n_criteria
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 — Criterion Pareto curve
# ---------------------------------------------------------------------------


class TestFig2CriterionPareto:
    def test_renders_without_error(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        plot_criterion_pareto(ax, _make_crit_data())
        out = tmp_path / "fig2.png"
        _render(fig, out)
        assert out.exists()
        assert _png_width(out) > 200

    def test_caveat_annotation_present(self) -> None:
        """The underpowered-Ridge caveat text box must be present."""
        fig, ax = plt.subplots()
        plot_criterion_pareto(ax, _make_crit_data())
        texts = [t.get_text() for t in ax.texts]
        assert any("underpowered Ridge" in t for t in texts), (
            "Expected caveat annotation mentioning 'underpowered Ridge'"
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 — Surrogate Pareto curve
# ---------------------------------------------------------------------------


class TestFig3SurrogatePareto:
    def test_renders_without_error(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        plot_surrogate_pareto(ax, _make_surr_data())
        out = tmp_path / "fig3.png"
        _render(fig, out)
        assert out.exists()
        assert _png_width(out) > 200

    def test_reference_line_present(self) -> None:
        """Both the R²=0 baseline and the regulation_score reference line must be present."""
        fig, ax = plt.subplots()
        surr_data = _make_surr_data()
        plot_surrogate_pareto(ax, surr_data)
        # Horizontal lines are axhline calls; each produces a Line2D with xdata = [0, 1].
        h_lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) == 2]
        assert len(h_lines) >= 2, f"Expected ≥2 h-lines (R²=0 + regulation ref), got {len(h_lines)}"
        # Verify one line sits at the regulation_score held-out R² value.
        reg_r2 = surr_data["held_out_r2"]["regulation_score"]
        y_values = [float(ln.get_ydata()[0]) for ln in h_lines]
        assert any(abs(y - reg_r2) < 1e-9 for y in y_values), (
            f"No h-line at regulation_score R²={reg_r2}; found y-values: {y_values}"
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4 — PCA biplot
# ---------------------------------------------------------------------------


class TestFig4PcaBiplot:
    def test_renders_without_error(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_pca_biplot(ax, _make_crit_data())
        out = tmp_path / "fig4.png"
        _render(fig, out)
        assert out.exists()
        assert _png_width(out) > 200

    def test_xlabel_contains_pc1(self) -> None:
        fig, ax = plt.subplots()
        plot_pca_biplot(ax, _make_crit_data())
        assert "PC1" in ax.get_xlabel()
        plt.close(fig)

    def test_full_and_alloff_in_conditions(self) -> None:
        """Synthetic data includes 'full' and 'all_off' — no exception raised."""
        data = _make_crit_data()
        assert "full" in data["conditions_used"]
        assert "all_off" in data["conditions_used"]
        fig, ax = plt.subplots()
        plot_pca_biplot(ax, data)  # must not raise
        plt.close(fig)

    def test_minimal_conditions(self, tmp_path: Path) -> None:
        """Works with the minimum of 3 conditions (PCA needs n ≥ n_components)."""
        rng = np.random.default_rng(7)
        data = {
            "criteria": _CRITERIA,
            "performance_metrics": _METRICS,
            "conditions_used": ["all_off", "full", "drop_metabolism"],
            "performance_matrix": rng.random((3, 7)).tolist(),
        }
        fig, ax = plt.subplots()
        plot_pca_biplot(ax, data)
        out = tmp_path / "fig4_min.png"
        _render(fig, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Fig 5 — Stability heatmap
# ---------------------------------------------------------------------------


class TestFig5StabilityHeatmap:
    def test_renders_without_error(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plot_stability_heatmap(ax, _make_crit_data())
        out = tmp_path / "fig5.png"
        _render(fig, out)
        assert out.exists()
        assert _png_width(out) > 200

    def test_ytick_count_matches_criteria(self) -> None:
        fig, ax = plt.subplots()
        data = _make_crit_data()
        plot_stability_heatmap(ax, data)
        assert len(ax.get_yticks()) == len(data["criteria"])
        plt.close(fig)

    def test_xtick_count_matches_metrics(self) -> None:
        fig, ax = plt.subplots()
        data = _make_crit_data()
        plot_stability_heatmap(ax, data)
        assert len(ax.get_xticks()) == len(data["stability_scores_enet"])
        plt.close(fig)
