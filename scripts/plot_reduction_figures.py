"""Plot Phase 1 figures for the criterion-reduction analysis.

Reads from:
    experiments/criterion_reduction_analysis.json
    experiments/surrogate_analysis.json

Writes to:
    figures/reduction/fig_criterion_stability.png
    figures/reduction/fig_criterion_pareto.png
    figures/reduction/fig_surrogate_pareto.png
    figures/reduction/fig_pca_biplot.png
    figures/reduction/fig_stability_heatmap.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from _project_root import PROJECT_ROOT  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
FIG_DIR = PROJECT_ROOT / "figures" / "reduction"

# ---------------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------------
CRITERION_LABELS: dict[str, str] = {
    "metabolism": "Metabolism",
    "boundary": "Boundary",
    "homeostasis": "Homeostasis",
    "response": "Response",
    "reproduction": "Reproduction",
    "evolution": "Evolution",
    "growth": "Growth",
}

METRIC_LABELS: dict[str, str] = {
    "alive_auc": "Alive AUC",
    "energy_stability": "Energy Stab.",
    "boundary_integrity": "Boundary Int.",
    "homeostasis_quality": "Homeostasis Q.",
    "reproduction_rate": "Reprod. Rate",
    "genome_diversity_late": "Genome Div.",
    "spatial_cohesion": "Spatial Coh.",
}

# ---------------------------------------------------------------------------
# Global matplotlib style (consistent with paper/figures/)
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

STABILITY_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Figure functions — each takes only an Axes + data dict for testability
# ---------------------------------------------------------------------------


def plot_criterion_stability(ax: plt.Axes, crit_data: dict) -> None:
    """Fig 1: Grouped bar chart of LASSO vs Enet stability by criterion.

    Criteria are sorted by Enet stability descending.
    A horizontal dashed line marks the pre-registered threshold (0.60).
    """
    lasso = crit_data["mean_stability_lasso"]
    enet = crit_data["mean_stability_enet"]

    # Sort criteria by Enet stability descending
    criteria = sorted(enet, key=lambda c: enet[c], reverse=True)

    x = np.arange(len(criteria))
    width = 0.35

    ax.bar(
        x - width / 2,
        [lasso[c] for c in criteria],
        width,
        label="LASSO",
        color="#4878CF",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        [enet[c] for c in criteria],
        width,
        label="Enet",
        color="#E87A5D",
        alpha=0.85,
    )

    ax.axhline(
        STABILITY_THRESHOLD,
        color="#333333",
        linestyle="--",
        linewidth=0.9,
        alpha=0.75,
        label=f"Threshold ({STABILITY_THRESHOLD:.2f})",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CRITERION_LABELS.get(c, c) for c in criteria], rotation=25, ha="right"
    )
    ax.set_ylabel("Stability Frequency")
    ax.set_ylim(0, 1.05)
    ax.set_title("Criterion Stability (500-bootstrap Enet vs LASSO)")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_criterion_pareto(ax: plt.Axes, crit_data: dict) -> None:
    """Fig 2: Criterion Pareto curve — R² vs number of criteria (k=1..7).

    R² is negative because Ridge regression is underpowered at n=24 training
    conditions.  The curve is shown for transparency; directional Enet stability
    (Fig. 1) is the primary evidence.
    """
    curve = crit_data["pareto_curve"]
    ks = [pt["k"] for pt in curve]
    r2 = [pt["r2_mean"] for pt in curve]
    lo = [pt["r2_ci_lo"] for pt in curve]
    hi = [pt["r2_ci_hi"] for pt in curve]

    ax.plot(ks, r2, "o-", color="#4878CF", linewidth=1.4, markersize=5, zorder=5)
    ax.fill_between(ks, lo, hi, alpha=0.18, color="#4878CF", label="95% CI")
    ax.axhline(0, color="#888888", linestyle=":", linewidth=0.8, label="R²=0 reference")

    # Annotate each point with the added criterion
    for pt in curve:
        label = CRITERION_LABELS.get(pt["added_feature"], pt["added_feature"])
        ax.annotate(
            f"+{label}",
            (pt["k"], pt["r2_mean"]),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=6,
            color="#444444",
        )

    ax.set_xlabel("Number of criteria (k)")
    ax.set_ylabel("Out-of-fold R²")
    ax.set_xticks(ks)
    ax.set_title("Criterion Pareto Curve (Ridge, k=1..7)")
    ax.legend(loc="upper right", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.02,
        0.04,
        "Note: n=24 training conditions; R² negative due to underpowered Ridge.\n"
        "Directional Enet stability (Fig. 1) is the primary criterion evidence.",
        transform=ax.transAxes,
        fontsize=6,
        color="#555555",
        verticalalignment="bottom",
        bbox=dict(
            facecolor="white",
            edgecolor="#cccccc",
            boxstyle="round,pad=0.3",
            alpha=0.85,
        ),
    )


def plot_surrogate_pareto(ax: plt.Axes, surr_data: dict) -> None:
    """Fig 3: Surrogate Pareto curve with elbow annotation at k=3.

    Shades near-zero (k≤2) vs useful-signal (k≥3) regions.
    Overlays held-out R² for regulation_score as a reference line.
    """
    curve = surr_data["pareto_curve"]
    ks = [pt["k"] for pt in curve]
    r2 = [pt["r2_mean"] for pt in curve]
    lo = [pt["r2_ci_lo"] for pt in curve]
    hi = [pt["r2_ci_hi"] for pt in curve]

    # Background region shading
    ax.axvspan(0.5, 2.5, color="#fde0dc", alpha=0.35, zorder=0, label="Near-zero signal (k≤2)")
    ax.axvspan(
        2.5, max(ks) + 0.5, color="#dff0d8", alpha=0.25, zorder=0, label="Useful signal (k≥3)"
    )

    ax.plot(ks, r2, "o-", color="#4878CF", linewidth=1.4, markersize=5, zorder=5)
    ax.fill_between(ks, lo, hi, alpha=0.18, color="#4878CF", label="95% CI")
    ax.axhline(0, color="#888888", linestyle=":", linewidth=0.8, label="R²=0 reference")

    # Elbow annotation at k=3 — look up by k value, not list index.
    elbow_pt = next((pt for pt in curve if pt["k"] == 3), None)
    elbow_r2 = elbow_pt["r2_mean"] if elbow_pt is not None else r2[-1]
    ax.annotate(
        "Elbow (k=3)\n+energy_autocorr",
        (3, elbow_r2),
        xytext=(3.4, elbow_r2 - 0.04),
        fontsize=6.5,
        color="#222222",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#555555"),
    )

    # Held-out R² reference for regulation_score (best single-target result)
    reg_r2 = surr_data["held_out_r2"]["regulation_score"]
    ax.axhline(
        reg_r2,
        color="#E87A5D",
        linestyle="--",
        linewidth=0.9,
        label=f"Best held-out R² — regulation ({reg_r2:.3f})",
    )

    ax.set_xlabel("Number of surrogate features (k)")
    ax.set_ylabel("Out-of-fold R² (mean over targets)")
    ax.set_xticks(ks)
    ax.set_title("Surrogate Pareto Curve (k=1..6)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_pca_biplot(ax: plt.Axes, crit_data: dict) -> None:
    """Fig 4: PCA biplot of ablation conditions in PC1–PC2 space.

    Conditions are colored by alive_auc (first performance metric).
    'full' is marked with a star, 'all_off' with an X.
    """
    perf_matrix = np.array(crit_data["performance_matrix"])
    conditions = crit_data["conditions_used"]
    metrics = crit_data["performance_metrics"]

    # Column-wise standardization (z-score) before PCA.
    # ddof=1 produces NaN when n_rows==1; guard both zero and NaN so PCA
    # receives finite input even in degenerate synthetic-data scenarios.
    mu = perf_matrix.mean(axis=0)
    sd = perf_matrix.std(axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    perf_std = (perf_matrix - mu) / sd

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(perf_std)

    var1, var2 = pca.explained_variance_ratio_[:2] * 100

    if "alive_auc" not in metrics:
        raise ValueError(
            f"'alive_auc' not found in performance_metrics; available: {metrics}"
        )
    alive_auc_idx = metrics.index("alive_auc")
    color_vals = perf_matrix[:, alive_auc_idx]
    norm = mcolors.Normalize(vmin=color_vals.min(), vmax=color_vals.max())
    cmap = plt.cm.RdYlGn

    # Base scatter for all conditions
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color_vals,
        cmap=cmap,
        norm=norm,
        s=28,
        alpha=0.80,
        edgecolors="#555555",
        linewidths=0.4,
        zorder=5,
    )

    # Special markers for 'full' (star) and 'all_off' (X)
    for i, cond in enumerate(conditions):
        face_color = cmap(norm(color_vals[i]))
        if cond == "full":
            ax.scatter(
                coords[i, 0],
                coords[i, 1],
                marker="*",
                s=200,
                c=[face_color],
                edgecolors="black",
                linewidths=0.8,
                zorder=10,
            )
            ax.annotate(
                "full",
                (coords[i, 0], coords[i, 1]),
                fontsize=6.5,
                xytext=(5, 4),
                textcoords="offset points",
                color="#111111",
            )
        elif cond == "all_off":
            ax.scatter(
                coords[i, 0],
                coords[i, 1],
                marker="X",
                s=90,
                c=[face_color],
                edgecolors="black",
                linewidths=0.8,
                zorder=10,
            )
            ax.annotate(
                "all_off",
                (coords[i, 0], coords[i, 1]),
                fontsize=6.5,
                xytext=(5, 4),
                textcoords="offset points",
                color="#111111",
            )

    plt.colorbar(sc, ax=ax, label="Alive AUC", shrink=0.8)
    ax.set_xlabel(f"PC1 ({var1:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% var.)")
    ax.set_title(f"PCA Biplot of Ablation Conditions (PC1+PC2 = {var1 + var2:.1f}%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_stability_heatmap(ax: plt.Axes, crit_data: dict) -> None:
    """Fig 5: 7×7 heatmap of Enet stability — criteria (Y) × metrics (X).

    Rows are sorted by mean stability descending.
    Cells ≥ pre-registered threshold (0.60) receive a white dashed border.
    """
    criteria = crit_data["criteria"]
    stability_enet = crit_data["stability_scores_enet"]
    metrics = list(stability_enet.keys())

    # Validate list lengths before indexing to produce an actionable error.
    for metric, scores in stability_enet.items():
        if len(scores) != len(criteria):
            raise ValueError(
                f"stability_scores_enet['{metric}'] has {len(scores)} entries, "
                f"expected {len(criteria)} (one per criterion)"
            )

    # Build matrix: rows=criteria, cols=metrics
    mat = np.zeros((len(criteria), len(metrics)))
    for j, metric in enumerate(metrics):
        for i in range(len(criteria)):
            mat[i, j] = stability_enet[metric][i]

    # Sort rows by mean stability descending
    row_order = np.argsort(mat.mean(axis=1))[::-1]
    mat_sorted = mat[row_order, :]
    crit_sorted = [criteria[i] for i in row_order]

    im = ax.imshow(mat_sorted, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Stability Frequency", shrink=0.85)

    # Cell value annotations and threshold borders
    for i in range(mat_sorted.shape[0]):
        for j in range(mat_sorted.shape[1]):
            val = mat_sorted[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if val > 0.65 else "#333333",
            )
            if val >= STABILITY_THRESHOLD:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="white",
                    linestyle="--",
                    linewidth=1.2,
                    zorder=5,
                )
                ax.add_patch(rect)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(
        [METRIC_LABELS.get(m, m) for m in metrics], rotation=35, ha="right", fontsize=7
    )
    ax.set_yticks(range(len(crit_sorted)))
    ax.set_yticklabels(
        [CRITERION_LABELS.get(c, c) for c in crit_sorted], fontsize=8
    )
    ax.set_title("Enet Stability Heatmap: Criterion × Performance Metric")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    crit_path = EXPERIMENTS_DIR / "criterion_reduction_analysis.json"
    surr_path = EXPERIMENTS_DIR / "surrogate_analysis.json"
    for p in (crit_path, surr_path):
        if not p.exists():
            print(f"ERROR: Required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    crit_data = _load_json(crit_path)
    surr_data = _load_json(surr_path)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_criterion_stability(ax, crit_data)
    _save(fig, FIG_DIR / "fig_criterion_stability.png")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    plot_criterion_pareto(ax, crit_data)
    _save(fig, FIG_DIR / "fig_criterion_pareto.png")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    plot_surrogate_pareto(ax, surr_data)
    _save(fig, FIG_DIR / "fig_surrogate_pareto.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_pca_biplot(ax, crit_data)
    _save(fig, FIG_DIR / "fig_pca_biplot.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_stability_heatmap(ax, crit_data)
    _save(fig, FIG_DIR / "fig_stability_heatmap.png")


if __name__ == "__main__":
    main()
