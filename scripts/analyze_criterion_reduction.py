"""Criterion-level analysis for Phase 1 criterion-reduction pipeline.

Loads per-seed JSON files from experiments/ablation_sweep/ and:
  1. Computes scalar performance metrics per condition (averaged over seeds)
  2. Builds 30×7 criterion-presence matrix (binary feature encoding)
  3. PCA on performance matrix to identify latent life-likeness dimensions
  4. LASSO + Elastic Net + 500-bootstrap stability selection
  5. Bootstrapped Pareto curve: explained life-likeness vs number of criteria
  6. Mixed LME robustness check (statsmodels, if installed)

Pre-registered decision rules (docs/research/decision_rules.md — committed before
running this script):
  - Stable feature: selection frequency > 0.60 (500 bootstraps)
  - Sufficient set: R² ≥ 0.85 on held-out conditions AND Cohen's d ≥ 0.5 vs all-off
  - Agreement gate: LASSO and Elastic Net must select same features
  - Pareto threshold: ≥ 85% of full-system alive_auc

Held-out test set (pre-specified): pairwise ablations involving 'evolution' (6 conditions).

Output: experiments/criterion_reduction_analysis.json

Usage:
    uv run python scripts/analyze_criterion_reduction.py
    uv run python scripts/analyze_criterion_reduction.py --sweep-dir experiments/ablation_sweep
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

# Re-use shared helpers from the sweep script so definitions stay DRY
from analyses.results.statistics import cohens_d
from experiment_ablation_sweep import (
    CRITERIA,
    STEPS,
    ablation_conditions,
    build_criterion_presence_matrix,
    compute_performance_metrics,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

try:
    import pandas as pd  # statsmodels depends on pandas
    import statsmodels.formula.api as smf

    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False
    warnings.warn(
        "statsmodels/pandas not installed; LME robustness check will be skipped. "
        "Install with: uv add statsmodels",
        stacklevel=1,
    )

# NumPy 2.0 compatibility
try:
    _trapz = np.trapezoid  # type: ignore[attr-defined]
except AttributeError:
    _trapz = np.trapz  # type: ignore[attr-defined]

# ── constants ─────────────────────────────────────────────────────────────────

PERFORMANCE_METRICS = [
    "alive_auc",
    "energy_stability",
    "boundary_integrity",
    "homeostasis_quality",
    "reproduction_rate",
    "genome_diversity_late",
    "spatial_cohesion",
]

# Pairwise conditions involving 'evolution' — pre-registered held-out test set
# (6 conditions, chosen a priori; see decision_rules.md §5)
HELD_OUT_CONDITIONS: frozenset[str] = frozenset(
    f"drop_{c1}_{c2}"
    for c1, c2 in __import__("itertools").combinations(CRITERIA, 2)
    if "evolution" in (c1, c2)
)

STABILITY_THRESHOLD = 0.60   # §1 of decision_rules.md
SUFFICIENCY_R2 = 0.85         # §2a
SUFFICIENCY_COHEN_D = 0.50    # §2b
PARETO_THRESHOLD = 0.85       # §4
N_BOOTSTRAPS = 500            # §1


# ── data loading ──────────────────────────────────────────────────────────────


def load_condition_data(
    sweep_dir: Path,
    condition_name: str,
    seeds: list[int],
) -> list[dict]:
    """Load all seed results for one condition.

    Returns list of run dicts (one per seed that exists on disk).
    Missing files are silently skipped; caller should check length.
    """
    results = []
    for seed in seeds:
        path = sweep_dir / f"{condition_name}_seed{seed}.json"
        if path.exists():
            with open(path) as f:
                results.append(json.load(f))
    return results


def load_all_conditions(
    sweep_dir: Path,
    seeds: list[int],
) -> dict[str, list[dict]]:
    """Load all 30 ablation conditions from the sweep directory.

    Returns:
        Dict mapping condition_name → list[run_dict] (one per seed).
    """
    all_cond = ablation_conditions()
    data: dict[str, list[dict]] = {}
    for cond_name, _ in all_cond:
        runs = load_condition_data(sweep_dir, cond_name, seeds)
        if runs:
            data[cond_name] = runs
    return data


# ── performance matrix construction ──────────────────────────────────────────


def build_performance_matrix(
    condition_data: dict[str, list[dict]],
) -> tuple[np.ndarray, list[str]]:
    """Compute performance matrix: (n_conditions × n_metrics), averaged over seeds.

    Args:
        condition_data: Dict mapping condition_name → list of run dicts.

    Returns:
        (matrix, condition_names) where matrix[i, j] is mean metric j for condition i.
    """
    condition_names = list(condition_data.keys())
    n_cond = len(condition_names)
    n_met = len(PERFORMANCE_METRICS)
    matrix = np.zeros((n_cond, n_met))

    for i, cond_name in enumerate(condition_names):
        runs = condition_data[cond_name]
        seed_metrics = [
            compute_performance_metrics(r.get("samples", []), STEPS)
            for r in runs
        ]
        for j, met in enumerate(PERFORMANCE_METRICS):
            vals = [m[met] for m in seed_metrics]
            matrix[i, j] = float(np.mean(vals)) if vals else 0.0

    return matrix, condition_names


def build_per_seed_matrix(
    condition_data: dict[str, list[dict]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-seed (600-row) matrices for LME robustness check.

    Returns:
        X_pres: (n_obs, n_criteria) criterion presence binary matrix
        Y_perf: (n_obs, n_metrics) performance metrics per observation
        seeds_idx: (n_obs,) integer seed index for random-effect grouping
    """
    rows_X, rows_Y, rows_seed = [], [], []

    for cond_name, runs in condition_data.items():
        pres = build_criterion_presence_matrix([cond_name], CRITERIA)[0]
        for run in runs:
            seed = run.get("seed", 0)
            perf = compute_performance_metrics(run.get("samples", []), STEPS)
            rows_X.append(pres)
            rows_Y.append([perf[m] for m in PERFORMANCE_METRICS])
            rows_seed.append(seed)

    X_pres = np.array(rows_X, dtype=float)
    Y_perf = np.array(rows_Y, dtype=float)
    seeds_idx = np.array(rows_seed, dtype=int)
    return X_pres, Y_perf, seeds_idx


# ── stability selection ───────────────────────────────────────────────────────


def stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstraps: int = N_BOOTSTRAPS,
    subsample_ratio: float = 0.5,
    model: str = "lasso",
    random_state: int = 42,
) -> np.ndarray:
    """Bootstrap stability selection with LASSO or Elastic Net.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        n_bootstraps: Number of bootstrap sub-samples.
        subsample_ratio: Fraction of samples per sub-sample (no replacement).
        model: "lasso" or "elasticnet".
        random_state: Base RNG seed.

    Returns:
        selection_freq: Array (n_features,) with values in [0, 1].
    """
    n_samples, n_features = X.shape
    n_sub = max(2, int(n_samples * subsample_ratio))
    rng = np.random.default_rng(random_state)
    counts = np.zeros(n_features)

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_sub, replace=False)
        X_sub, y_sub = X[idx], y[idx]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)
        seed_int = int(rng.integers(0, 2**31))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if model == "elasticnet":
                    est = ElasticNetCV(cv=min(3, n_sub), max_iter=2000, random_state=seed_int)
                else:
                    est = LassoCV(cv=min(3, n_sub), max_iter=2000, random_state=seed_int)
                est.fit(X_scaled, y_sub)
                counts += (np.abs(est.coef_) > 1e-10).astype(float)
            except Exception:
                pass  # degenerate sub-sample; skip

    return counts / n_bootstraps


def _combined_stability(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstraps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run stability selection for both LASSO and Elastic Net."""
    lasso_freq = stability_selection(X, y, n_bootstraps=n_bootstraps, model="lasso")
    enet_freq = stability_selection(X, y, n_bootstraps=n_bootstraps, model="elasticnet")
    return lasso_freq, enet_freq


# ── Pareto curve ──────────────────────────────────────────────────────────────


def compute_pareto_curve(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    stability_scores: np.ndarray,
    n_boot: int = 500,
    random_state: int = 42,
) -> list[dict]:
    """Bootstrapped Pareto curve: explained variance vs number of features.

    Features are added in decreasing stability-score order.
    R² is estimated via Ridge regression on each bootstrap resample.

    Args:
        X: StandardScaled feature matrix (n_samples, n_features).
        y: Target (n_samples,). Should be the primary life-likeness metric.
        feature_names: Names for the n_features columns of X.
        stability_scores: Selection frequencies (n_features,); determines add order.
        n_boot: Bootstrap samples for CI estimation.
        random_state: RNG seed.

    Returns:
        List of dicts [{k, added_feature, r2_mean, r2_ci_lo, r2_ci_hi}, ...].
    """
    n_features = X.shape[1]
    sorted_idx = np.argsort(stability_scores)[::-1]
    rng = np.random.default_rng(random_state)
    n = len(y)
    curve = []

    for k in range(1, n_features + 1):
        selected = sorted_idx[:k]
        X_k = X[:, selected]
        r2_boots: list[float] = []

        for _ in range(n_boot):
            boot_idx = rng.choice(n, size=n, replace=True)
            X_b, y_b = X_k[boot_idx], y[boot_idx]
            try:
                est = Ridge(alpha=1.0)
                est.fit(X_b, y_b)
                r2_boots.append(float(r2_score(y_b, est.predict(X_b))))
            except Exception:
                pass

        if r2_boots:
            r2_arr = np.array(r2_boots)
            r2_mean = float(np.mean(r2_arr))
            r2_lo = float(np.percentile(r2_arr, 2.5))
            r2_hi = float(np.percentile(r2_arr, 97.5))
        else:
            r2_mean = r2_lo = r2_hi = 0.0

        curve.append(
            {
                "k": k,
                "added_feature": feature_names[sorted_idx[k - 1]],
                "r2_mean": r2_mean,
                "r2_ci_lo": r2_lo,
                "r2_ci_hi": r2_hi,
            }
        )
    return curve


# ── LME robustness check ──────────────────────────────────────────────────────


def _run_lme_check(
    X_pres: np.ndarray,
    Y_perf: np.ndarray,
    seeds_idx: np.ndarray,
) -> dict[str, dict]:
    """Run mixed-effects LME: metric ~ criteria_flags + (1|seed).

    Returns dict mapping metric_name → {AIC, criterion_coefs}.
    Skipped (returns {}) if statsmodels is not installed.
    """
    if not _HAS_STATSMODELS:
        return {}

    results: dict[str, dict] = {}
    crit_cols = [f"c_{c}" for c in CRITERIA]
    df = pd.DataFrame(X_pres, columns=crit_cols)
    df["seed_group"] = seeds_idx.astype(str)

    formula = " + ".join(crit_cols)

    for j, met in enumerate(PERFORMANCE_METRICS):
        df["target"] = Y_perf[:, j]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md = smf.mixedlm(
                    f"target ~ {formula}", df, groups=df["seed_group"]
                )
                mdf = md.fit(reml=True, method="lbfgs")
            results[met] = {
                "AIC": float(mdf.aic),
                "criterion_coefs": {
                    c: float(mdf.params.get(f"c_{c}", 0.0)) for c in CRITERIA
                },
            }
        except Exception as exc:
            results[met] = {"error": str(exc)}

    return results


# ── main analysis ─────────────────────────────────────────────────────────────


def _identify_minimal_set(
    lasso_freq: np.ndarray,
    enet_freq: np.ndarray,
    threshold: float,
) -> list[str]:
    """Return criteria stable under both LASSO and Elastic Net."""
    lasso_stable = set(np.where(lasso_freq > threshold)[0])
    enet_stable = set(np.where(enet_freq > threshold)[0])
    agreed = sorted(lasso_stable & enet_stable)
    return [CRITERIA[i] for i in agreed]


def run_analysis(
    sweep_dir: Path,
    seeds: list[int],
    out_path: Path,
) -> dict:
    """Run the full criterion-reduction analysis.

    Args:
        sweep_dir: Directory containing per-seed JSON files.
        seeds: Discovery seed list (should exclude confirmatory seeds 20+).
        out_path: Where to write criterion_reduction_analysis.json.

    Returns:
        Analysis result dict (also written to out_path).
    """
    print(f"Loading data from {sweep_dir} ...")
    condition_data = load_all_conditions(sweep_dir, seeds)
    if not condition_data:
        raise FileNotFoundError(
            f"No ablation sweep data found in {sweep_dir}. "
            "Run experiment_ablation_sweep.py first."
        )
    print(f"  Loaded {len(condition_data)} conditions.")

    # Split into train (discovery) and test (held-out)
    train_conds = {k: v for k, v in condition_data.items() if k not in HELD_OUT_CONDITIONS}
    test_conds = {k: v for k, v in condition_data.items() if k in HELD_OUT_CONDITIONS}
    print(
        f"  Train conditions: {len(train_conds)} | "
        f"Held-out test: {len(test_conds)}"
    )

    # ── Step A: performance matrix (averaged over seeds) ──────────────────
    print("Building performance matrices ...")
    perf_matrix, cond_names = build_performance_matrix(train_conds)
    # Presence matrix aligned with cond_names
    pres_matrix = build_criterion_presence_matrix(cond_names, CRITERIA)

    # ── Step B: PCA on performance matrix ────────────────────────────────
    print("Running PCA ...")
    scaler_perf = StandardScaler()
    perf_scaled = scaler_perf.fit_transform(perf_matrix)
    pca = PCA()
    pca.fit(perf_scaled)
    explained_variance = pca.explained_variance_ratio_.tolist()
    n_pcs_90 = int(np.searchsorted(np.cumsum(explained_variance), 0.90)) + 1
    print(
        f"  {n_pcs_90} PCs explain ≥90% of variance "
        f"(top-3 ratios: {explained_variance[:3]})"
    )

    # ── Step C: Stability selection (per target metric) ──────────────────
    print(f"Running stability selection ({N_BOOTSTRAPS} bootstraps) ...")
    scaler_pres = StandardScaler()
    X_train = scaler_pres.fit_transform(pres_matrix)

    all_lasso: dict[str, np.ndarray] = {}
    all_enet: dict[str, np.ndarray] = {}

    for j, met in enumerate(PERFORMANCE_METRICS):
        y = perf_matrix[:, j]
        lf, ef = _combined_stability(X_train, y, N_BOOTSTRAPS)
        all_lasso[met] = lf
        all_enet[met] = ef
        print(
            f"  {met}: LASSO stable={list(np.where(lf > STABILITY_THRESHOLD)[0])} "
            f"| Enet stable={list(np.where(ef > STABILITY_THRESHOLD)[0])}"
        )

    # Aggregate stability: mean frequency across all target metrics
    mean_lasso = np.mean(list(all_lasso.values()), axis=0)
    mean_enet = np.mean(list(all_enet.values()), axis=0)
    minimal_set = _identify_minimal_set(mean_lasso, mean_enet, STABILITY_THRESHOLD)
    print(f"  Minimal sufficient set: {minimal_set}")

    # ── Step D: Pareto curve ──────────────────────────────────────────────
    print("Computing Pareto curve ...")
    # Primary target: alive_auc (index 0)
    y_primary = perf_matrix[:, PERFORMANCE_METRICS.index("alive_auc")]
    pareto_curve = compute_pareto_curve(
        X_train, y_primary, CRITERIA, mean_lasso, n_boot=500
    )

    # ── Step E: held-out R² ───────────────────────────────────────────────
    held_out_r2: float | None = None
    if test_conds and minimal_set:
        test_perf, test_cond_names = build_performance_matrix(test_conds)
        test_pres = build_criterion_presence_matrix(test_cond_names, CRITERIA)
        X_test = scaler_pres.transform(test_pres)
        y_test = test_perf[:, PERFORMANCE_METRICS.index("alive_auc")]
        if len(minimal_set) > 0 and len(y_test) > 0:
            # Select only minimal-set columns for Ridge prediction
            min_idx = [CRITERIA.index(c) for c in minimal_set]
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train[:, min_idx], y_primary)
            y_pred = ridge.predict(X_test[:, min_idx])
            held_out_r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else None

    # ── Step F: Cohen's d vs all-off baseline ─────────────────────────────
    cohen_d_vs_alloff: float | None = None
    if "all_off" in condition_data and "full" in condition_data:
        alloff_runs = condition_data["all_off"]
        full_runs = condition_data["full"]
        alloff_auc = np.array([
            compute_performance_metrics(r.get("samples", []), STEPS)["alive_auc"]
            for r in alloff_runs
        ])
        full_auc = np.array([
            compute_performance_metrics(r.get("samples", []), STEPS)["alive_auc"]
            for r in full_runs
        ])
        cohen_d_vs_alloff = cohens_d(full_auc, alloff_auc)

    # ── Step G: LME robustness check ──────────────────────────────────────
    print("Running LME robustness check ...")
    X_pres_all, Y_perf_all, seeds_idx = build_per_seed_matrix(train_conds)
    lme_results = _run_lme_check(X_pres_all, Y_perf_all, seeds_idx)
    if lme_results:
        print("  LME complete.")
    else:
        print("  LME skipped (statsmodels not installed).")

    # ── Assemble output ───────────────────────────────────────────────────
    result = {
        "schema_version": 1,
        "conditions_used": cond_names,
        "held_out_conditions": sorted(HELD_OUT_CONDITIONS),
        "performance_matrix": perf_matrix.tolist(),
        "performance_metrics": PERFORMANCE_METRICS,
        "criterion_presence_matrix": pres_matrix.tolist(),
        "criteria": CRITERIA,
        "pca_explained_variance": explained_variance,
        "pca_n_components_90pct": n_pcs_90,
        "stability_scores_lasso": {
            met: all_lasso[met].tolist() for met in PERFORMANCE_METRICS
        },
        "stability_scores_enet": {
            met: all_enet[met].tolist() for met in PERFORMANCE_METRICS
        },
        "mean_stability_lasso": {c: float(mean_lasso[i]) for i, c in enumerate(CRITERIA)},
        "mean_stability_enet": {c: float(mean_enet[i]) for i, c in enumerate(CRITERIA)},
        "minimal_sufficient_set": minimal_set,
        "pareto_curve": pareto_curve,
        "held_out_r2": held_out_r2,
        "cohen_d_full_vs_alloff": cohen_d_vs_alloff,
        "lme_robustness": lme_results,
        "decision_rules": {
            "stability_threshold": STABILITY_THRESHOLD,
            "sufficiency_r2": SUFFICIENCY_R2,
            "sufficiency_cohen_d": SUFFICIENCY_COHEN_D,
            "pareto_threshold": PARETO_THRESHOLD,
            "n_bootstraps": N_BOOTSTRAPS,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {out_path}")

    # Print summary
    print("\n── SUMMARY ──────────────────────────────────────────")
    print(f"PCA: {n_pcs_90} components explain ≥90% of variance")
    print(f"Minimal sufficient set (stability >{STABILITY_THRESHOLD}): {minimal_set}")
    if pareto_curve:
        k3 = next((p for p in pareto_curve if p["k"] == 3), None)
        k7 = pareto_curve[-1]
        if k3:
            print(
                f"Pareto: k=3 → R²={k3['r2_mean']:.3f} "
                f"vs k={k7['k']} → R²={k7['r2_mean']:.3f}"
            )
    if held_out_r2 is not None:
        suffix = " ✓ SUFFICIENT" if held_out_r2 >= SUFFICIENCY_R2 else " ✗ INSUFFICIENT"
        print(f"Held-out R²={held_out_r2:.3f}{suffix}")
    print("─────────────────────────────────────────────────────")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Criterion-reduction analysis on ablation sweep data."
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help="Path to ablation_sweep directory (default: experiments/ablation_sweep).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: experiments/criterion_reduction_analysis.json).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(20)),
        help="Discovery seed list (default: 0–19).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    sweep_dir = (
        Path(args.sweep_dir) if args.sweep_dir else repo_root / "experiments" / "ablation_sweep"
    )
    out_path = (
        Path(args.out)
        if args.out
        else repo_root / "experiments" / "criterion_reduction_analysis.json"
    )
    run_analysis(sweep_dir, args.seeds, out_path)


if __name__ == "__main__":
    main()
