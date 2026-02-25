"""Observable surrogate analysis for Phase 1 criterion-reduction pipeline.

Uses the same 600 runs as analyze_criterion_reduction.py but ignores which
criteria were toggled. Instead, extracts 14 scalar features from each run's
StepMetrics time series and identifies which 2–4 raw observables predict the
same performance targets.

Feature set (14 scalars per run):
  alive_auc, energy_mean_late, energy_autocorr, waste_slope,
  boundary_stability, homeostasis_var, birth_rate, death_rate,
  turnover_ratio, spatial_cohesion_late, genome_diversity_late,
  maturity_late, resource_efficiency, reproduction_events

Target scores (4 scalars, same definitions as analyze_criterion_reduction.py):
  alive_auc, regulation_score, interdependence_score, adaptation_score

Pre-registered decision rules: docs/research/decision_rules.md
  - Stable feature: selection frequency > 0.60 (500 bootstraps)
  - Agreement gate: LASSO and Elastic Net must agree
  - VIF threshold: remove features with VIF > 10 before fitting
  - Pareto with 95% bootstrapped CI bands

Output: experiments/surrogate_analysis.json

Usage:
    uv run python scripts/analyze_surrogates.py
    uv run python scripts/analyze_surrogates.py --sweep-dir experiments/ablation_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_criterion_reduction import (
    HELD_OUT_CONDITIONS,
    N_BOOTSTRAPS,
    STABILITY_THRESHOLD,
    _identify_minimal_set,
    compute_pareto_curve,
    stability_selection,
)
from experiment_ablation_sweep import (
    STEPS,
    ablation_conditions,
    compute_performance_metrics,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# NumPy 2.0 compatibility
try:
    _trapz = np.trapezoid  # type: ignore[attr-defined]
except AttributeError:
    _trapz = np.trapz  # type: ignore[attr-defined]

# ── feature definitions ───────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    "alive_auc",
    "energy_mean_late",
    "energy_autocorr",
    "waste_slope",
    "boundary_stability",
    "homeostasis_var",
    "birth_rate",
    "death_rate",
    "turnover_ratio",
    "spatial_cohesion_late",
    "genome_diversity_late",
    "maturity_late",
    "resource_efficiency",
    "reproduction_events",
]

TARGET_NAMES: list[str] = [
    "alive_auc",
    "regulation_score",
    "interdependence_score",
    "adaptation_score",
]

_EPS = 1e-8  # denominator guard


# ── feature extraction ────────────────────────────────────────────────────────


def extract_run_features(run_data: dict, total_steps: int) -> dict[str, float]:
    """Extract 14 scalar features from a single run's StepMetrics time series.

    Args:
        run_data: Dict with keys "samples", "total_reproduction_events".
        total_steps: Simulation step count for rate normalization.

    Returns:
        Dict mapping each name in FEATURE_NAMES to a finite float.
    """
    samples = run_data.get("samples", [])
    total_repr = run_data.get("total_reproduction_events", 0)

    if not samples:
        return {name: 0.0 for name in FEATURE_NAMES}

    steps_arr = np.array([s["step"] for s in samples], dtype=float)
    alive_arr = np.array([s["alive_count"] for s in samples], dtype=float)
    energy_arr = np.array([s["energy_mean"] for s in samples], dtype=float)
    waste_arr = np.array([s["waste_mean"] for s in samples], dtype=float)
    boundary_arr = np.array([s["boundary_mean"] for s in samples], dtype=float)
    resource_arr = np.array([s.get("resource_total", 0.0) for s in samples], dtype=float)
    genome_arr = np.array([s.get("genome_diversity", 0.0) for s in samples], dtype=float)
    spatial_arr = np.array([s.get("spatial_cohesion_mean", 0.0) for s in samples], dtype=float)
    maturity_arr = np.array([s.get("maturity_mean", 0.0) for s in samples], dtype=float)
    birth_arr = np.array([s["birth_count"] for s in samples], dtype=float)
    death_arr = np.array([s["death_count"] for s in samples], dtype=float)

    # internal_state_std is [f32; 4] → mean across 4 dims per sample
    ist_arr = np.array([
        sum(s.get("internal_state_std", [0.0, 0.0, 0.0, 0.0])) / 4.0
        for s in samples
    ], dtype=float)

    n_late = min(50, len(samples))
    n_total = max(total_steps, 1)

    # ── 1. alive_auc ──────────────────────────────────────────────────────
    alive_auc = float(_trapz(alive_arr, steps_arr) / n_total)

    # ── 2. energy_mean_late ───────────────────────────────────────────────
    energy_mean_late = float(np.mean(energy_arr[-n_late:]))

    # ── 3. energy_autocorr (lag-1 autocorrelation) ────────────────────────
    if len(energy_arr) >= 2:
        e_centered = energy_arr - energy_arr.mean()
        var = np.var(e_centered)
        if var > _EPS:
            energy_autocorr = float(
                np.mean(e_centered[:-1] * e_centered[1:]) / var
            )
        else:
            energy_autocorr = 0.0
    else:
        energy_autocorr = 0.0

    # ── 4. waste_slope (linear trend via least-squares) ───────────────────
    if len(steps_arr) >= 2:
        t_norm = (steps_arr - steps_arr.mean()) / (steps_arr.std() + _EPS)
        waste_slope = float(np.dot(t_norm, waste_arr) / (np.dot(t_norm, t_norm) + _EPS))
    else:
        waste_slope = 0.0

    # ── 5. boundary_stability (1/std of late-phase boundary_mean) ─────────
    bnd_late = boundary_arr[-n_late:]
    bnd_std = float(np.std(bnd_late)) if len(bnd_late) >= 2 else 0.0
    boundary_stability = 1.0 / (bnd_std + _EPS)

    # ── 6. homeostasis_var (mean internal_state_std in late phase) ────────
    homeostasis_var = float(np.mean(ist_arr[-n_late:]))

    # ── 7 & 8. birth_rate, death_rate ─────────────────────────────────────
    birth_rate = float(np.sum(birth_arr)) / n_total
    death_rate = float(np.sum(death_arr)) / n_total

    # ── 9. turnover_ratio ─────────────────────────────────────────────────
    turnover_ratio = birth_rate / (death_rate + _EPS)

    # ── 10. spatial_cohesion_late ─────────────────────────────────────────
    spatial_cohesion_late = float(np.mean(spatial_arr[-n_late:]))

    # ── 11. genome_diversity_late ─────────────────────────────────────────
    genome_diversity_late = float(np.mean(genome_arr[-n_late:]))

    # ── 12. maturity_late ─────────────────────────────────────────────────
    maturity_late = float(np.mean(maturity_arr[-n_late:]))

    # ── 13. resource_efficiency ───────────────────────────────────────────
    resource_mean = float(np.mean(resource_arr))
    resource_efficiency = energy_mean_late / (resource_mean + _EPS)

    # ── 14. reproduction_events ───────────────────────────────────────────
    reproduction_events = float(total_repr) / n_total

    feats = {
        "alive_auc": alive_auc,
        "energy_mean_late": energy_mean_late,
        "energy_autocorr": energy_autocorr,
        "waste_slope": waste_slope,
        "boundary_stability": boundary_stability,
        "homeostasis_var": homeostasis_var,
        "birth_rate": birth_rate,
        "death_rate": death_rate,
        "turnover_ratio": turnover_ratio,
        "spatial_cohesion_late": spatial_cohesion_late,
        "genome_diversity_late": genome_diversity_late,
        "maturity_late": maturity_late,
        "resource_efficiency": resource_efficiency,
        "reproduction_events": reproduction_events,
    }

    # Sanitise any non-finite values (clip rather than NaN-propagate)
    for k in feats:
        v = feats[k]
        if not np.isfinite(v):
            feats[k] = 0.0

    return feats


# ── target score extraction ───────────────────────────────────────────────────


def extract_target_scores(run_data: dict, total_steps: int) -> dict[str, float]:
    """Compute 4 target scores from a single run.

    Definitions mirror the performance metrics in analyze_criterion_reduction.py
    for cross-method comparability.

    Returns:
        Dict with keys: alive_auc, regulation_score,
        interdependence_score, adaptation_score.
    """
    perf = compute_performance_metrics(run_data.get("samples", []), total_steps)
    return {
        "alive_auc": perf["alive_auc"],
        "regulation_score": perf["boundary_integrity"],
        "interdependence_score": perf["reproduction_rate"],
        "adaptation_score": perf["genome_diversity_late"],
    }


# ── VIF collinearity check ────────────────────────────────────────────────────


def compute_vif(X: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Compute Variance Inflation Factor for each column of X.

    VIF_j = 1 / (1 − R²_j), where R²_j is the R² from regressing column j
    on all other columns.  VIF > 10 indicates problematic collinearity.

    Args:
        X: Feature matrix (n_samples, n_features), raw (not scaled).
        feature_names: Column names.

    Returns:
        Dict mapping feature_name → VIF value.
    """
    vif_dict: dict[str, float] = {}

    for j, name in enumerate(feature_names):
        y_j = X[:, j]
        X_rest = np.delete(X, j, axis=1)
        if X_rest.shape[1] == 0:
            vif_dict[name] = 1.0
            continue
        try:
            ridge = Ridge(alpha=1e-6, fit_intercept=True)
            ridge.fit(X_rest, y_j)
            r2 = float(r2_score(y_j, ridge.predict(X_rest)))
            vif_dict[name] = 1.0 / (1.0 - min(r2, 1.0 - 1e-12))
        except Exception:
            vif_dict[name] = float("inf")

    return vif_dict


# ── data loading ──────────────────────────────────────────────────────────────


def load_all_runs(sweep_dir: Path, seeds: list[int]) -> list[tuple[str, dict]]:
    """Load all run data as (condition_name, run_dict) pairs.

    Returns list sorted by (condition_name, seed) for reproducibility.
    """
    all_cond = ablation_conditions()
    runs: list[tuple[str, dict]] = []
    for cond_name, _ in all_cond:
        for seed in seeds:
            path = sweep_dir / f"{cond_name}_seed{seed}.json"
            if path.exists():
                with open(path) as f:
                    run = json.load(f)
                runs.append((cond_name, run))
    return runs


# ── full analysis ─────────────────────────────────────────────────────────────


def run_analysis(
    sweep_dir: Path,
    seeds: list[int],
    out_path: Path,
    vif_threshold: float = 10.0,
) -> dict:
    """Run the full surrogate-observable analysis.

    Args:
        sweep_dir: Directory with per-seed JSON files.
        seeds: Discovery seed list.
        out_path: Output JSON path.
        vif_threshold: Features with VIF above this are excluded.

    Returns:
        Analysis result dict (also written to out_path).
    """
    print(f"Loading all runs from {sweep_dir} ...")
    all_runs = load_all_runs(sweep_dir, seeds)
    if not all_runs:
        raise FileNotFoundError(
            f"No sweep data found in {sweep_dir}. "
            "Run experiment_ablation_sweep.py first."
        )
    print(f"  {len(all_runs)} run files loaded.")

    # ── Step A: feature extraction ────────────────────────────────────────
    print("Extracting features ...")
    feature_rows: list[list[float]] = []
    target_rows: list[list[float]] = []
    condition_labels: list[str] = []
    train_mask: list[bool] = []

    for cond_name, run in all_runs:
        feats = extract_run_features(run, STEPS)
        tgts = extract_target_scores(run, STEPS)
        feature_rows.append([feats[f] for f in FEATURE_NAMES])
        target_rows.append([tgts[t] for t in TARGET_NAMES])
        condition_labels.append(cond_name)
        train_mask.append(cond_name not in HELD_OUT_CONDITIONS)

    X_raw = np.array(feature_rows, dtype=float)
    Y = np.array(target_rows, dtype=float)
    is_train = np.array(train_mask, dtype=bool)
    print(
        f"  Feature matrix: {X_raw.shape} | "
        f"Train={is_train.sum()} | Held-out={(~is_train).sum()}"
    )

    # ── Step B: VIF collinearity check ───────────────────────────────────
    print(f"Computing VIF (threshold={vif_threshold}) ...")
    vif_scores = compute_vif(X_raw[is_train], FEATURE_NAMES)
    retained_features = [f for f in FEATURE_NAMES if vif_scores.get(f, 0.0) <= vif_threshold]
    removed_features = [f for f in FEATURE_NAMES if f not in retained_features]
    if removed_features:
        print(f"  Removed due to VIF > {vif_threshold}: {removed_features}")
    else:
        print(f"  All {len(FEATURE_NAMES)} features retained (VIF ≤ {vif_threshold}).")
    retained_idx = [FEATURE_NAMES.index(f) for f in retained_features]
    X_retained = X_raw[:, retained_idx]

    # ── Step C: Scaling ───────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_retained[is_train])
    X_test_scaled = scaler.transform(X_retained[~is_train]) if (~is_train).any() else None

    # ── Step D: Stability selection per target ────────────────────────────
    print(f"Running stability selection ({N_BOOTSTRAPS} bootstraps) ...")
    all_lasso: dict[str, np.ndarray] = {}
    all_enet: dict[str, np.ndarray] = {}

    for j, tgt in enumerate(TARGET_NAMES):
        y_train = Y[is_train, j]
        lf = stability_selection(
            X_train_scaled, y_train, n_bootstraps=N_BOOTSTRAPS, model="lasso"
        )
        ef = stability_selection(
            X_train_scaled, y_train, n_bootstraps=N_BOOTSTRAPS, model="elasticnet"
        )
        all_lasso[tgt] = lf
        all_enet[tgt] = ef
        stable_l = [retained_features[i] for i in np.where(lf > STABILITY_THRESHOLD)[0]]
        stable_e = [retained_features[i] for i in np.where(ef > STABILITY_THRESHOLD)[0]]
        print(f"  {tgt}: LASSO={stable_l} | Enet={stable_e}")

    mean_lasso = np.mean(list(all_lasso.values()), axis=0)
    mean_enet = np.mean(list(all_enet.values()), axis=0)
    minimal_set = _identify_minimal_set(mean_lasso, mean_enet, STABILITY_THRESHOLD)
    minimal_feature_names = [retained_features[i] for i in range(len(retained_features))
                             if i in [retained_features.index(f) for f in minimal_set]]
    print(f"  Minimal surrogate set: {minimal_feature_names}")

    # ── Step E: Pareto curve ──────────────────────────────────────────────
    print("Computing Pareto curve ...")
    y_primary_train = Y[is_train, TARGET_NAMES.index("alive_auc")]
    pareto_curve = compute_pareto_curve(
        X_train_scaled, y_primary_train, retained_features, mean_lasso, n_boot=500
    )

    # ── Step F: Held-out R² ───────────────────────────────────────────────
    held_out_r2: dict[str, float | None] = {}
    if X_test_scaled is not None and len(minimal_feature_names) > 0:
        min_idx_local = [retained_features.index(f) for f in minimal_feature_names]
        for j, tgt in enumerate(TARGET_NAMES):
            y_train_j = Y[is_train, j]
            y_test_j = Y[~is_train, j]
            if len(y_test_j) < 2:
                held_out_r2[tgt] = None
                continue
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_scaled[:, min_idx_local], y_train_j)
            y_pred_j = ridge.predict(X_test_scaled[:, min_idx_local])
            held_out_r2[tgt] = float(r2_score(y_test_j, y_pred_j))
    else:
        held_out_r2 = {t: None for t in TARGET_NAMES}

    # ── Assemble output ───────────────────────────────────────────────────
    result = {
        "schema_version": 1,
        "n_runs": len(all_runs),
        "feature_names": FEATURE_NAMES,
        "retained_features": retained_features,
        "removed_features_vif": removed_features,
        "vif_scores": vif_scores,
        "target_names": TARGET_NAMES,
        "feature_matrix_shape": list(X_raw.shape),
        "stability_scores_lasso": {
            tgt: all_lasso[tgt].tolist() for tgt in TARGET_NAMES
        },
        "stability_scores_enet": {
            tgt: all_enet[tgt].tolist() for tgt in TARGET_NAMES
        },
        "mean_stability_lasso": {
            f: float(mean_lasso[i]) for i, f in enumerate(retained_features)
        },
        "mean_stability_enet": {
            f: float(mean_enet[i]) for i, f in enumerate(retained_features)
        },
        "minimal_surrogate_set": minimal_feature_names,
        "pareto_curve": pareto_curve,
        "held_out_r2": held_out_r2,
        "decision_rules": {
            "stability_threshold": STABILITY_THRESHOLD,
            "vif_threshold": vif_threshold,
            "n_bootstraps": N_BOOTSTRAPS,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {out_path}")

    # Print summary
    print("\n── SURROGATE SUMMARY ────────────────────────────────")
    print(f"Minimal surrogate set: {minimal_feature_names}")
    if pareto_curve:
        k2 = next((p for p in pareto_curve if p["k"] == 2), None)
        k3 = next((p for p in pareto_curve if p["k"] == 3), None)
        if k2:
            lo, hi = k2["r2_ci_lo"], k2["r2_ci_hi"]
            print(f"Pareto k=2 → R²={k2['r2_mean']:.3f} [{lo:.3f}, {hi:.3f}]")
        if k3:
            lo, hi = k3["r2_ci_lo"], k3["r2_ci_hi"]
            print(f"Pareto k=3 → R²={k3['r2_mean']:.3f} [{lo:.3f}, {hi:.3f}]")
    for tgt, r2 in held_out_r2.items():
        if r2 is not None:
            print(f"Held-out R² ({tgt}): {r2:.3f}")
    print("─────────────────────────────────────────────────────")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observable surrogate analysis on ablation sweep data."
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help="Path to ablation_sweep directory (default: experiments/ablation_sweep).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: experiments/surrogate_analysis.json).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(20)),
        help="Discovery seed list (default: 0–19).",
    )
    parser.add_argument(
        "--vif-threshold",
        type=float,
        default=10.0,
        help="VIF threshold for collinearity pruning (default: 10.0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    sweep_dir = (
        Path(args.sweep_dir) if args.sweep_dir else repo_root / "experiments" / "ablation_sweep"
    )
    out_path = (
        Path(args.out) if args.out else repo_root / "experiments" / "surrogate_analysis.json"
    )
    run_analysis(sweep_dir, args.seeds, out_path, vif_threshold=args.vif_threshold)


if __name__ == "__main__":
    main()
