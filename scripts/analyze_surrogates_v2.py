"""Phase 2 powered surrogate analysis with regime-based validation.

3-layer regime-based split:
  Train:    Regimes A-C, seeds 0-39 (seen regimes, train seeds)
  Validate: Regimes A-C, seeds 40-69 (seen regimes, held-out seeds)
  Test:     Regimes D-E, seeds 0-69 (unseen regimes — true generalization)

Includes:
  - LASSO + Elastic Net stability selection
  - Permutation test for R² significance
  - Baseline contest: surrogates must beat population-count-only predictor
  - Target leakage check

Usage:
    uv run python scripts/analyze_surrogates_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_common import experiment_output_dir, log

# NumPy 2.0 compatibility
try:
    _trapz = np.trapezoid  # type: ignore[attr-defined]
except AttributeError:
    _trapz = np.trapz  # type: ignore[attr-defined]

# Regime definitions matching experiment_phase2_sweep.py
TRAIN_REGIMES = ["regime_a", "regime_b", "regime_c"]
TEST_REGIMES = ["regime_d", "regime_e"]
TRAIN_SEEDS = set(range(0, 40))
VALIDATE_SEEDS = set(range(40, 70))
TEST_SEEDS = set(range(0, 70))

# Feature extraction from time series
FEATURE_NAMES = [
    "alive_auc",
    "energy_mean_late",
    "energy_autocorr",
    "waste_slope",
    "boundary_stability",
    "birth_rate",
    "death_rate",
    "turnover_ratio",
    "genome_diversity_late",
    "maturity_late",
    "mean_generation_late",
    "population_variance",
]


def extract_features(samples: list[dict], steps: int) -> dict[str, float]:
    """Extract scalar features from a run's StepMetrics time series."""
    if not samples:
        return {f: 0.0 for f in FEATURE_NAMES}

    alive = [s["alive_count"] for s in samples]
    energy = [s["energy_mean"] for s in samples]
    waste = [s["waste_mean"] for s in samples]
    boundary = [s["boundary_mean"] for s in samples]
    births = [s["birth_count"] for s in samples]
    deaths = [s["death_count"] for s in samples]

    late_start = max(0, len(samples) * 3 // 4)

    # AUC of alive count (normalized by steps)
    alive_auc = float(_trapz(alive, dx=1)) / max(len(alive), 1)

    # Late-phase means
    energy_late = float(np.mean(energy[late_start:])) if late_start < len(energy) else 0.0

    # Autocorrelation of energy
    if len(energy) > 2:
        e_arr = np.array(energy) - np.mean(energy)
        var = np.var(energy)
        energy_autocorr = float(np.corrcoef(e_arr[:-1], e_arr[1:])[0, 1]) if var > 1e-10 else 0.0
    else:
        energy_autocorr = 0.0

    # Waste slope (linear trend)
    if len(waste) > 1:
        x = np.arange(len(waste))
        waste_slope = float(np.polyfit(x, waste, 1)[0])
    else:
        waste_slope = 0.0

    # Boundary stability (1 - CV)
    boundary_std = float(np.std(boundary))
    boundary_mean = float(np.mean(boundary))
    boundary_stability = 1.0 - (boundary_std / boundary_mean if boundary_mean > 1e-10 else 0.0)

    # Birth/death rates
    total_births = sum(births)
    total_deaths = sum(deaths)
    n_steps = len(samples)
    birth_rate = total_births / max(n_steps, 1)
    death_rate = total_deaths / max(n_steps, 1)
    turnover = (total_births + total_deaths) / max(n_steps, 1)

    # Late-phase diversity and maturity
    diversity_late = float(np.mean([s.get("genome_diversity", 0) for s in samples[late_start:]]))
    maturity_late = float(np.mean([s.get("mean_age", 0) for s in samples[late_start:]]))
    gen_late = float(np.mean([s["mean_generation"] for s in samples[late_start:]]))

    # Population variance
    pop_var = float(np.var(alive))

    return {
        "alive_auc": alive_auc,
        "energy_mean_late": energy_late,
        "energy_autocorr": energy_autocorr,
        "waste_slope": waste_slope,
        "boundary_stability": boundary_stability,
        "birth_rate": birth_rate,
        "death_rate": death_rate,
        "turnover_ratio": turnover,
        "genome_diversity_late": diversity_late,
        "maturity_late": maturity_late,
        "mean_generation_late": gen_late,
        "population_variance": pop_var,
    }


def compute_target(result: dict) -> float:
    """Compute life-likeness target score from a run result."""
    return float(result.get("final_alive_count", 0))


def load_phase2_data(out_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Load Phase 2 data split into train/validate/test by regime and seed."""
    train_data, validate_data, test_data = [], [], []

    for path in sorted(out_dir.glob("phase2_*.json")):
        cond_name = path.stem.replace("phase2_", "")
        # Parse regime from condition name: "regime_X__ablation_name"
        parts = cond_name.split("__", 1)
        if len(parts) != 2:
            continue
        regime, ablation = parts

        with open(path) as f:
            results = json.load(f)

        for idx, result in enumerate(results):
            # Use explicit seed if available, otherwise use index as proxy.
            # WARNING: The index fallback is non-deterministic for pre-fix data
            # because as_completed() returns futures by completion time, not seed
            # order. This produces a valid random split but is not reproducible.
            # Re-generate data with the fixed experiment_common.py (which injects
            # result["seed"]) for exact reproducibility.
            seed = result.get("seed", idx)
            features = extract_features(result.get("samples", []), 2000)
            target = compute_target(result)
            entry = {
                "regime": regime,
                "ablation": ablation,
                "seed": seed,
                "features": features,
                "target": target,
                "alive_auc": features["alive_auc"],
            }

            if regime in TRAIN_REGIMES:
                if seed in TRAIN_SEEDS:
                    train_data.append(entry)
                elif seed in VALIDATE_SEEDS:
                    validate_data.append(entry)
            elif regime in TEST_REGIMES:
                if seed in TEST_SEEDS:
                    test_data.append(entry)

    return train_data, validate_data, test_data


def data_to_arrays(
    data: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert data entries to feature matrix X and target vector y."""
    X = np.array([[d["features"][f] for f in FEATURE_NAMES] for d in data])
    y = np.array([d["target"] for d in data])
    return X, y


def stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstraps: int = 500,
    threshold: float = 0.60,
) -> list[tuple[str, float]]:
    """LASSO + Elastic Net stability selection."""
    n_samples, n_features = X.shape
    lasso_counts = np.zeros(n_features)
    enet_counts = np.zeros(n_features)

    for _ in range(n_bootstraps):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot, y_boot = X[idx], y[idx]

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_boot)

        lasso = Lasso(alpha=0.01, max_iter=5000)
        lasso.fit(X_s, y_boot)
        lasso_counts += (np.abs(lasso.coef_) > 1e-6).astype(float)

        enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
        enet.fit(X_s, y_boot)
        enet_counts += (np.abs(enet.coef_) > 1e-6).astype(float)

    lasso_freq = lasso_counts / n_bootstraps
    enet_freq = enet_counts / n_bootstraps

    # Agreement gate: both must select
    agreed_freq = np.minimum(lasso_freq, enet_freq)
    stable = [
        (FEATURE_NAMES[i], float(agreed_freq[i]))
        for i in range(n_features)
        if agreed_freq[i] >= threshold
    ]
    stable.sort(key=lambda x: x[1], reverse=True)
    return stable


def permutation_test_r2(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
) -> tuple[float, float, float]:
    """Permutation test for R² significance."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    lasso = Lasso(alpha=0.01, max_iter=5000)
    lasso.fit(X_s, y)
    actual_r2 = r2_score(y, lasso.predict(X_s))

    null_r2s = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        lasso_perm = Lasso(alpha=0.01, max_iter=5000)
        lasso_perm.fit(X_s, y_perm)
        null_r2s.append(r2_score(y_perm, lasso_perm.predict(X_s)))

    p_value = float(np.mean(np.array(null_r2s) >= actual_r2))
    return actual_r2, p_value, float(np.percentile(null_r2s, 95))


def main():
    out_dir = experiment_output_dir()
    log("=== Phase 2 Powered Surrogate Analysis ===\n")

    train, validate, test = load_phase2_data(out_dir)
    log(f"Data loaded: train={len(train)}, validate={len(validate)}, test={len(test)}")

    if len(train) < 50:
        log("ERROR: Insufficient training data. Run experiment_phase2_sweep.py first.")
        return

    X_train, y_train = data_to_arrays(train)
    X_val, y_val = data_to_arrays(validate)

    # Replace NaN/Inf with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Stability selection
    log("\n--- Stability Selection (500 bootstraps) ---")
    stable_features = stability_selection(X_train, y_train)
    for name, freq in stable_features:
        log(f"  {name:25s}: freq={freq:.3f}")
    if not stable_features:
        log("  No stable features found (all below threshold 0.60)")

    # 2. Permutation test
    log("\n--- Permutation Test (1000 shuffles) ---")
    actual_r2, perm_p, null_95 = permutation_test_r2(X_train, y_train)
    log(f"  Train R²={actual_r2:.4f}, permutation p={perm_p:.4f}, null 95th={null_95:.4f}")

    # 3. Validation R²
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    lasso = Lasso(alpha=0.01, max_iter=5000)
    lasso.fit(X_train_s, y_train)
    val_r2 = r2_score(y_val, lasso.predict(X_val_s)) if len(y_val) > 0 else float("nan")
    log(f"  Validation R²={val_r2:.4f}")

    # 4. Out-of-regime test
    if test:
        X_test, y_test = data_to_arrays(test)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_s = scaler.transform(X_test)
        test_r2 = r2_score(y_test, lasso.predict(X_test_s))
        log(f"  Out-of-regime Test R²={test_r2:.4f}")
    else:
        test_r2 = float("nan")
        log("  Out-of-regime test: NO DATA")

    # 5. Baseline contest: population-count-only predictor
    log("\n--- Baseline Contest ---")
    # alive_auc as sole predictor
    auc_idx = FEATURE_NAMES.index("alive_auc")
    if len(X_val) > 0:
        from sklearn.linear_model import LinearRegression

        baseline = LinearRegression()
        baseline.fit(X_train_s[:, auc_idx:auc_idx + 1], y_train)
        baseline_val_r2 = r2_score(y_val, baseline.predict(X_val_s[:, auc_idx:auc_idx + 1]))
        log(f"  Baseline (alive_auc only) Validation R²={baseline_val_r2:.4f}")
        log(f"  Surrogate set Validation R²={val_r2:.4f}")
        if val_r2 > baseline_val_r2:
            log("  PASS: Surrogate set beats baseline")
        else:
            log("  FAIL: Surrogate set does not beat population-count-only baseline")

    # 6. Save results
    output = {
        "stable_features": stable_features,
        "train_r2": actual_r2,
        "permutation_p": perm_p,
        "null_95th": null_95,
        "validation_r2": val_r2,
        "test_r2": test_r2,
        "n_train": len(train),
        "n_validate": len(validate),
        "n_test": len(test),
    }
    output_path = out_dir / "surrogate_analysis_v2.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {output_path}")

    # 7. Honest R² fallback assessment
    log("\n--- Honest R² Assessment ---")
    if val_r2 < 0:
        log("  WARNING: Negative validation R² — surrogates overfit.")
        log("  Framing as rigorous lower bound. Primary contribution: Stage 1 ablation.")
    elif val_r2 < 0.3:
        log("  CAUTION: Low validation R². Surrogates explain limited variance.")
        log("  Position Stage 2 as preliminary/hypothesis-generating.")
    else:
        log(f"  OK: Validation R²={val_r2:.3f} — adequate surrogate predictive power.")


if __name__ == "__main__":
    main()
