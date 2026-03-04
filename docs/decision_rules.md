# Pre-Registration: Decision Rules for Surrogate Analysis

**Timestamp:** 2026-03-03 (created before Phase 2 analysis)

This document records the decision rules for surrogate variable selection
and honest reporting, specified *a priori* before examining Phase 2 results.

## 1. Stability Selection Threshold

- Method: LASSO + Elastic Net stability selection (500 bootstrap iterations)
- Inclusion criterion: stability frequency **≥ 0.60**
- Features below this threshold are excluded from the final surrogate set

## 2. R² Significance Threshold

- Method: Permutation test (1000 permutations) on training R²
- Significance threshold: permutation **p < 0.05**
- If p ≥ 0.05, the surrogate model is not considered significant

## 3. Baseline Contest Rule

- Surrogates must beat a **single-predictor baseline** using `alive_auc` only
- Comparison: Validation R² of full surrogate set vs. alive_auc-only predictor
- If surrogates do not exceed baseline, no claim of multi-surrogate benefit

## 4. Honest Fallback Triggers

| Condition | Action |
|-----------|--------|
| Validation R² < 0 (negative) | Surrogates overfit. Frame as rigorous lower bound. Primary contribution shifts to Stage 1 ablation results. |
| Validation R² ∈ [0, 0.3) | Low predictive power. Position Stage 2 as preliminary / hypothesis-generating. |
| Validation R² ≥ 0.3 | Adequate surrogate predictive power. Report normally. |
| Permutation p ≥ 0.05 | Surrogate model not significant. Report as null result. |
| Surrogates ≤ baseline | No multi-surrogate benefit. Report alive_auc as sufficient single predictor. |

## 5. Data Separation Protocol

### Phase 1 (Stage 1 ablation)

- **Calibration set:** seeds 0–99 (threshold tuning, training)
- **Final test set:** seeds 100–199 (evaluation with fixed thresholds)
- **Statistics:** Mann-Whitney U, Holm-Bonferroni correction (7 simultaneous tests), Cohen's d

### Phase 2 (Surrogate expansion — regime-based split)

- **Train set:** Regimes A–C, seeds 0–39 (seen regimes, train seeds)
- **Validation set:** Regimes A–C, seeds 40–69 (seen regimes, held-out seeds)
- **Test set:** Regimes D–E, seeds 0–69 (unseen regimes — true generalization)
- **Note:** The generalization axis is the environment regime, not the seed.
  Seeds 40–69 appear in both validation (regimes A–C) and test (regimes D–E),
  which is intentional: the test set evaluates regime-level generalization
  while maximizing statistical power within each regime.
