# Pre-Registered Decision Rules — Phase 1 Criterion Reduction

**Committed before any analysis is run.** These thresholds are fixed and may not be
revised after inspecting the data. Revisions require a new pre-registration document
with a timestamp prior to any re-analysis.

**Date registered**: 2026-02-25
**Experiment**: Phase 1 Criterion-Reduction Analysis Pipeline
**Scripts**: `scripts/analyze_criterion_reduction.py`, `scripts/analyze_surrogates.py`
**Data source**: `experiments/ablation_sweep/` (30 conditions × 20 seeds = 600 runs)

---

## 1. Stability Selection Threshold

> **Rule**: A feature (criterion or observable) is classified as "stable" if and only if
> its selection frequency exceeds **0.60** across 500 bootstrap subsamples.

- Bootstrap subsample size: 50% of available observations (no replacement)
- Models: LASSO (primary) and Elastic Net (secondary)
- Agreement gate: both models must select the same feature set for it to enter the
  minimal sufficient set (see Rule 4)

## 2. Minimal Set Sufficiency

> **Rule**: A minimal criterion/observable set is declared "sufficient" if and only if
> **both** of the following hold simultaneously:
>
> (a) R² ≥ **0.85** on the held-out test conditions (evolution-involved pairwise pairs)
> (b) Cohen's d ≥ **0.50** vs the `all_off` baseline (alive_auc distribution)

## 3. LASSO / Elastic Net Agreement

> **Rule**: The reported minimal set is the intersection of features selected by LASSO
> and Elastic Net (both at stability frequency > 0.60). Features selected by only one
> method are reported separately as "candidate" features with their individual
> frequencies.

## 4. Performance Tolerance (Pareto Criterion)

> **Rule**: A k-criterion subset is declared "pareto-efficient" if the minimal-set
> alive_auc (averaged over 20 discovery seeds) is ≥ **85%** of the full 7-criterion
> system's alive_auc.

The Pareto curve is reported with **bootstrapped 95% CI bands** (500 re-samples).
The minimal set is the smallest k satisfying the ≥ 85% threshold.

## 5. Held-Out Test Set (Pre-Specified)

> **Rule**: The following 6 conditions are reserved as the held-out test set and are
> excluded from all model fitting (stability selection, LASSO, PCA):
>
> `drop_evolution_metabolism`, `drop_evolution_boundary`,
> `drop_evolution_homeostasis`, `drop_evolution_response`,
> `drop_evolution_reproduction`, `drop_evolution_growth`
>
> (Equivalently: any pairwise ablation that removes `evolution`.)
>
> These 6 conditions are selected **a priori** based on evolution being the criterion
> with the longest time-scale effect, not by inspecting the data.

## 6. Confirmatory Validation

> **Rule**: After identifying the minimal set from the 600-run discovery sweep
> (seeds 0–19), a **confirmatory experiment** is required before paper submission:
>
> - Seeds: **20–39** (new seeds not used during discovery)
> - Steps: **5000** (longer horizon than discovery runs)
> - Conditions: full system + minimal criterion set only
> - Pass criterion: minimal-set alive_auc ≥ 85% of full-system alive_auc on new seeds

Confirmatory results are stored in `experiments/confirmatory/`.

---

## Observable Surrogates (analyze_surrogates.py)

The same rules (1–5) apply. Additionally:

### VIF Collinearity Check

> **Rule**: Before fitting, compute Variance Inflation Factor (VIF) for all 14 features.
> Features with VIF > **10** are removed from the feature matrix before stability
> selection. Removed features are reported but not considered part of the minimal set.

### Cross-Validation Splits

> **Rule**: For condition-stratified CV splits in `analyze_surrogates.py`, splits are
> stratified by condition name (not random). This ensures each fold contains
> observations from all condition types.

---

*This document is version-controlled. Any post-hoc modification is detectable via
`git log docs/research/decision_rules.md`.*
