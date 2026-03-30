# ALIFE 2026 Full Paper Submission Guide

**Deadline**: April 12, 2026 (Anywhere on Earth)
**Submission system**: CMT (link in conference email)

---

## 1. Track

| Rank | Selection |
|------|-----------|
| 1 (primary) | Main Track |
| 2 (optional) | Artificial Life for Science and Engineering |

## 2. Title

Minimal Diagnostic Principles of Life: Criterion-Ablation and Observable Surrogate Selection in a Digital Ecosystem

## 3. Presentation Preference

- [x] Regular Oral

## 4. Keywords

### Biology Expertise
- [x] evolutionary biology
- [x] organism biology
- [x] ecology

### Mathematics / Computer Science Expertise
- [x] evolutionary algorithms (as a tool for understanding biology)
- [x] multi-agent systems
- [x] machine learning (mathematical knowledge needed)

### Simulation and In Silico Experiments
- [x] simulation of spatial / physical phenomena

### Other Expertise
- [x] open-ended evolution / novelty
- [x] theories of (self-)organisation
- [x] artificial chemistry / autocatalysis

## 5. Author Information

| Field | Value |
|-------|-------|
| First/Given Name | Yuya |
| Last/Family Name | Mukai |
| Email | *(fill in your email)* |
| Company/Institution/Org | Mukai Entertainment |
| Dept./Company/Institution | *(leave blank or repeat Mukai Entertainment)* |
| Country | Japan |

## 6. Student Paper

- [ ] No (not eligible for Best Student Paper Award)

## 7. Abstract

*(Copy from paper. 177 words, under 250-word limit.)*

All seven textbook biological criteria for life can be integrated as functionally interdependent processes within a single artificial life system. Here we implement cellular organization, metabolism, homeostasis, growth, reproduction, response to stimuli, and evolution (six at Level 4; evolution at Level 3) so that each criterion satisfies three conditions: sustained resource consumption, measurable degradation upon removal, and feedback coupling. We call this functional analogy.

Criterion-ablation experiments (n=30 per condition) show that disabling any single criterion causes statistically significant population decline (Holm-Bonferroni corrected, all p <= 0.004728; Cliff's delta 0.39-1.00). Reproduction, response, and metabolism produce the strongest effects (delta% of -91, -89, and -87, respectively). Pairwise ablations reveal sub-additive interactions consistent with shared failure pathways, and proxy controls provide evidence against tautological definitions.

A two-stage surrogate analysis then asks which observable measurements predict life-likeness without running the full ablation sweep. Phase 1 identifies six stable surrogates via Elastic Net stability selection. Phase 2 scales to 200 conditions across 5 environment regimes (14,000 runs), achieving out-of-regime test R^2 = 0.950 (permutation p < 0.001), which exceeds the single-predictor baseline of R^2 = 0.921.

## 8. Paper Upload

Upload `paper/main.pdf` (rebuild first with `cd paper && tectonic main.tex`).

## Notes

- Supplementary material (`supplementary.pdf`) should be uploaded separately if the system allows, or combined with the main PDF.
- The paper is 8 pages (not counting references and acknowledgements).
- Acknowledgements state: "This work received no external funding."
