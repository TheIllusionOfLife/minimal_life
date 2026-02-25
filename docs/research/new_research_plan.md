Below is a **detailed research plan (Markdown)** for pursuing **Direction #3**: *surrogating the 7 functionalities with fewer functionalities (latent principles) while preserving the same life/viability metrics*, framed as an **ALIFE conference full paper** follow-up/extension to your current “7 criteria + ablation” system. 

---

# Research Plan: Minimal Diagnostic Principles of Life from a 7-Criteria Digital Life System

## 0. One-sentence goal 🎯

Derive a **small set of “minimal diagnostic principles”** (2–4) that can **predict life-likeness / viability / interdependence** as well as (or better than) the full 7-criteria checklist, using measurable signals that could plausibly generalize to **unknown substrates** (deep biosphere, astrobiology, artificial agents).

---

## 1. Motivation and positioning (why ALIFE should care) 🌍

### 1.1 Problem

Biology offers many life definitions; in practice, **life detection** depends on what you can measure. Your current work provides a strong operational foundation: **7 criteria implemented as dynamic processes + ablation tests** showing necessity and coupling. 

But for “life in unfamiliar places,” we often *cannot measure* the internal implementation of those 7 criteria. We need **diagnostics** that:

* are **observable** (from limited sensors),
* are **substrate-agnostic** (not tied to your specific mechanisms),
* still capture **what matters for persistence**.

### 1.2 Proposed leap

Turn your system into a **life-detection lab**:

* Generate diverse “organism/ecology variants” with known ground truth labels (under your operational definitions).
* Learn which **few** measurable signals best predict:

  1. **persistence / viability**,
  2. **interdependence / closure-like coupling**,
  3. **adaptation capacity** (optional).

This reframes the contribution from “we integrated 7 criteria” to:

> **“We infer a minimal set of diagnostic principles of life from a controlled digital ecosystem.”**

---

## 2. Core research questions (RQs) ❓

### **RQ1: Minimal set**

What is the **smallest set of measurable observables** that predicts life-likeness metrics **as well as** the full 7-criteria implementation?

### **RQ2: Robustness**

Do these minimal diagnostics **generalize** across:

* different environments (resource regimes / perturbations),
* different organism designs (e.g., different metabolism implementations you already have), 
* different “semi-life” variants (optional link to your idea #2 later)?

### **RQ3: Interpretability**

Can we map the diagnostics to **interpretable principles** (e.g., regulation strength, energy throughput, causal closure proxy), rather than a black-box classifier?

### **RQ4: Practical detection**

Given limited observation budgets (e.g., only population trajectories + local measurements), what diagnostics remain effective?

---

## 3. Definitions: what are we trying to predict? 🧩

You already have strong outcome measures and distinctions:

* Final alive count (N_T), AUC, lifespan, early-horizon (T=500) survival, spatial cohesion. 
* Individual-level vs population-level distinction (important). 

For #3, you need **targets** (labels) that represent “life-likeness” without circularity.

### 3.1 Proposed prediction targets (choose 2–3)

1. **Persistence score** 📈

   * AUC of alive count, plus survival under perturbation
2. **Regulated organization score** 🧠

   * measures of variance suppression / return-to-baseline after perturbation
3. **Interdependence score** 🔁

   * how much processes depend on each other (intervention-based or predictive coupling)
4. **Adaptation capacity score** 🧬 (optional for ALIFE full paper)

   * improvement over repeated stress cycles; trait–fitness association

**Recommendation:** For ALIFE scope, pick **(1) + (3)** as the “spine” and optionally add (2). Evolution/adaptation can be a “bonus” unless you already have strong evidence.

---

## 4. Candidate “observables” (what we measure to build surrogates) 🔎

The diagnostics must be:

* **measurable** from simulation logs (and later imaginable in real detection),
* **not identical** to internal toggles (“metabolism enabled”).

### 4.1 Observable families (high value)

1. **Energy / throughput proxies** ⚡

   * mean energy, energy variance, energy autocorrelation time
   * resource intake rate, waste production rate
2. **Regulation / homeostasis proxies** 🎛️

   * “return rate” after perturbation (time to recover internal variable ranges)
   * variance suppression: (\mathrm{Var}(x)) under stress vs baseline
3. **Boundary / organization proxies** 🧱

   * spatial cohesion, boundary integrity statistics, fragmentation events
4. **Behavioral coupling proxies** 🧭

   * movement efficiency: resource gradient ascent efficiency
   * responsiveness: mutual information between local resource signal and motion
5. **Reproduction / turnover proxies** 👶

   * birth/death rates, age distribution entropy, replacement rate
6. **Interdependence / closure proxies** 🔁

   * directed predictability (Granger-like) among observables (you already do this) 
   * intervention response: delta in other rates when one process is perturbed (you already have effect summaries) 

### 4.2 Observation constraints (for “life detection realism”)

Define 2–3 “sensor budgets”:

* **Budget A (rich)**: full internal logs (upper bound)
* **Budget B (field-like)**: population time series + spatial structure + resource field only
* **Budget C (minimal)**: population time series only

Your story becomes stronger if the minimal diagnostics still work under Budget B/C.

---

## 5. Experimental design: dataset generation 🧪

To infer minimal diagnostics, you need **variation**—otherwise the model just learns your single tuned regime.

### 5.1 Create a “Life Variant Suite” (LVS)

Generate many system variants by sampling:

1. **Environment regimes** 🌦️

   * resource regeneration rates
   * diffusion on/off (note: your current doc mentions “diffusing resource field” and later “no diffusion”; make this deliberate here) 
   * spatial patchiness, moving resource hotspots
   * periodic shocks (resource drought, toxin spikes, boundary stress)
2. **Organism mechanism variants** 🧬

   * your metabolism engines (Counter/Toy/Graph) are perfect built-in diversity 
   * adjust homeostasis decay rate / controller capacity
   * adjust boundary repair cost scaling
3. **Partial functionality variants** 🧩

   * controlled degradations (graded ablation is already in your mindset) 
   * mid-run ablation after stabilization (you already have scripts) 

**Recommendation:** For ALIFE, aim for something like:

* 10–20 environment regimes × 10–30 mechanism variants × n seeds
  That’s enough to discover robust latent structure without exploding compute.

### 5.2 Ground-truth labeling strategy (avoid circularity)

Instead of “life = criteria enabled,” define life-likeness as **performance under challenges**:

* persistence under perturbation suite
* interdependence score above threshold
* regulation score above threshold

Then later compare how these correlate with 7-criteria presence/strength.

---

## 6. Methods: finding the minimal surrogate set 🧠

### 6.1 Model families (interpretability-first)

1. **Sparse linear models** (LASSO / elastic net)

   * yields 2–6 observables
2. **Decision lists / shallow trees**

   * yields human-readable rules
3. **Factor analysis / PCA**

   * yields latent axes (“energy throughput”, “regulation strength”, “closure”)
4. **Symbolic regression** (optional, risky)

   * can produce elegant “physics-like” formulas, but can overfit

**Recommendation:** Lead with **sparse regression + stability selection**, then show a **factor interpretation**.

### 6.2 What counts as “surrogate success” ✅

A surrogate set (S) is successful if:

* It predicts target metrics with small error on **held-out regimes** (not just held-out seeds),
* It stays strong under **sensor budgets** B/C,
* It remains stable across metabolism implementations (Counter/Toy/Graph). 

### 6.3 Key statistical safeguards

* Split by **environment regime**, not only seeds (to test generalization)
* Report **calibration curves** (not only R²)
* Use **bootstrapped confidence intervals** on performance deltas
* Pre-register (internally) the target metrics and success criteria to avoid “metric shopping”

---

## 7. Proposed “minimal principles” candidates (hypotheses) 💡

You’ll likely find 2–4 latent principles. Here are good *theory-friendly* candidates to test:

1. **Sustained throughput under constraint** ⚡

   * not just “uses energy,” but maintains energy flux while paying maintenance costs
2. **Active regulation / return-to-manifold** 🎛️

   * measured as recovery time + variance suppression under shocks
3. **Organizational integrity** 🧱

   * persistence of bounded structure / cohesion under stress
4. **Causal interdependence / closure proxy** 🔁

   * if disabling/perturbing one observable collapses others, you have a closure-like signature

These map nicely to astrobiology constraints: you can’t see genes, but you might see fluxes, regulation, structural persistence, and causal coupling.

---

## 8. Deliverables for an ALIFE full paper 📦

### 8.1 Main paper contributions (3 bullets)

1. **A Life Detection Benchmark** derived from a 7-criteria digital ecosystem: diverse regimes + logged observables. 
2. **Minimal diagnostic set** (2–4 observables) that predicts life-likeness metrics across regimes and sensor budgets.
3. **Interpretation as principles** connecting to autonomy/closure and practical detection.

### 8.2 Figures that will “carry” the paper 📊

1. **Pareto curve**: performance vs number of observables (7 → 4 → 3 → 2)
2. **Generalization heatmap**: train-on-regimes vs test-on-regimes
3. **Sensor budget drop**: A vs B vs C performance
4. **Latent axes plot**: factor loadings + interpretation (“throughput”, “regulation”, “integrity”)
5. (Optional) **Interdependence graph** derived from interventions/predictability

### 8.3 Open artifacts (ALIFE loves this)

* a small **benchmark dataset** (run manifests + summary features)
* a **repro script** producing the minimal diagnostic set and figures

---

## 9. Risk register + how to avoid dead ends ⚠️

1. **Risk: surrogate learns “population size” only**
   ✅ Use multi-target prediction (persistence + interdependence) and perturbation suite.
2. **Risk: results are system-specific**
   ✅ Force generalization across metabolism implementations and regime splits. 
3. **Risk: diagnostics become uninterpretable**
   ✅ Prefer sparse + shallow models; require interpretability as a success constraint.
4. **Risk: circularity (“life = what survives”)**
   ✅ Define life-likeness as **multi-dimensional** (persistence + regulation + coupling), not survival alone.

---

## 10. Concrete “first run” plan (minimal viable research loop) 🔁

1. **Define 3 target scores** (Persistence, Regulation, Interdependence)
2. **Generate 12 regimes** (baseline + 11 variations)
3. **Generate 20 mechanism variants** (metabolism engine × homeostasis strength × boundary cost)
4. Run n seeds each; compute a feature table
5. Fit sparse models to predict targets; select minimal set via stability selection
6. Validate on held-out regimes; plot Pareto curves
7. Write “principles” interpretation section

---

# Questions for you (so I can update this document) ❓

1. **What venue track exactly?** ALIFE conference full paper (8 pages) vs short paper vs late-breaking?
2. What is your preferred **definition of “life-likeness target”** for #3?

   * A) persistence under perturbations
   * B) interdependence/closure proxy
   * C) regulation/homeostasis
   * D) combination (which weights?)
3. What observables can you log *reliably* right now without refactoring?

   * energy, waste, boundary integrity, internal state, movement, births/deaths, etc. 
4. Can you easily implement **environment regime variations** (diffusion on/off, patchiness, shocks), or is that costly?
5. Do you want the minimal diagnostics to work under a **field-like sensor budget** (no internal state), or is “digital-only” acceptable for ALIFE?

Reply with your answers (even short bullets), and I’ll revise this Markdown into a tighter **submission-ready research protocol**, including an experiment matrix and an outline you can paste into the paper.
