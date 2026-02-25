# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Minimal Life** is an artificial life (ALife) research project investigating the **minimal diagnostic principles of life**: finding 2–4 surrogate observables that predict life-likeness as accurately as the full seven-criteria system does.

**Target venue**: ALIFE 2026 Full Paper (8p), deadline ~April 1, 2026.

**Stance**: Weak ALife — the system is a functional model of life, not a claim of life itself.

## Document Structure

| Document | Role |
|----------|------|
| `docs/research/new_research_plan.md` | **Authoritative plan**: research goals, methodology, surrogate selection strategy |

Old research artifacts are archived under `docs/archive/research/`.

## Architecture Decisions

- **Hybrid two-layer**: Swarm agents (10-50 per organism) form organism-level structures; organisms (10-50) inhabit a continuous 2D environment
- **Language**: Rust (core simulation) + Python (experiment management, analysis). Bound via PyO3/maturin
- **LaTeX**: Use `tectonic` for paper compilation (not pdflatex/latexmk)
- **Neural controllers**: Evolutionary NN (main). LLM (Ollama) only for a single ablation study experiment
- **Compute**: Mac Mini M2 Pro. Target: >100 timesteps/sec for 2,500 agents
- **Metabolism**: Graph-based metabolic networks, genetically encoded and evolvable
- **Genotype**: Variable-length encoding covering metabolic network + developmental program + NN architecture

## Research Goal: Minimal Diagnostic Principles

The central question: among the seven biological criteria, which 2–4 observables form a minimal sufficient set that predicts life-likeness?

**Approach**:
- Run the full seven-criteria simulation
- Measure surrogate observables (e.g., boundary coherence, metabolic flux, homeostatic range)
- Use criterion-ablation experiments to quantify individual criterion contributions
- Identify the minimal predictor set via correlation/regression analysis

**Data separation protocol**:
- Calibration set: seeds 0-99 (threshold tuning)
- Final test set: seeds 100-199 (evaluation with fixed thresholds)
- Statistics: Mann-Whitney U, Holm-Bonferroni correction (7 simultaneous tests), Cohen's d

## Pivot Strategy

| Trigger | Pivot |
|---------|-------|
| Metabolic network unsustainable | Graph-based → ODE-based metabolism |
| Hybrid two-layer unstable | Drop swarm, simplify to agent-based |
| 7-criteria integration infeasible by deadline | Narrow paper to 3-5 working criteria |
| Full paper infeasible by Week 4 | Switch to Extended Abstract (2-4p) |

## Language Notes

Research documents may be bilingual (Japanese + English). When generating research content, match the language of the target document.
