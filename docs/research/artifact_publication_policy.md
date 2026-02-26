# Data Policy (Zenodo)

This project uses Zenodo as the canonical archive for heavy experiment data.

## Publication Split (Mandatory)

For paper submissions, artifact publication is split across two channels:

1. **GitHub repository/PR** (lightweight, reviewable):
   - paper source and compiled PDF (`paper/`),
   - experiment scripts and analysis code (`scripts/`),
   - compact derived outputs tracked by git (see `.gitignore` allowlist):
     `coupling_analysis.json`, `final_graph_statistics.json`,
     `final_statistics.json`, `criterion_reduction_analysis.json`,
     `surrogate_analysis.json`,
   - run configs (`configs/`) and manifests (`experiments/*_manifest.json`),
   - manifest-binding registry
     (`docs/research/result_manifest_bindings.json`).
2. **Zenodo record** (heavy, immutable, citable):
   - raw per-seed experiment JSON (`experiments/ablation_sweep/`,
     `experiments/*_{condition}.json`),
   - ecology stress, invariance, midrun, and evolution raw outputs,
   - per-experiment TSV exports.

This split is required for reproducibility, reviewability, and long-term
archival.

## Scope

- Raw experiment outputs (per-seed JSON, TSV time-series, long-horizon runs)
  are archived on Zenodo.
- The Git repository keeps code, configs, and lightweight summaries only.

## What Must Not Be Committed to Git

- Raw per-seed JSON files under `experiments/` (gitignored by default).
- TSV exports and gzipped archives.
- Any single file or directory exceeding ~5 MB.

## What Should Be Kept in Git

- Reproducibility code and configs (`crates/`, `python/`, `scripts/`,
  `configs/`).
- Compact summary artifacts used by paper and analysis (allowlisted in
  `.gitignore`).
- Paper assets needed for review (`paper/main.tex`, figures, `paper/main.pdf`).
- Run manifests documenting exact parameters, seed lists, and config digests.
- References to archived datasets (Zenodo DOI, record URL, checksums).

## Zenodo Archival Requirements

For each major experiment release:

1. Upload the raw dataset bundle to Zenodo.
2. Record metadata:
   - title and version tag (matching Git tag/commit),
   - creation date,
   - generator command/config profile,
   - checksums (SHA256),
   - license (MIT, matching repository).
3. Capture the DOI and record URL in repository docs.

## Execution Runbook

### Prerequisites

- `ZENODO_TOKEN` must be available in your shell.
  Verify: `zsh -ic 'echo ${ZENODO_TOKEN:+present}'` should print `present`.
- If non-interactive shells cannot see the token, run publish commands via
  `zsh -ic '<command>'`.

### Step 1: Prepare artifacts

Compress experiment families into per-family archives:

```bash
mkdir -p zenodo_staging

# Ablation sweep (per-seed raw data)
tar -czf zenodo_staging/ablation_sweep.tar.gz experiments/ablation_sweep/

# Other heavy experiment families
for family in ecology_stress invariance midrun trait_evo; do
  tar -czf "zenodo_staging/${family}.tar.gz" experiments/${family}_*.json
done
```

### Step 2: Generate checksums

```bash
# Per-archive checksum
shasum -a 256 zenodo_staging/*.tar.gz > docs/research/zenodo_archive_sha256.txt

# Optional: per-file checksums (if reviewers request individual verification)
# find experiments/ -name '*.json' -size +1M | sort | \
#   while read -r f; do shasum -a 256 "$f"; done \
#   > docs/research/zenodo_perfile_sha256.txt
```

### Step 3: Generate metadata

```bash
uv run python scripts/prepare_zenodo_metadata.py \
  zenodo_staging/*.tar.gz \
  --experiment-name ablation_criterion_reduction \
  --steps 2000 \
  --seed-start 0 --seed-end 129 \
  --paper-binding tab:ablation=experiments/final_graph_statistics.json \
  --paper-binding tab:phase1=experiments/criterion_reduction_analysis.json \
  --zenodo-doi <RESERVED_DOI> \
  --output docs/research/zenodo_metadata.json
```

### Step 4: Upload to Zenodo

1. Go to <https://zenodo.org/uploads/new> (or use the Zenodo API with
   `ZENODO_TOKEN`).
2. Upload all `.tar.gz` files from `zenodo_staging/`.
3. Fill in metadata:
   - **Title**: `minimal_life: Experiment Data for ALIFE 2026 Submission`
   - **Version**: Git tag (e.g., `v1.0-submission`)
   - **Description**: Criterion-ablation, pairwise, evolution, ecology stress,
     invariance, and midrun experiment raw outputs.
   - **License**: MIT
   - **Related identifiers**: GitHub repository URL
4. Publish and record the DOI.

### Step 5: Update repository references

1. Update the DOI in `paper/main.tex` (data availability paragraph).
2. Add a `@misc` entry to `paper/references.bib`.
3. Commit `docs/research/zenodo_metadata.json` and
   `docs/research/zenodo_archive_sha256.txt`.

## Paper-Ready Release Checklist

Before calling a paper PR submission-ready:

1. GitHub PR contains the lightweight paper package:
   - [ ] Updated manuscript source and `paper/main.pdf`
   - [ ] Generated figures under `paper/figures/`
   - [ ] Compact analysis summaries used by paper text
   - [ ] Manifests and binding registry up to date
2. Zenodo bundle contains raw experimental evidence:
   - [ ] All `experiments/` raw outputs for reported results
   - [ ] Per-archive SHA256 checksums (per-file optional)
   - [ ] Metadata JSON with commit provenance
3. Cross-linking complete:
   - [ ] Zenodo DOI in `paper/main.tex` data availability section
   - [ ] Dataset entry in `paper/references.bib`
   - [ ] DOI consistent between paper text and repository docs

## Citation in the Paper

Cite the Zenodo dataset DOI in the paper's data availability section and add
a `@misc` entry to `paper/references.bib`:

```bibtex
@misc{minimal_life_data_2026,
  author    = {<authors>},
  title     = {minimal\_life: Experiment Data for ALIFE 2026},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {<DOI>},
  url       = {https://doi.org/<DOI>},
}
```

## Reproducibility Note

The repository allows users to:

- rerun experiments from configs:
  `uv run python scripts/experiment_final_graph.py` (see `STRUCTURE.md`
  Experiment Execution Order), or
- download archived raw data from Zenodo and reproduce figures/tables:
  `uv run python scripts/generate_figures.py`.
