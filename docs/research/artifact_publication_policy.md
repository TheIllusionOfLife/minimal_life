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

- **Python**: `requests` must be installed (`uv sync` pulls it from
  `pyproject.toml`).
- **`ZENODO_TOKEN`**: personal access token with `deposit:write` and
  `deposit:actions` scopes.
  - Create at <https://zenodo.org/account/settings/applications/>.
  - Export in your shell profile (e.g., `~/.zshrc`):
    `export ZENODO_TOKEN="your_token_here"`
  - Verify: `echo ${ZENODO_TOKEN:+present}` should print `present`.

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

Upload artifacts using the `upload_zenodo.py` script, which calls the
[Zenodo REST API](https://developers.zenodo.org/) via Python `requests`.

**Dry run (draft only — recommended first):**

```bash
uv run python scripts/upload_zenodo.py \
  --metadata docs/research/zenodo_metadata.json \
  --creator "Last, First; Affiliation" \
  --version v1.0-submission \
  --github-url https://github.com/TheIllusionOfLife/minimal_life
```

This creates a draft deposit, uploads all files, and sets metadata — but does
**not** publish. You can review the draft in the Zenodo web UI before
committing.

**Publish (irreversible):**

```bash
uv run python scripts/upload_zenodo.py \
  --metadata docs/research/zenodo_metadata.json \
  --creator "Last, First; Affiliation" \
  --version v1.0-submission \
  --github-url https://github.com/TheIllusionOfLife/minimal_life \
  --publish
```

**Testing with sandbox:**

```bash
# Uses sandbox.zenodo.org (requires a separate ZENODO_TOKEN from sandbox)
uv run python scripts/upload_zenodo.py \
  --metadata docs/research/zenodo_metadata.json --sandbox
```

The script:
1. Reads artifact paths and SHA256 checksums from `zenodo_metadata.json`.
2. Verifies local files match the recorded checksums before uploading.
3. Creates a draft deposit via `POST /deposit/depositions`.
4. Uploads each archive via `PUT` to the bucket URL (new API, up to 50 GB).
5. Sets title, description, creators, license, keywords, conference info,
   and related identifiers.
6. Optionally publishes via `POST /deposit/depositions/{id}/actions/publish`.

After publishing, record the DOI printed to stderr.

**Creating a new version of an existing record:**

```bash
uv run python scripts/upload_zenodo.py \
  --metadata docs/research/zenodo_metadata.json \
  --new-version 18780935 \
  --creator "Last, First; Affiliation" \
  --version v2.0 --publish
```

This creates a version draft from the published record, removes inherited
files, uploads fresh artifacts, and publishes with a new DOI.

**Editing metadata of a published record (no re-upload):**

```bash
uv run python scripts/upload_zenodo.py \
  --edit 18780935 --title "Updated title" --publish
```

**Fetching BibTeX for a published record:**

```bash
uv run python scripts/upload_zenodo.py --fetch-bibtex 18780935
```

This prints the BibTeX entry to stdout, which can be appended to
`paper/references.bib`.

### Step 5: Update repository references

1. Update the DOI in `paper/main.tex` (data availability paragraph).
2. Add a `@misc` entry to `paper/references.bib`.
3. Re-run `prepare_zenodo_metadata.py` with the published `--zenodo-doi` to
   update `docs/research/zenodo_metadata.json`.
4. Commit `docs/research/zenodo_metadata.json` and
   `docs/research/zenodo_archive_sha256.txt`.

## Submission Sequence

The following steps must happen **in order**. The key constraint is that
Zenodo records and GitHub Releases are immutable once published, so all
review rounds must complete before archival.

```
1. Merge all paper/code PRs to main
2. AI peer review rounds (CodeRabbit, Codex, Gemini, etc.)
   └─ iterate: fix issues → push → re-review → until clean
3. Final "submission-ready" commit on main
4. Upload dataset to Zenodo (if not already done)
   └─ uv run python scripts/upload_zenodo.py ... --publish
5. Move tag to final commit:
   git tag -f v1.0-submission
   git push origin v1.0-submission --force
6. Create GitHub Release (triggers Zenodo code archival):
   gh release create v1.0-submission \
     --title "v1.0-submission: ALIFE 2026 Paper" \
     --notes "Initial submission to ALIFE 2026."
7. Manual: go to https://zenodo.org/account/settings/github/
   └─ verify the code record was created and has correct metadata
   └─ note the code DOI (separate from the dataset DOI)
8. Submit paper to ALIFE 2026 portal
```

**Why this order matters:**
- Steps 1–3 allow code to change freely during review.
- Step 4 can happen earlier if experiment data is stable (dataset DOI is
  independent of code changes).
- Steps 5–6 freeze the code; the tag must not move after the Release.
- Step 7 is a **manual verification** on the Zenodo website — the GitHub
  integration is automatic but the result should be checked (author names,
  related identifiers, license).

## Paper-Ready Checklist

Before creating the GitHub Release (step 6 above):

1. GitHub main branch contains the final paper package:
   - [ ] Updated manuscript source and `paper/main.pdf`
   - [ ] Generated figures under `paper/figures/`
   - [ ] Compact analysis summaries used by paper text
   - [ ] Manifests and binding registry up to date
   - [ ] `.zenodo.json` has real author names (not placeholders)
2. Zenodo dataset record is published:
   - [ ] All `experiments/` raw outputs for reported results
   - [ ] Per-archive SHA256 checksums (per-file optional)
   - [ ] Metadata JSON with commit provenance
3. Cross-linking complete:
   - [ ] Zenodo dataset DOI in `paper/main.tex` data availability section
   - [ ] Dataset entry in `paper/references.bib`
   - [ ] DOI consistent between paper text and repository docs
4. Manual Zenodo actions:
   - [ ] Repository toggled ON at
     <https://zenodo.org/account/settings/github/>
   - [ ] After Release: verify code record at Zenodo (author, license,
     related identifiers link to dataset DOI)

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

## GitHub–Zenodo Integration (Code Archival)

In addition to the dataset DOI (uploaded via `upload_zenodo.py`), the
repository is connected to Zenodo's GitHub integration. Creating a
**GitHub Release** triggers Zenodo to auto-archive a `.zip` of the source
code and mint a separate **code DOI**.

Setup:
1. Toggle the repository ON at
   <https://zenodo.org/account/settings/github/>.
2. Maintain `.zenodo.json` at the repository root — this controls the
   metadata (authors, keywords, license, related identifiers) that Zenodo
   uses instead of guessing from GitHub.
3. Create a GitHub Release from the submission tag:
   ```bash
   gh release create v1.0-submission \
     --title "v1.0-submission: ALIFE 2026 Paper" \
     --notes "Initial submission to ALIFE 2026."
   ```
4. Zenodo auto-creates a code record; add the code DOI to the README.

This gives the project **two citable DOIs**:
- **Dataset DOI** (`upload_zenodo.py`): raw experiment data
- **Code DOI** (GitHub integration): source code snapshot

## Reproducibility Note

The repository allows users to:

- rerun experiments from configs:
  `uv run python scripts/experiment_final_graph.py` (see `STRUCTURE.md`
  Experiment Execution Order), or
- download archived raw data from Zenodo and reproduce figures/tables:
  `uv run python scripts/generate_figures.py`.
