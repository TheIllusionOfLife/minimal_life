#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PHASE2_PATTERN="experiments/phase2_*.json"
PHASE2_LOG="experiments/logs/phase2_rerun_2026-03-04.log"
ANALYSIS_LOG="experiments/logs/surrogate_v2_seedfix_2026-03-05.log"
PREP_LOG="experiments/logs/prepare_zenodo_phase2_seedfix_2026-03-05.log"
UPLOAD_LOG="experiments/logs/zenodo_phase2_seedfix_2026-03-05.log"
TARBALL="zenodo_staging/phase2_seedfix_raw_json_2026-03-05.tar.gz"
META_OUT="docs/research/zenodo_phase2_seedfix_metadata.json"

mkdir -p experiments/logs zenodo_staging

echo "[post-merge] Waiting for phase2 sweep to finish..."
while pgrep -f "scripts/experiment_phase2_sweep.py" >/dev/null; do
  sleep 120
done

echo "[post-merge] Validating Phase 2 artifacts..."
count=$(ls ${PHASE2_PATTERN} 2>/dev/null | wc -l | tr -d ' ')
if [[ "${count}" -ne 200 ]]; then
  echo "ERROR: expected 200 Phase 2 JSON files, found ${count}" >&2
  exit 1
fi

uv run python - <<'PY'
import glob
import json
import sys

files = sorted(glob.glob("experiments/phase2_*.json"))
for path in files:
    data = json.load(open(path))
    if not data:
        print(f"ERROR: empty data in {path}", file=sys.stderr)
        sys.exit(1)
    if not all("seed" in row for row in data):
        print(f"ERROR: missing seed field in {path}", file=sys.stderr)
        sys.exit(1)
print(f"OK: validated seed field in {len(files)} files")
PY

echo "[post-merge] Running surrogate analysis..."
uv run python scripts/analyze_surrogates_v2.py >"${ANALYSIS_LOG}" 2>&1

if [[ ! -f experiments/surrogate_analysis_v2.json ]]; then
  echo "ERROR: surrogate_analysis_v2.json was not generated" >&2
  exit 1
fi

echo "[post-merge] Bundling artifacts..."
tar -czf "${TARBALL}" experiments/phase2_*.json
cp experiments/surrogate_analysis_v2.json zenodo_staging/surrogate_analysis_v2.json

echo "[post-merge] Preparing Zenodo metadata..."
uv run python scripts/prepare_zenodo_metadata.py \
  "${TARBALL}" \
  zenodo_staging/surrogate_analysis_v2.json \
  --experiment-name phase2_surrogate_seedfixed \
  --steps 2000 \
  --seed-start 0 \
  --seed-end 69 \
  --entrypoint "uv run python scripts/experiment_phase2_sweep.py > experiments/phase2_data.tsv" \
  --paper-binding "tab:phase2=experiments/surrogate_analysis_v2.json" \
  --output "${META_OUT}" >"${PREP_LOG}" 2>&1

if [[ -z "${ZENODO_TOKEN:-}" ]]; then
  echo "[post-merge] ZENODO_TOKEN is not set; skipping upload." | tee "${UPLOAD_LOG}"
  exit 0
fi

echo "[post-merge] Uploading to Zenodo draft..."
uv run python scripts/upload_zenodo.py --metadata "${META_OUT}" >"${UPLOAD_LOG}" 2>&1

echo "[post-merge] Completed. See logs:"
echo "  ${PHASE2_LOG}"
echo "  ${ANALYSIS_LOG}"
echo "  ${PREP_LOG}"
echo "  ${UPLOAD_LOG}"
