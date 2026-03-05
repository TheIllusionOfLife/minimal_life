#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-43200}" # 12h default timeout
SLEEP_SEC="${SLEEP_SEC:-120}"
SURROGATE_JSON="experiments/surrogate_analysis_v2.json"
SURROGATE_STAGE="zenodo_staging/surrogate_analysis_v2.json"
META_OUT="${META_OUT:-docs/research/zenodo_phase2_seedfix_metadata.json}"

PHASE2_LOG_DEFAULT="$(ls -1t experiments/logs/phase2_rerun_*.log 2>/dev/null | head -n 1 || true)"
PHASE2_LOG="${PHASE2_LOG:-${PHASE2_LOG_DEFAULT:-experiments/logs/phase2_rerun_${RUN_ID}.log}}"
ANALYSIS_LOG="${ANALYSIS_LOG:-experiments/logs/surrogate_v2_seedfix_${RUN_ID}.log}"
PREP_LOG="${PREP_LOG:-experiments/logs/prepare_zenodo_phase2_seedfix_${RUN_ID}.log}"
UPLOAD_LOG="${UPLOAD_LOG:-experiments/logs/zenodo_phase2_seedfix_${RUN_ID}.log}"
TARBALL="${TARBALL:-zenodo_staging/phase2_seedfix_raw_json_${RUN_ID}.tar.gz}"

mkdir -p experiments/logs zenodo_staging

echo "[post-merge] Waiting for phase2 sweep to finish..."
waited=0
while pgrep -f "scripts/experiment_phase2_sweep.py" >/dev/null; do
  if (( waited >= MAX_WAIT_SEC )); then
    echo "ERROR: phase2 sweep did not finish within ${MAX_WAIT_SEC}s" >&2
    exit 1
  fi
  sleep "${SLEEP_SEC}"
  waited=$((waited + SLEEP_SEC))
done

echo "[post-merge] Validating Phase 2 artifacts..."
shopt -s nullglob
phase2_files=(experiments/phase2_*.json)
shopt -u nullglob
count="${#phase2_files[@]}"
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
    with open(path) as f:
        data = json.load(f)
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

if [[ ! -f "${SURROGATE_JSON}" ]]; then
  echo "ERROR: surrogate_analysis_v2.json was not generated" >&2
  exit 1
fi

echo "[post-merge] Bundling artifacts..."
tar -czf "${TARBALL}" "${phase2_files[@]}"
cp "${SURROGATE_JSON}" "${SURROGATE_STAGE}"

echo "[post-merge] Preparing Zenodo metadata..."
uv run python scripts/prepare_zenodo_metadata.py \
  "${TARBALL}" \
  "${SURROGATE_STAGE}" \
  --experiment-name phase2_surrogate_seedfixed \
  --steps 2000 \
  --seed-start 0 \
  --seed-end 69 \
  --entrypoint "uv run python scripts/experiment_phase2_sweep.py > experiments/phase2_data.tsv" \
  --paper-binding "tab:phase2=experiments/surrogate_analysis_v2.json" \
  --output "${META_OUT}" >"${PREP_LOG}" 2>&1

if [[ -z "${ZENODO_TOKEN:-}" ]]; then
  echo "[post-merge] ZENODO_TOKEN is not set; skipping upload." | tee "${UPLOAD_LOG}"
else
  echo "[post-merge] Uploading to Zenodo draft..."
  uv run python scripts/upload_zenodo.py --metadata "${META_OUT}" >"${UPLOAD_LOG}" 2>&1
fi

echo "[post-merge] Completed. See logs:"
echo "  ${PHASE2_LOG}"
echo "  ${ANALYSIS_LOG}"
echo "  ${PREP_LOG}"
echo "  ${UPLOAD_LOG}"
