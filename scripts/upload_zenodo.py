"""Upload experiment artifacts to Zenodo via the REST API.

Reads artifact paths from zenodo_metadata.json (produced by
prepare_zenodo_metadata.py), creates a draft deposit, uploads each file,
sets metadata, and optionally publishes.

Usage:
    # Dry run (default) — create draft + upload, but do NOT publish:
    uv run python scripts/upload_zenodo.py \
        --metadata docs/research/zenodo_metadata.json

    # Publish immediately (irreversible):
    uv run python scripts/upload_zenodo.py \
        --metadata docs/research/zenodo_metadata.json --publish

    # Use sandbox for testing:
    uv run python scripts/upload_zenodo.py \
        --metadata docs/research/zenodo_metadata.json --sandbox

Environment:
    ZENODO_TOKEN — personal access token with deposit:write and
                   deposit:actions scopes.

See docs/research/artifact_publication_policy.md for the full workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

ZENODO_API = "https://zenodo.org/api"
SANDBOX_API = "https://sandbox.zenodo.org/api"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_response(resp: requests.Response, context: str) -> None:
    if resp.status_code >= 400:
        print(f"ERROR [{context}]: {resp.status_code}", file=sys.stderr)
        try:
            print(json.dumps(resp.json(), indent=2), file=sys.stderr)
        except ValueError:
            print(resp.text[:500], file=sys.stderr)
        sys.exit(1)


def create_deposit(base_url: str, token: str) -> dict:
    """Create an empty draft deposit and return the full response JSON."""
    resp = requests.post(
        f"{base_url}/deposit/depositions",
        json={},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    _check_response(resp, "create deposit")
    data = resp.json()
    print(f"Created deposit {data['id']}", file=sys.stderr)
    print(f"  Pre-reserved DOI: {data['metadata']['prereserve_doi']['doi']}", file=sys.stderr)
    return data


def upload_file(bucket_url: str, token: str, path: Path) -> dict:
    """Upload a single file via the bucket URL (new API, supports up to 50 GB)."""
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Uploading {path.name} ({size_mb:.1f} MB) ...", file=sys.stderr, end="", flush=True)
    with open(path, "rb") as fp:
        resp = requests.put(
            f"{bucket_url}/{path.name}",
            data=fp,
            headers={"Authorization": f"Bearer {token}"},
            timeout=600,
        )
    _check_response(resp, f"upload {path.name}")
    data = resp.json()
    print(f" OK (checksum: {data.get('checksum', 'n/a')})", file=sys.stderr)
    return data


def set_metadata(
    base_url: str,
    token: str,
    deposition_id: int,
    *,
    title: str,
    description: str,
    creators: list[dict[str, str]],
    version: str | None = None,
    related_identifiers: list[dict[str, str]] | None = None,
) -> dict:
    """Update the deposit metadata."""
    metadata: dict = {
        "title": title,
        "upload_type": "dataset",
        "description": description,
        "creators": creators,
        "license": "MIT",
        "access_right": "open",
    }
    if version:
        metadata["version"] = version
    if related_identifiers:
        metadata["related_identifiers"] = related_identifiers

    resp = requests.put(
        f"{base_url}/deposit/depositions/{deposition_id}",
        json={"metadata": metadata},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    _check_response(resp, "set metadata")
    print("  Metadata updated.", file=sys.stderr)
    return resp.json()


def publish(base_url: str, token: str, deposition_id: int) -> dict:
    """Publish the deposit (IRREVERSIBLE)."""
    resp = requests.post(
        f"{base_url}/deposit/depositions/{deposition_id}/actions/publish",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    _check_response(resp, "publish")
    data = resp.json()
    print(f"  Published! DOI: {data['doi']}", file=sys.stderr)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("docs/research/zenodo_metadata.json"),
        help="Path to zenodo_metadata.json (default: docs/research/zenodo_metadata.json).",
    )
    parser.add_argument(
        "--title",
        default="minimal_life: Experiment Data for ALIFE 2026 Submission",
        help="Zenodo record title.",
    )
    parser.add_argument(
        "--description",
        default=(
            "Criterion-ablation, pairwise, midrun, ecology stress, invariance, "
            "and evolution raw experiment outputs for the minimal_life ALIFE 2026 paper."
        ),
        help="Zenodo record description.",
    )
    parser.add_argument(
        "--creator",
        action="append",
        default=[],
        help="Creator in 'Last, First; Affiliation' format. Repeatable.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version tag (e.g., v1.0-submission).",
    )
    parser.add_argument(
        "--github-url",
        default=None,
        help="GitHub repository URL for related identifiers.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish after upload (IRREVERSIBLE). Without this flag, deposit stays as draft.",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use Zenodo sandbox (sandbox.zenodo.org) for testing.",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        default=True,
        help="Verify local SHA256 against metadata before uploading (default: True).",
    )
    return parser.parse_args()


def _parse_creator(raw: str) -> dict[str, str]:
    parts = raw.split(";", 1)
    name = parts[0].strip()
    if not name:
        raise ValueError(f"invalid creator: '{raw}'")
    entry: dict[str, str] = {"name": name}
    if len(parts) > 1:
        affiliation = parts[1].strip()
        if affiliation:
            entry["affiliation"] = affiliation
        return entry
    return entry


def main() -> int:
    args = parse_args()

    # --- Token ---
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print(
            "ERROR: ZENODO_TOKEN environment variable not set.\n"
            "Create a token at https://zenodo.org/account/settings/applications/\n"
            "with scopes: deposit:write, deposit:actions",
            file=sys.stderr,
        )
        return 1

    # --- Load metadata ---
    if not args.metadata.exists():
        print(f"ERROR: metadata file not found: {args.metadata}", file=sys.stderr)
        return 1
    with open(args.metadata) as f:
        meta = json.load(f)

    artifacts = meta.get("artifacts", [])
    if not artifacts:
        print("ERROR: no artifacts listed in metadata.", file=sys.stderr)
        return 1

    # --- Resolve and verify artifact paths ---
    artifact_paths: list[Path] = []
    for entry in artifacts:
        p = Path(entry["path"])
        if not p.exists():
            print(f"ERROR: artifact not found: {p}", file=sys.stderr)
            return 1
        if args.verify_checksums:
            local_hash = _sha256(p)
            expected = entry.get("sha256", "")
            if local_hash != expected:
                print(
                    f"ERROR: checksum mismatch for {p}\n"
                    f"  expected: {expected}\n"
                    f"  got:      {local_hash}",
                    file=sys.stderr,
                )
                return 1
        artifact_paths.append(p)

    base_url = SANDBOX_API if args.sandbox else ZENODO_API
    env_label = "SANDBOX" if args.sandbox else "PRODUCTION"
    print(f"Target: {env_label} ({base_url})", file=sys.stderr)

    # --- Step 1: Create draft deposit ---
    deposit = create_deposit(base_url, token)
    deposition_id = deposit["id"]
    bucket_url = deposit["links"]["bucket"]
    prereserved_doi = deposit["metadata"]["prereserve_doi"]["doi"]

    # --- Step 2: Upload files ---
    print(f"Uploading {len(artifact_paths)} file(s) ...", file=sys.stderr)
    for path in artifact_paths:
        upload_file(bucket_url, token, path)

    # --- Step 3: Set metadata ---
    if args.creator:
        creators = [_parse_creator(c) for c in args.creator]
    else:
        creators = [{"name": "<authors>"}]
    related: list[dict[str, str]] = []
    if args.github_url:
        related.append(
            {"identifier": args.github_url, "relation": "isSupplementTo", "scheme": "url"}
        )

    set_metadata(
        base_url,
        token,
        deposition_id,
        title=args.title,
        description=args.description,
        creators=creators,
        version=args.version,
        related_identifiers=related or None,
    )

    # --- Step 4: Publish (optional) ---
    if args.publish:
        print("Publishing (IRREVERSIBLE) ...", file=sys.stderr)
        result = publish(base_url, token, deposition_id)
        doi = result["doi"]
        print(f"\nDOI: {doi}", file=sys.stderr)
        print(f"URL: https://doi.org/{doi}", file=sys.stderr)
    else:
        print(
            f"\nDraft deposit created (NOT published).\n"
            f"  Deposit ID:      {deposition_id}\n"
            f"  Pre-reserved DOI: {prereserved_doi}\n"
            f"  Edit/publish at:  {base_url.replace('/api', '')}/deposit/{deposition_id}\n"
            f"\nTo publish, re-run with --publish or publish manually in the web UI.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
