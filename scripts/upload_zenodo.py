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

    # Create a new version of an existing published record:
    uv run python scripts/upload_zenodo.py \
        --metadata docs/research/zenodo_metadata.json \
        --new-version 18780935

    # Edit metadata of an existing published record (no re-upload):
    uv run python scripts/upload_zenodo.py \
        --edit 18780935 --title "Updated title" --publish

    # Fetch BibTeX for a published record:
    uv run python scripts/upload_zenodo.py --fetch-bibtex 18780935

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
            detail = resp.json()
            print(json.dumps(detail, indent=2), file=sys.stderr)
        except ValueError:
            print(resp.text[:500], file=sys.stderr)
        sys.exit(1)


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _json_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _parse_creator(raw: str) -> dict[str, str]:
    """Parse 'Last, First; Affiliation; 0000-0000-0000-0000' into a creator dict."""
    parts = [p.strip() for p in raw.split(";")]
    name = parts[0]
    if not name:
        raise ValueError(f"invalid creator: '{raw}'")
    entry: dict[str, str] = {"name": name}
    if len(parts) > 1 and parts[1]:
        entry["affiliation"] = parts[1]
    if len(parts) > 2 and parts[2]:
        entry["orcid"] = parts[2]
    return entry


# ---------------------------------------------------------------------------
# API operations
# ---------------------------------------------------------------------------


def create_deposit(base_url: str, token: str) -> dict:
    """Create an empty draft deposit and return the full response JSON."""
    resp = requests.post(
        f"{base_url}/deposit/depositions",
        json={},
        headers=_json_headers(token),
        timeout=30,
    )
    _check_response(resp, "create deposit")
    data = resp.json()
    print(f"Created deposit {data['id']}", file=sys.stderr)
    doi = data["metadata"]["prereserve_doi"]["doi"]
    print(f"  Pre-reserved DOI: {doi}", file=sys.stderr)
    return data


def create_new_version(base_url: str, token: str, record_id: int) -> dict:
    """Create a new version draft from a published record.

    Returns the new draft (not the original record).
    """
    resp = requests.post(
        f"{base_url}/deposit/depositions/{record_id}/actions/newversion",
        headers=_auth_headers(token),
        timeout=30,
    )
    _check_response(resp, "new version")
    latest_draft_url = resp.json()["links"]["latest_draft"]
    resp2 = requests.get(
        latest_draft_url,
        headers=_auth_headers(token),
        timeout=15,
    )
    _check_response(resp2, "fetch new version draft")
    draft = resp2.json()
    print(f"Created new version draft {draft['id']}", file=sys.stderr)
    print(
        f"  DOI: {draft['metadata']['prereserve_doi']['doi']}",
        file=sys.stderr,
    )
    return draft


def edit_published(base_url: str, token: str, record_id: int) -> dict:
    """Unlock a published record for metadata edits."""
    resp = requests.post(
        f"{base_url}/deposit/depositions/{record_id}/actions/edit",
        headers=_auth_headers(token),
        timeout=30,
    )
    _check_response(resp, "edit published")
    data = resp.json()
    print(f"Unlocked deposit {record_id} for editing.", file=sys.stderr)
    return data


def discard_edits(base_url: str, token: str, record_id: int) -> dict:
    """Discard unpublished edits on a record."""
    resp = requests.post(
        f"{base_url}/deposit/depositions/{record_id}/actions/discard",
        headers=_auth_headers(token),
        timeout=30,
    )
    _check_response(resp, "discard")
    return resp.json()


def upload_file(bucket_url: str, token: str, path: Path) -> dict:
    """Upload a single file via the bucket URL (new API, up to 50 GB total)."""
    size_mb = path.stat().st_size / (1024 * 1024)
    print(
        f"  Uploading {path.name} ({size_mb:.1f} MB) ...",
        file=sys.stderr,
        end="",
        flush=True,
    )
    with open(path, "rb") as fp:
        resp = requests.put(
            f"{bucket_url}/{path.name}",
            data=fp,
            headers=_auth_headers(token),
            timeout=600,
        )
    _check_response(resp, f"upload {path.name}")
    data = resp.json()
    print(f" OK (checksum: {data.get('checksum', 'n/a')})", file=sys.stderr)
    return data


def delete_file(base_url: str, token: str, dep_id: int, file_id: str) -> None:
    """Delete a file from a draft deposit (legacy API)."""
    resp = requests.delete(
        f"{base_url}/deposit/depositions/{dep_id}/files/{file_id}",
        headers=_auth_headers(token),
        timeout=15,
    )
    _check_response(resp, f"delete file {file_id}")


def set_metadata(
    base_url: str,
    token: str,
    deposition_id: int,
    *,
    title: str,
    description: str,
    creators: list[dict[str, str]],
    version: str | None = None,
    keywords: list[str] | None = None,
    related_identifiers: list[dict[str, str]] | None = None,
    conference_title: str | None = None,
    conference_url: str | None = None,
    language: str | None = None,
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
    if keywords:
        metadata["keywords"] = keywords
    if related_identifiers:
        metadata["related_identifiers"] = related_identifiers
    if conference_title:
        metadata["conference_title"] = conference_title
    if conference_url:
        metadata["conference_url"] = conference_url
    if language:
        metadata["language"] = language

    resp = requests.put(
        f"{base_url}/deposit/depositions/{deposition_id}",
        json={"metadata": metadata},
        headers=_json_headers(token),
        timeout=30,
    )
    _check_response(resp, "set metadata")
    print("  Metadata updated.", file=sys.stderr)
    return resp.json()


def publish_deposit(base_url: str, token: str, deposition_id: int) -> dict:
    """Publish the deposit (IRREVERSIBLE)."""
    resp = requests.post(
        f"{base_url}/deposit/depositions/{deposition_id}/actions/publish",
        headers=_auth_headers(token),
        timeout=30,
    )
    _check_response(resp, "publish")
    data = resp.json()
    print(f"  Published! DOI: {data['doi']}", file=sys.stderr)
    return data


def fetch_bibtex(base_url: str, record_id: int) -> str:
    """Fetch BibTeX citation for a published record (no auth required)."""
    resp = requests.get(
        f"{base_url}/records/{record_id}",
        headers={"Accept": "application/x-bibtex"},
        timeout=15,
    )
    _check_response(resp, "fetch bibtex")
    return resp.text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selectors (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--new-version",
        type=int,
        metavar="RECORD_ID",
        help="Create a new version of an existing published record.",
    )
    mode.add_argument(
        "--edit",
        type=int,
        metavar="RECORD_ID",
        help="Edit metadata of an existing published record (no file upload).",
    )
    mode.add_argument(
        "--fetch-bibtex",
        type=int,
        metavar="RECORD_ID",
        help="Fetch and print BibTeX for a published record, then exit.",
    )

    # Artifact source
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("docs/research/zenodo_metadata.json"),
        help="Path to zenodo_metadata.json.",
    )

    # Metadata fields
    parser.add_argument("--title", default=None, help="Zenodo record title.")
    parser.add_argument("--description", default=None, help="Record description.")
    parser.add_argument(
        "--creator",
        action="append",
        default=[],
        help="Creator in 'Last, First; Affiliation; ORCID' format. Repeatable.",
    )
    parser.add_argument("--version", default=None, help="Version tag.")
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Keyword tag. Repeatable.",
    )
    parser.add_argument("--github-url", default=None, help="GitHub repo URL.")
    parser.add_argument(
        "--conference-title",
        default=None,
        help="Conference name (e.g., 'ALIFE 2026').",
    )
    parser.add_argument(
        "--conference-url",
        default=None,
        help="Conference website URL.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="ISO 639-2/3 language code (e.g., 'eng').",
    )

    # Actions
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish after upload (IRREVERSIBLE).",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use Zenodo sandbox for testing.",
    )
    parser.add_argument(
        "--no-verify-checksums",
        action="store_true",
        help="Skip SHA256 verification before upload.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main workflows
# ---------------------------------------------------------------------------

DEFAULT_TITLE = "minimal_life: Experiment Data for ALIFE 2026 Submission"
DEFAULT_DESCRIPTION = (
    "Criterion-ablation, pairwise, midrun, ecology stress, invariance, "
    "and evolution raw experiment outputs for the minimal_life ALIFE 2026 paper."
)


def _build_related(args: argparse.Namespace) -> list[dict[str, str]] | None:
    related: list[dict[str, str]] = []
    if args.github_url:
        related.append({
            "identifier": args.github_url,
            "relation": "isSupplementTo",
            "scheme": "url",
        })
    return related or None


def _build_creators(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.creator:
        return [_parse_creator(c) for c in args.creator]
    return [{"name": "<authors>"}]


def _apply_metadata(
    base_url: str, token: str, dep_id: int, args: argparse.Namespace
) -> None:
    set_metadata(
        base_url,
        token,
        dep_id,
        title=args.title or DEFAULT_TITLE,
        description=args.description or DEFAULT_DESCRIPTION,
        creators=_build_creators(args),
        version=args.version,
        keywords=args.keyword or None,
        related_identifiers=_build_related(args),
        conference_title=args.conference_title,
        conference_url=args.conference_url,
        language=args.language,
    )


def _load_and_verify_artifacts(
    args: argparse.Namespace, meta: dict
) -> list[Path]:
    artifacts = meta.get("artifacts", [])
    if not artifacts:
        print("ERROR: no artifacts listed in metadata.", file=sys.stderr)
        sys.exit(1)

    paths: list[Path] = []
    for entry in artifacts:
        p = Path(entry["path"])
        if not p.exists():
            print(f"ERROR: artifact not found: {p}", file=sys.stderr)
            sys.exit(1)
        if not args.no_verify_checksums:
            local_hash = _sha256(p)
            expected = entry.get("sha256", "")
            if local_hash != expected:
                print(
                    f"ERROR: checksum mismatch for {p}\n"
                    f"  expected: {expected}\n"
                    f"  got:      {local_hash}",
                    file=sys.stderr,
                )
                sys.exit(1)
        paths.append(p)
    return paths


def workflow_upload(args: argparse.Namespace, base_url: str, token: str) -> int:
    """Standard workflow: create draft → upload → set metadata → publish."""
    if not args.metadata.exists():
        print(f"ERROR: metadata file not found: {args.metadata}", file=sys.stderr)
        return 1
    with open(args.metadata) as f:
        meta = json.load(f)

    artifact_paths = _load_and_verify_artifacts(args, meta)

    deposit = create_deposit(base_url, token)
    dep_id = deposit["id"]
    bucket_url = deposit["links"]["bucket"]
    prereserved_doi = deposit["metadata"]["prereserve_doi"]["doi"]

    print(f"Uploading {len(artifact_paths)} file(s) ...", file=sys.stderr)
    for path in artifact_paths:
        upload_file(bucket_url, token, path)

    _apply_metadata(base_url, token, dep_id, args)

    if args.publish:
        print("Publishing (IRREVERSIBLE) ...", file=sys.stderr)
        result = publish_deposit(base_url, token, dep_id)
        print(f"\nDOI: {result['doi']}", file=sys.stderr)
        print(f"URL: https://doi.org/{result['doi']}", file=sys.stderr)
    else:
        web = base_url.replace("/api", "")
        print(
            f"\nDraft deposit created (NOT published).\n"
            f"  Deposit ID:       {dep_id}\n"
            f"  Pre-reserved DOI: {prereserved_doi}\n"
            f"  Edit/publish at:  {web}/deposit/{dep_id}\n"
            f"\nTo publish, re-run with --publish or use the web UI.",
            file=sys.stderr,
        )
    return 0


def workflow_new_version(
    args: argparse.Namespace, base_url: str, token: str
) -> int:
    """Create a new version of a published record, replace files, publish."""
    if not args.metadata.exists():
        print(f"ERROR: metadata file not found: {args.metadata}", file=sys.stderr)
        return 1
    with open(args.metadata) as f:
        meta = json.load(f)

    artifact_paths = _load_and_verify_artifacts(args, meta)

    draft = create_new_version(base_url, token, args.new_version)
    dep_id = draft["id"]
    bucket_url = draft["links"]["bucket"]

    # Remove old files inherited from previous version
    for old_file in draft.get("files", []):
        print(f"  Removing old file: {old_file['filename']}", file=sys.stderr)
        delete_file(base_url, token, dep_id, old_file["id"])

    print(f"Uploading {len(artifact_paths)} file(s) ...", file=sys.stderr)
    for path in artifact_paths:
        upload_file(bucket_url, token, path)

    _apply_metadata(base_url, token, dep_id, args)

    if args.publish:
        print("Publishing new version (IRREVERSIBLE) ...", file=sys.stderr)
        result = publish_deposit(base_url, token, dep_id)
        print(f"\nDOI: {result['doi']}", file=sys.stderr)
        print(f"URL: https://doi.org/{result['doi']}", file=sys.stderr)
    else:
        web = base_url.replace("/api", "")
        print(
            f"\nNew version draft created (NOT published).\n"
            f"  Deposit ID:      {dep_id}\n"
            f"  Edit/publish at: {web}/deposit/{dep_id}",
            file=sys.stderr,
        )
    return 0


def workflow_edit(args: argparse.Namespace, base_url: str, token: str) -> int:
    """Edit metadata of a published record without re-uploading files."""
    edit_published(base_url, token, args.edit)
    _apply_metadata(base_url, token, args.edit, args)

    if args.publish:
        print("Re-publishing with updated metadata ...", file=sys.stderr)
        result = publish_deposit(base_url, token, args.edit)
        print(f"\nDOI: {result['doi']}", file=sys.stderr)
    else:
        web = base_url.replace("/api", "")
        print(
            f"\nMetadata updated (NOT re-published).\n"
            f"  Edit/publish at: {web}/deposit/{args.edit}\n"
            f"  To discard changes, use the Zenodo web UI.",
            file=sys.stderr,
        )
    return 0


def main() -> int:
    args = parse_args()
    base_url = SANDBOX_API if args.sandbox else ZENODO_API

    # --- BibTeX fetch (no auth needed) ---
    if args.fetch_bibtex:
        bibtex = fetch_bibtex(base_url, args.fetch_bibtex)
        print(bibtex)
        return 0

    # --- Token required for all other workflows ---
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print(
            "ERROR: ZENODO_TOKEN environment variable not set.\n"
            "Create a token at https://zenodo.org/account/settings/"
            "applications/\n"
            "with scopes: deposit:write, deposit:actions",
            file=sys.stderr,
        )
        return 1

    env_label = "SANDBOX" if args.sandbox else "PRODUCTION"
    print(f"Target: {env_label} ({base_url})", file=sys.stderr)

    if args.new_version:
        return workflow_new_version(args, base_url, token)
    if args.edit is not None:
        return workflow_edit(args, base_url, token)
    return workflow_upload(args, base_url, token)


if __name__ == "__main__":
    sys.exit(main())
