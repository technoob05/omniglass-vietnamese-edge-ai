#!/usr/bin/env python3
"""Create a checksum-pinned, service-neutral QCS8550 experiment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "qcs8550_english_experiment.json"
PORTABLE_FILES = (
    ROOT / "scripts" / "qcs8550_preflight.py",
    ROOT / "scripts" / "benchmark_qcs8550_english_stack.py",
    DEFAULT_MANIFEST,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(destination: Path, artifacts: list[Path], manifest: Path = DEFAULT_MANIFEST) -> dict:
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Destination must be absent or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    payload = destination / "payload"
    payload.mkdir()

    sources = list(PORTABLE_FILES)
    sources[-1] = manifest.resolve()
    sources.extend(path.resolve() for path in artifacts)
    if len({str(path) for path in sources}) != len(sources):
        raise ValueError("Duplicate source path")

    records = []
    for index, source in enumerate(sources):
        if not source.is_file():
            raise FileNotFoundError(source)
        target = payload / f"{index:02d}_{source.name}"
        shutil.copy2(source, target)
        records.append(
            {
                "source_name": source.name,
                "bundle_path": str(target.relative_to(destination)),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )

    inventory = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "staged_not_deployed",
        "claim_boundary": "This bundle has not run on a physical QCS8550 and does not modify board services.",
        "files": records,
    }
    inventory_path = destination / "BUNDLE_MANIFEST.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    args = parser.parse_args()
    inventory = stage(args.destination, args.artifact, args.manifest)
    print(json.dumps(inventory, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
