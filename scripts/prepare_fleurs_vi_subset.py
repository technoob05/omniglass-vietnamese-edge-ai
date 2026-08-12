#!/usr/bin/env python3
"""Materialize a small, revision-locked Vietnamese FLEURS ASR subset.

The source archive is streamed and deliberately abandoned as soon as the
requested number of eligible WAV members has been copied.  The full archive
is never persisted.  This keeps the benchmark input small while preserving an
auditable relationship to the official Google FLEURS repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import requests
import soundfile as sf


DATASET_ID = "google/fleurs"
DATASET_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
CONFIG = "vi_vn"
SPLIT = "test"
LICENSE = "CC-BY-4.0"
TEST_TSV_PATH = "data/vi_vn/test.tsv"
TEST_ARCHIVE_PATH = "data/vi_vn/audio/test.tar.gz"
TEST_ARCHIVE_LFS_SHA256 = "b359261216ef14b8eea159abce794e270c2a0be903218cf1e1e8f7c7c1565b5c"
TEST_ARCHIVE_BYTES = 544_139_865
TEST_ROWS = 857


def source_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{path}"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_tsv(payload: bytes) -> dict[str, dict[str, object]]:
    text = payload.decode("utf-8")
    rows: dict[str, dict[str, object]] = {}
    for row_number, fields in enumerate(csv.reader(io.StringIO(text), delimiter="\t"), 1):
        # Six official vi_vn test rows quote an embedded tab between the
        # normalized transcription and phoneme columns.  Python's CSV parser
        # correctly preserves that tab inside one field, so split it back into
        # the two documented columns.  Record this source quirk in the manifest.
        if len(fields) == 6 and "\t" in fields[3]:
            normalized_text, phonemes = fields[3].split("\t", 1)
            fields = [*fields[:3], normalized_text, phonemes, *fields[4:]]
        if len(fields) != 7:
            raise RuntimeError(f"Unexpected FLEURS TSV row {row_number}: {len(fields)} fields")
        sentence_id, filename, raw_text, normalized_text, phonemes, num_samples, gender = fields
        if filename in rows:
            raise RuntimeError(f"Duplicate audio filename in TSV: {filename}")
        rows[filename] = {
            "sentence_id": int(sentence_id),
            "filename": filename,
            "raw_transcription": raw_text,
            "transcription": normalized_text,
            "phonemes": phonemes,
            "num_samples": int(num_samples),
            "gender": gender.lower(),
            "tsv_row": row_number,
        }
    if len(rows) != TEST_ROWS:
        raise RuntimeError(f"Expected {TEST_ROWS} Vietnamese test rows, found {len(rows)}")
    return rows


def wav_metadata(payload: bytes) -> dict[str, object]:
    info = sf.info(io.BytesIO(payload))
    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "audio_subtype": info.subtype,
        "frames": info.frames,
        "duration_seconds": round(info.frames / info.samplerate, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if not 30 <= args.count <= 100:
        raise ValueError("--count must stay within the approved 30-100 utterance range")

    output_dir = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    tsv_url = source_url(TEST_TSV_PATH)
    tsv_response = requests.get(tsv_url, timeout=args.timeout)
    tsv_response.raise_for_status()
    tsv_payload = tsv_response.content
    metadata_by_name = parse_tsv(tsv_payload)

    selected: list[dict[str, object]] = []
    skipped_too_long: list[str] = []
    archive_url = source_url(TEST_ARCHIVE_PATH)
    with requests.get(archive_url, stream=True, timeout=args.timeout) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
            for member in archive:
                if len(selected) >= args.count:
                    break
                if not member.isfile() or not member.name.lower().endswith(".wav"):
                    continue
                member_path = PurePosixPath(member.name)
                filename = member_path.name
                row = metadata_by_name.get(filename)
                if row is None:
                    raise RuntimeError(f"Archive member absent from pinned TSV: {member.name}")
                if int(row["num_samples"]) > int(args.max_seconds * 16_000):
                    skipped_too_long.append(filename)
                    continue
                fileobj = archive.extractfile(member)
                if fileobj is None:
                    raise RuntimeError(f"Could not read archive member: {member.name}")
                wav_payload = fileobj.read()
                output_path = audio_dir / filename
                temporary = output_path.with_suffix(".wav.part")
                temporary.write_bytes(wav_payload)
                temporary.replace(output_path)
                actual = wav_metadata(wav_payload)
                if actual["sample_rate"] != 16_000 or actual["channels"] != 1:
                    raise RuntimeError(f"Unexpected audio format for {filename}: {actual}")
                selected.append({
                    "sample_index": len(selected),
                    "sample_id": f"fleurs-{CONFIG}-{SPLIT}-{row['sentence_id']}-{output_path.stem}",
                    "audio_path": str(output_path),
                    "archive_member": member.name,
                    "audio_sha256": sha256_bytes(wav_payload),
                    **row,
                    **actual,
                })

    if len(selected) != args.count:
        raise RuntimeError(f"Requested {args.count} samples but streamed only {len(selected)}")

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "config": CONFIG,
            "split": SPLIT,
            "license": LICENSE,
            "license_source": (
                f"https://huggingface.co/datasets/{DATASET_ID}/blob/"
                f"{DATASET_REVISION}/README.md"
            ),
            "test_rows": TEST_ROWS,
            "official_tsv_repaired_quoted_tab_rows": 6,
            "sample_rate": 16_000,
            "accent_metadata_available": False,
            "known_scope_limitations": [
                "read speech rather than spontaneous assistive commands",
                "Vietnamese test split exposes no regional-accent field",
                "official Dataset Viewer statistics report all 857 vi_vn test recordings as male",
            ],
        },
        "common_voice_decision": {
            "used": False,
            "reason": (
                "The official Hugging Face Common Voice repository contains no data files and "
                "states that downloads moved to Mozilla Data Collective in October 2025."
            ),
            "repository": "mozilla-foundation/common_voice_17_0",
            "observed_revision": "11dc88355e899d1bf2df74f01b904a8544a17b33",
            "dataset_viewer_valid": False,
        },
        "source_artifacts": {
            "test_tsv": {
                "url": tsv_url,
                "bytes": len(tsv_payload),
                "sha256": sha256_bytes(tsv_payload),
            },
            "test_audio_archive": {
                "url": archive_url,
                "full_archive_not_persisted": True,
                "official_lfs_sha256": TEST_ARCHIVE_LFS_SHA256,
                "official_bytes": TEST_ARCHIVE_BYTES,
            },
            "dataset_viewer_statistics": (
                "https://datasets-server.huggingface.co/statistics?"
                "dataset=google%2Ffleurs&config=vi_vn&split=test"
            ),
        },
        "selection": {
            "rule": (
                "first N regular WAV members in pinned test archive order whose TSV duration "
                "is at most max_seconds"
            ),
            "requested": args.count,
            "selected": len(selected),
            "max_seconds": args.max_seconds,
            "skipped_too_long_before_completion": skipped_too_long,
            "unique_sentence_ids": len({sample["sentence_id"] for sample in selected}),
            "gender_counts": dict(Counter(str(sample["gender"]) for sample in selected)),
            "total_audio_seconds": round(sum(float(sample["duration_seconds"]) for sample in selected), 4),
        },
        "samples": selected,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "selected": len(selected),
        "unique_sentence_ids": manifest["selection"]["unique_sentence_ids"],
        "total_audio_seconds": manifest["selection"]["total_audio_seconds"],
        "gender_counts": manifest["selection"]["gender_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
