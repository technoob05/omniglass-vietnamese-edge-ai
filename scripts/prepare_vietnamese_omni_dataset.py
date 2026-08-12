#!/usr/bin/env python3
"""Validate and convert consented Vietnamese multimodal records to ms-swift JSONL.

The input is deliberately provenance-first.  Each JSONL row must include a
stable speaker/session group, explicit training permission, Vietnamese prompt
and answer, and at least one image or audio file.  Media is referenced, never
copied, so raw glasses recordings do not accidentally enter the Git repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALLOWED_TASKS = {
    "describe_scene",
    "read_text",
    "find_object",
    "answer_visual_question",
    "safety_abstain",
    "audio_understanding",
    "tool_plan",
}
ALLOWED_IMAGE_MEDIA = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_AUDIO_MEDIA = {".wav", ".flac", ".mp3", ".m4a"}
VISUAL_TASKS = {"describe_scene", "read_text", "find_object", "answer_visual_question"}
SYSTEM_PROMPT = (
    "Bạn là trợ lý thị giác tiếng Việt. Hãy trả lời ngắn, tự nhiên và dựa đúng vào dữ liệu cảm biến. "
    "Nếu không nhìn hoặc nghe đủ rõ, hãy nói rằng bạn chưa chắc chắn. Không bịa khoảng cách, chữ viết "
    "hoặc nguy hiểm. Đây không phải thiết bị dẫn đường an toàn."
)


class DatasetError(ValueError):
    pass


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_split(group_id: str, train: int, validation: int) -> str:
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"


def _resolve_media(base: Path, values: Any, field: str, allowed_extensions: set[str]) -> list[Path]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise DatasetError(f"{field} must be a path string or list of path strings")
    result: list[Path] = []
    for item in values:
        path = (base / item).resolve() if not Path(item).is_absolute() else Path(item).resolve()
        if not path.is_file():
            raise DatasetError(f"missing {field} file: {path}")
        if path.suffix.lower() not in allowed_extensions:
            raise DatasetError(f"unsupported {field} extension: {path.suffix}")
        result.append(path)
    return result


def convert_records(
    input_path: Path,
    output_dir: Path,
    *,
    train_percent: int = 80,
    validation_percent: int = 10,
) -> dict[str, Any]:
    if train_percent <= 0 or validation_percent < 0 or train_percent + validation_percent >= 100:
        raise DatasetError("split percentages must leave a non-empty test partition")
    base = input_path.resolve().parent
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    group_splits: dict[str, str] = {}
    media_splits: dict[str, set[str]] = defaultdict(set)
    task_counts: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"line {line_number}: each record must be a JSON object")
            record_id = _nfc(str(row.get("id", "")))
            group_id = _nfc(str(row.get("group_id") or row.get("speaker_id") or ""))
            if not record_id or record_id in ids:
                raise DatasetError(f"line {line_number}: id is missing or duplicated: {record_id!r}")
            if not group_id:
                raise DatasetError(f"line {line_number}: group_id/speaker_id is required")
            ids.add(record_id)

            consent = row.get("consent", {})
            if consent is None:
                consent = {}
            if not isinstance(consent, dict):
                raise DatasetError(f"line {line_number}: consent must be a JSON object")
            if consent.get("training") is not True:
                raise DatasetError(f"line {line_number}: explicit consent.training=true is required")
            raw_media_public = consent.get("raw_media_public", False)
            if not isinstance(raw_media_public, bool):
                raise DatasetError(f"line {line_number}: consent.raw_media_public must be boolean")
            if raw_media_public:
                raise DatasetError(
                    f"line {line_number}: keep raw media private; publish only a separately reviewed derivative"
                )
            license_value = row["license"] if "license" in row else consent.get("license", "")
            if license_value is None:
                license_value = consent.get("license", "")
            if not isinstance(license_value, str):
                raise DatasetError(f"line {line_number}: license/provenance label must be a string")
            license_id = _nfc(license_value)
            if not license_id:
                raise DatasetError(f"line {line_number}: license/provenance label is required")

            task = str(row.get("task", "")).strip()
            if task not in ALLOWED_TASKS:
                raise DatasetError(f"line {line_number}: unsupported task {task!r}")
            question = _nfc(str(row.get("question_vi", "")))
            answer = _nfc(str(row.get("answer_vi", "")))
            if not question or not answer:
                raise DatasetError(f"line {line_number}: question_vi and answer_vi are required")

            images = _resolve_media(base, row.get("images"), "images", ALLOWED_IMAGE_MEDIA)
            audios = _resolve_media(base, row.get("audios"), "audios", ALLOWED_AUDIO_MEDIA)
            if not images and not audios:
                raise DatasetError(f"line {line_number}: at least one image or audio is required")
            if task in VISUAL_TASKS and not images:
                raise DatasetError(f"line {line_number}: {task} requires an image")
            if task == "audio_understanding" and not audios:
                raise DatasetError(f"line {line_number}: audio_understanding requires audio")

            split = str(row.get("split") or _stable_split(group_id, train_percent, validation_percent))
            if split == "val":
                split = "validation"
            if split not in {"train", "validation", "test"}:
                raise DatasetError(f"line {line_number}: invalid split {split!r}")
            previous = group_splits.setdefault(group_id, split)
            if previous != split:
                raise DatasetError(f"line {line_number}: group {group_id!r} leaks across splits")

            for media in images + audios:
                media_splits[_sha256(media)].add(split)

            tags = "".join("<image>" for _ in images) + "".join("<audio>" for _ in audios)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{tags}\n{question}".strip()},
                {"role": "assistant", "content": answer},
            ]
            converted: dict[str, Any] = {
                "messages": messages,
                "metadata": {
                    "id": record_id,
                    "group_id": group_id,
                    "task": task,
                    "language": "vi-VN",
                    "license": license_id,
                },
            }
            if images:
                converted["images"] = [str(path) for path in images]
            if audios:
                converted["audios"] = [str(path) for path in audios]
            rows.append({"split": split, "data": converted})
            task_counts[task] += 1

    leaked = sorted(digest for digest, splits in media_splits.items() if len(splits) > 1)
    if leaked:
        raise DatasetError(f"identical media appears in multiple splits ({len(leaked)} SHA-256 collisions)")
    if not rows:
        raise DatasetError("input contains no records")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts: Counter[str] = Counter()
    for split in ("train", "validation", "test"):
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in rows:
                if item["split"] == split:
                    handle.write(json.dumps(item["data"], ensure_ascii=False, sort_keys=True) + "\n")
                    split_counts[split] += 1

    report = {
        "schema_version": "1.0",
        "source_manifest": str(input_path.resolve()),
        "record_count": len(rows),
        "group_count": len(group_splits),
        "split_counts": dict(split_counts),
        "task_counts": dict(task_counts),
        "media_sha256_count": len(media_splits),
        "leakage_checks": {"group": "passed", "media_sha256": "passed"},
        "privacy": "raw media referenced only; not copied",
    }
    (output_dir / "dataset-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Provenance-first source JSONL")
    parser.add_argument("output_dir", type=Path, help="Private output directory for ms-swift JSONL")
    parser.add_argument("--train-percent", type=int, default=80)
    parser.add_argument("--validation-percent", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = convert_records(
            args.input,
            args.output_dir,
            train_percent=args.train_percent,
            validation_percent=args.validation_percent,
        )
    except (DatasetError, OSError) as exc:
        print(f"dataset validation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
