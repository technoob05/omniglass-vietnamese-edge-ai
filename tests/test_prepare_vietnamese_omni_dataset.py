from __future__ import annotations

import importlib.util
import json
import sys
import unicodedata
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "prepare_vietnamese_omni_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_vietnamese_omni_dataset", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def converter():
    return load_module()


def write_media(path: Path, payload: bytes = b"media") -> Path:
    path.write_bytes(payload)
    return path


def write_manifest(path: Path, *records: object) -> Path:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def record(**overrides: object) -> dict:
    value: dict[str, object] = {
        "id": "sample-1",
        "group_id": "speaker-1/session-1",
        "consent": {"training": True, "raw_media_public": False},
        "license": "LicenseRef-Consented-Private-v1",
        "task": "describe_scene",
        "question_vi": "Bạn nhìn thấy gì?",
        "answer_vi": "Tôi thấy một chiếc ghế.",
        "images": ["frame.jpg"],
        "split": "train",
    }
    value.update(overrides)
    return value


def test_valid_conversion_is_ms_swift_compatible_and_nfc(tmp_path: Path, converter) -> None:
    write_media(tmp_path / "frame.jpg", b"jpeg-one")
    write_media(tmp_path / "utterance.wav", b"wav-one")
    question_nfd = unicodedata.normalize("NFD", "Ghế màu gì?")
    answer_nfd = unicodedata.normalize("NFD", "Ghế màu đỏ.")
    group_nfd = unicodedata.normalize("NFD", "người-1/phiên-1")
    source = write_manifest(
        tmp_path / "source.jsonl",
        record(
            id="mẫu-1",
            group_id=group_nfd,
            task="answer_visual_question",
            question_vi=f"  {question_nfd}  ",
            answer_vi=f"  {answer_nfd}  ",
            images="frame.jpg",
            audios="utterance.wav",
        ),
    )

    report = converter.convert_records(source, tmp_path / "converted")

    assert report["record_count"] == 1
    assert report["group_count"] == 1
    assert report["split_counts"] == {"train": 1}
    assert report["task_counts"] == {"answer_visual_question": 1}
    assert report["leakage_checks"] == {"group": "passed", "media_sha256": "passed"}

    rows = [
        json.loads(line)
        for line in (tmp_path / "converted" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    converted = rows[0]
    assert converted["messages"][0]["role"] == "system"
    assert "trợ lý thị giác tiếng Việt" in converted["messages"][0]["content"]
    assert "Ã" not in converted["messages"][0]["content"]
    assert converted["messages"][1] == {
        "role": "user",
        "content": "<image><audio>\nGhế màu gì?",
    }
    assert converted["messages"][2] == {"role": "assistant", "content": "Ghế màu đỏ."}
    assert converted["metadata"]["group_id"] == unicodedata.normalize("NFC", group_nfd)
    assert converted["metadata"]["language"] == "vi-VN"
    assert converted["images"] == [str((tmp_path / "frame.jpg").resolve())]
    assert converted["audios"] == [str((tmp_path / "utterance.wav").resolve())]
    assert (tmp_path / "converted" / "validation.jsonl").read_text(encoding="utf-8") == ""
    assert (tmp_path / "converted" / "test.jsonl").read_text(encoding="utf-8") == ""


def test_speaker_id_fallback_and_val_alias(tmp_path: Path, converter) -> None:
    write_media(tmp_path / "frame.png")
    row = record(group_id=None, speaker_id="speaker-fallback", images="frame.png", split="val")
    report = converter.convert_records(
        write_manifest(tmp_path / "source.jsonl", row), tmp_path / "converted"
    )
    assert report["split_counts"] == {"validation": 1}


def test_implicit_split_is_stable_for_every_record_in_group(tmp_path: Path, converter) -> None:
    write_media(tmp_path / "one.jpg", b"one")
    write_media(tmp_path / "two.jpg", b"two")
    first = record(id="one", images="one.jpg")
    second = record(id="two", images="two.jpg")
    first.pop("split")
    second.pop("split")
    report = converter.convert_records(
        write_manifest(tmp_path / "source.jsonl", first, second), tmp_path / "converted"
    )
    assert max(report["split_counts"].values()) == 2
    assert sum(report["split_counts"].values()) == 2


@pytest.mark.parametrize(
    ("bad_consent", "message"),
    [
        ({}, "explicit consent.training=true"),
        ({"training": False}, "explicit consent.training=true"),
        ({"training": 1}, "explicit consent.training=true"),
        ({"training": True, "raw_media_public": True}, "keep raw media private"),
        ({"training": True, "raw_media_public": "false"}, "must be boolean"),
        (["training"], "consent must be a JSON object"),
    ],
)
def test_consent_is_explicit_and_strict(
    tmp_path: Path, converter, bad_consent: object, message: str
) -> None:
    write_media(tmp_path / "frame.jpg")
    source = write_manifest(tmp_path / "source.jsonl", record(consent=bad_consent))
    with pytest.raises(converter.DatasetError, match=message):
        converter.convert_records(source, tmp_path / "converted")


@pytest.mark.parametrize(
    ("license_value", "message"),
    [("", "license/provenance label is required"), ([], "must be a string")],
)
def test_license_provenance_label_is_required_and_typed(
    tmp_path: Path, converter, license_value: object, message: str
) -> None:
    write_media(tmp_path / "frame.jpg")
    source = write_manifest(tmp_path / "source.jsonl", record(license=license_value))
    with pytest.raises(converter.DatasetError, match=message):
        converter.convert_records(source, tmp_path / "converted")


def test_license_can_come_from_consent_object(tmp_path: Path, converter) -> None:
    write_media(tmp_path / "frame.jpg")
    source = write_manifest(
        tmp_path / "source.jsonl",
        record(license=None, consent={"training": True, "license": "LicenseRef-Private-v1"}),
    )
    converter.convert_records(source, tmp_path / "converted")
    row = json.loads((tmp_path / "converted" / "train.jsonl").read_text(encoding="utf-8"))
    assert row["metadata"]["license"] == "LicenseRef-Private-v1"


def test_nfc_equivalent_ids_are_duplicates(tmp_path: Path, converter) -> None:
    write_media(tmp_path / "one.jpg", b"one")
    write_media(tmp_path / "two.jpg", b"two")
    id_nfc = "mẫu"
    id_nfd = unicodedata.normalize("NFD", id_nfc)
    source = write_manifest(
        tmp_path / "source.jsonl",
        record(id=id_nfc, images="one.jpg"),
        record(id=id_nfd, images="two.jpg"),
    )
    with pytest.raises(converter.DatasetError, match="id is missing or duplicated"):
        converter.convert_records(source, tmp_path / "converted")


def test_group_leakage_is_rejected(tmp_path: Path, converter) -> None:
    write_media(tmp_path / "one.jpg", b"one")
    write_media(tmp_path / "two.jpg", b"two")
    source = write_manifest(
        tmp_path / "source.jsonl",
        record(id="one", images="one.jpg", split="train"),
        record(id="two", images="two.jpg", split="test"),
    )
    with pytest.raises(converter.DatasetError, match="group .* leaks across splits"):
        converter.convert_records(source, tmp_path / "converted")


def test_byte_identical_media_leakage_is_rejected_across_groups(tmp_path: Path, converter) -> None:
    write_media(tmp_path / "one.jpg", b"same-image")
    write_media(tmp_path / "copy.jpg", b"same-image")
    source = write_manifest(
        tmp_path / "source.jsonl",
        record(id="one", group_id="group-one", images="one.jpg", split="train"),
        record(id="two", group_id="group-two", images="copy.jpg", split="test"),
    )
    with pytest.raises(converter.DatasetError, match="identical media appears in multiple splits"):
        converter.convert_records(source, tmp_path / "converted")


@pytest.mark.parametrize("task", sorted(load_module().VISUAL_TASKS))
def test_visual_tasks_require_images(tmp_path: Path, converter, task: str) -> None:
    write_media(tmp_path / "utterance.wav")
    source = write_manifest(
        tmp_path / "source.jsonl", record(task=task, images=None, audios="utterance.wav")
    )
    with pytest.raises(converter.DatasetError, match=f"{task} requires an image"):
        converter.convert_records(source, tmp_path / "converted")


def test_audio_understanding_requires_audio(tmp_path: Path, converter) -> None:
    write_media(tmp_path / "frame.jpg")
    source = write_manifest(
        tmp_path / "source.jsonl", record(task="audio_understanding", images="frame.jpg")
    )
    with pytest.raises(converter.DatasetError, match="audio_understanding requires audio"):
        converter.convert_records(source, tmp_path / "converted")


@pytest.mark.parametrize(
    ("field", "filename"),
    [("images", "wrong.wav"), ("audios", "wrong.jpg")],
)
def test_media_type_cannot_be_smuggled_through_wrong_field(
    tmp_path: Path, converter, field: str, filename: str
) -> None:
    write_media(tmp_path / filename)
    overrides: dict[str, object] = {"images": None, "audios": None, field: filename, "task": "tool_plan"}
    source = write_manifest(tmp_path / "source.jsonl", record(**overrides))
    with pytest.raises(converter.DatasetError, match=f"unsupported {field} extension"):
        converter.convert_records(source, tmp_path / "converted")


@pytest.mark.parametrize("task", ["safety_abstain", "tool_plan"])
def test_cross_modal_tasks_accept_audio_only(tmp_path: Path, converter, task: str) -> None:
    write_media(tmp_path / "utterance.flac")
    source = write_manifest(
        tmp_path / "source.jsonl",
        record(task=task, images=None, audios="utterance.flac"),
    )
    converter.convert_records(source, tmp_path / "converted")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "each record must be a JSON object"),
        ({"id": "only-an-id"}, "group_id/speaker_id is required"),
    ],
)
def test_record_shape_is_validated(tmp_path: Path, converter, payload: object, message: str) -> None:
    source = write_manifest(tmp_path / "source.jsonl", payload)
    with pytest.raises(converter.DatasetError, match=message):
        converter.convert_records(source, tmp_path / "converted")


@pytest.mark.parametrize(("train", "validation"), [(0, 10), (80, -1), (90, 10), (100, 0)])
def test_invalid_split_percentages_are_rejected(
    tmp_path: Path, converter, train: int, validation: int
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("", encoding="utf-8")
    with pytest.raises(converter.DatasetError, match="non-empty test partition"):
        converter.convert_records(
            source, tmp_path / "converted", train_percent=train, validation_percent=validation
        )


def test_empty_manifest_is_rejected(tmp_path: Path, converter) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("\n", encoding="utf-8")
    with pytest.raises(converter.DatasetError, match="input contains no records"):
        converter.convert_records(source, tmp_path / "converted")
