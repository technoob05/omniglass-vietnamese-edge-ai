from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "manifests" / "vietnamese_dataset_candidates.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_candidates_are_unique_and_machine_auditable() -> None:
    manifest = load_manifest()
    assert manifest["schema_version"] == "1.0"
    candidates = manifest["candidates"]
    assert len({item["id"] for item in candidates}) == len(candidates)
    for item in candidates:
        assert item["modalities"]
        assert item["role"]
        assert item["status"]
        assert "commercial_use_allowed" in item["license"]
        assert item["rights_review"]["status"]
        source = item["source"]
        if source["kind"] != "private":
            assert source["url"].startswith("https://")
        if source["revision"] is not None:
            assert len(source["revision"]) == 40
            int(source["revision"], 16)


def test_unpinned_or_unlicensed_sources_are_not_approved_for_product_training() -> None:
    for item in load_manifest()["candidates"]:
        source = item["source"]
        license_info = item["license"]
        if source["revision"] is None and source["kind"] != "private":
            assert item["status"].startswith("blocked_")
        if license_info.get("spdx") is None and source["kind"] != "private":
            assert license_info["commercial_use_allowed"] is not True
            assert item["status"].startswith(("blocked_", "research_only_"))


def test_fleurs_test_is_never_a_training_source() -> None:
    candidates = {item["id"]: item for item in load_manifest()["candidates"]}
    fleurs = candidates["google-fleurs-vi_vn"]
    assert fleurs["status"] == "evaluation_only"
    assert fleurs["split_policy"]["test"] == "evaluation_only_never_train"
    assert "vlm_sft" not in fleurs["role"]


def test_card_license_does_not_silently_clear_vqa_images() -> None:
    candidates = {item["id"]: item for item in load_manifest()["candidates"]}
    for candidate_id in ("uitnlp-openvivqa", "dangindev-viet-cultural-vqa"):
        item = candidates[candidate_id]
        assert item["license"]["spdx"]
        assert item["license"]["commercial_use_allowed"] is None
        assert item["status"].startswith("blocked_pending_asset")
