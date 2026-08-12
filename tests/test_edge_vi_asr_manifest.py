from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "manifests" / "edge_vi_asr_baselines.json"
SCRIPT = ROOT / "scripts" / "smoke_edge_vi_asr.py"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_is_explicit_about_claim_and_license_boundaries() -> None:
    manifest = load_manifest()
    assert manifest["schema_version"] == "1.0"
    baselines = {item["id"]: item for item in manifest["baselines"]}
    assert len(baselines) == len(manifest["baselines"])

    sherpa = baselines["sherpa-onnx-zipformer-vi-30m-int8-cpu"]
    assert sherpa["licenses"]["model"]["spdx"] == "CC-BY-NC-ND-4.0"
    assert sherpa["licenses"]["model"]["commercial_use_allowed"] is False
    assert sherpa["runtime"]["provider"] == "cpu"
    assert len(sherpa["runtime"]["verified_x86_64_wheels"]) == 2
    assert sherpa["runtime"]["arm64_runtime_artifact"] is None
    assert all(target["status"] != "verified" for target in sherpa["targets"] if "QCS" in target["target"])
    assert any(claim["status"] == "blocked" and "HTP/NPU" in claim["claim"] for claim in sherpa["claims"])

    qnn = baselines["phowhisper-medium-qnn-feasibility"]
    assert qnn["overall_status"] == "planned_blocked"
    assert qnn["qnn_plan"]["context_binary_sha256"] is None
    assert qnn["qnn_plan"]["artifact_status"] == "absent"
    assert len(qnn["qnn_plan"]["blockers"]) >= 4
    assert all(target["status"] == "planned_blocked" for target in qnn["targets"])


def test_all_frozen_artifacts_have_well_formed_hashes() -> None:
    manifest = load_manifest()
    for baseline in manifest["baselines"]:
        artifacts = baseline.get("artifacts", []) + baseline.get("source_artifacts", [])
        artifacts += baseline.get("runtime", {}).get("verified_x86_64_wheels", [])
        for artifact in artifacts:
            digest = artifact["sha256"]
            assert len(digest) == 64
            int(digest, 16)
    archive = manifest["baselines"][0]["download"]
    assert archive["url"].startswith("https://")
    assert archive["archive_bytes"] > 0
    assert len(archive["sha256"]) == 64


def test_smoke_harness_loads_manifest_without_sherpa_dependency() -> None:
    spec = importlib.util.spec_from_file_location("smoke_edge_vi_asr", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    manifest, baseline = module.load_manifest(MANIFEST)
    assert manifest["schema_version"] == "1.0"
    assert baseline["id"] == module.BASELINE_ID
