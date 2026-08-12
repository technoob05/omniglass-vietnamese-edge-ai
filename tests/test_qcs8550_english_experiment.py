from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "manifests" / "qcs8550_english_experiment.json"
EXPORT = ROOT / "scripts" / "export_qcs8550_english_qnn.sh"
STAGE = ROOT / "scripts" / "stage_qcs8550_english_bundle.py"
BENCH = ROOT / "scripts" / "benchmark_qcs8550_english_stack.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_uses_small_htp_stack_and_h100_conversation_fallback() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["experiment"]["status"] == "physical_htp_smoke_verified__full_workflow_pending"
    smoke = manifest["physical_board_evidence"]["qnn_smoke"]
    assert smoke["inferences_completed"] == 20
    assert smoke["accelerator_execute_average_ms"] > 0
    components = {item["id"]: item for item in manifest["minimum_runnable_stack"]}
    assert components["detector"]["placement"] == "QNN HTP"
    assert "YOLO11-N" in components["detector"]["implementation"]
    assert components["depth"]["required"] is False
    assert components["english_conversation"]["placement"] == "H100"
    assert manifest["resource_policy"]["aggregate_peak_rss_gate_gib"] == 12


def test_minicpm_v46_is_explicitly_research_only() -> None:
    assessment = json.loads(MANIFEST.read_text(encoding="utf-8"))["minicpm_v46_htp_assessment"]
    assert assessment["decision"] == "research_lane_only"
    assert any("not a QNN context binary" in reason for reason in assessment["why_not_baseline"])
    assert any("projector" in requirement for requirement in assessment["promotion_requirements"])


def test_export_is_pinned_dry_by_default_and_does_not_install_or_manage_services() -> None:
    source = EXPORT.read_text(encoding="utf-8")
    assert "f413a03dc8845739afce27cd3e691d6a5a7339a3" in source
    assert 'RUN_EXPORT="${RUN_EXPORT:-0}"' in source
    assert "yolov11_det" in source and "depth_anything_v2" in source
    assert "qnn_context_binary" in source and "w8a16" in source
    for forbidden in ("pip install", "systemctl", "service ", "pkill", "rm -rf"):
        assert forbidden not in source


def test_stage_creates_checksum_manifest_without_touching_services(tmp_path: Path) -> None:
    module = load(STAGE, "stage_qcs8550_english_bundle")
    artifact = tmp_path / "detector.bin"
    artifact.write_bytes(b"qnn-context")
    destination = tmp_path / "bundle"
    inventory = module.stage(destination, [artifact])
    assert inventory["status"] == "staged_not_deployed"
    assert len(inventory["files"]) == 4
    assert all(len(record["sha256"]) == 64 for record in inventory["files"])
    assert (destination / "BUNDLE_MANIFEST.json").is_file()


def test_benchmark_uses_retrieve_context_and_rejects_missing_htp() -> None:
    module = load(BENCH, "benchmark_qcs8550_english_stack")
    command = module.qnn_command(
        Path("qnn-net-run"), Path("libQnnHtp.so"), Path("detector.bin"), Path("input.txt"), Path("out")
    )
    assert "--retrieve_context" in command
    assert "--profiling_level" in command
    assert module.HTP_PATTERN.search("QnnHtp device initialized")
    assert not module.HTP_PATTERN.search("CPU backend initialized")
    source = BENCH.read_text(encoding="utf-8")
    for forbidden in ("systemctl", "service ", "pkill", "rm -rf"):
        assert forbidden not in source
