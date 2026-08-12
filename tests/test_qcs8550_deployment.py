from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "manifests" / "qcs8550_deployment.json"
SCRIPT = ROOT / "scripts" / "qcs8550_preflight.py"
ACCEPTANCE_SCRIPT = ROOT / "scripts" / "qcs8550_acceptance.py"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_script():
    spec = importlib.util.spec_from_file_location("qcs8550_preflight", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_acceptance_script():
    spec = importlib.util.spec_from_file_location("qcs8550_acceptance", ACCEPTANCE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_never_claims_physical_board_verification() -> None:
    manifest = load_manifest()
    assert manifest["schema_version"] == "1.0"
    assert manifest["target"]["status"] == "not_physically_verified"
    assert manifest["upstream_runtime"]["qcs8550_linux_support_status"] == "upstream_verified"
    assert all(gate["status"] == "blocked" for gate in manifest["mandatory_gates"])
    assert "provisional_targets_not_measurements" in manifest


def test_runtime_candidates_preserve_vietnamese_and_license_boundaries() -> None:
    components = {item["id"]: item for item in load_manifest()["components"]}
    assert components["zipformer_vi_cpu"]["status"].endswith("planned_linux_arm64")
    assert "CC-BY-NC-ND-4.0" in components["zipformer_vi_cpu"]["license_gate"]
    assert components["phowhisper_qnn"]["status"] == "blocked"
    assert "English" in components["vietnamese_tts"]["blocker"]
    assert components["minicpmo_full_duplex"]["status"] == "blocked_research_only"


def test_preflight_is_read_only_and_keeps_all_release_gates_blocked() -> None:
    module = load_script()
    report = module.collect(MANIFEST)
    assert "no camera frames or microphone audio" in report["collector"]["privacy"]
    assert report["target_detection"]["claim_boundary"] == "Detection is not deployment verification."
    assert report["release_gates"]
    assert set(report["release_gates"].values()) == {"blocked_pending_physical_evidence"}


def test_detection_requires_explicit_qcs8550_identity() -> None:
    module = load_script()
    assert module.detect_qcs8550(["Qualcomm QCS8550", "Linux aarch64"])
    assert module.detect_qcs8550(["KIMQ 8550 board"])
    assert not module.detect_qcs8550(["AMD64 development host", "Windows"])


def test_missing_physical_metrics_remain_blocked() -> None:
    module = load_acceptance_script()
    report = module.evaluate(
        load_manifest(),
        {"target_detection": {"qcs8550_detected": False}},
        {"measurements": {}, "evidence": {}},
    )
    assert report["decision"] == "blocked"
    assert set(gate["status"] for gate in report["gates"].values()) == {"blocked"}


def test_complete_metrics_only_become_human_review_candidate() -> None:
    module = load_acceptance_script()
    digest = "a" * 64
    metrics = {
        "evidence": {"artifact_sha256": [digest], "accuracy_report_sha256": digest},
        "measurements": {
            "offline_boot_ok": True,
            "camera_audio_io_ok": True,
            "model_accuracy_passed": True,
            "detector_tracker_sustained_fps": 20,
            "critical_camera_to_audio_p95_ms": 250,
            "hybrid_vqa_first_text_p95_ms": 2500,
            "hybrid_vqa_first_audio_p95_ms": 3000,
            "htp_fallback_events": 0,
            "soak_minutes": 30,
            "thermal_latency_regression": 0.10,
            "unbounded_memory_growth": False,
            "power_thermal_trace_present": True,
            "network_loss_core_actions_ok": True,
            "raw_media_in_logs": False,
            "encrypted_transport": True,
            "mic_cloud_indicator": True,
            "turns": 100,
            "return_to_listening_rate": 0.99,
        },
    }
    report = module.evaluate(
        load_manifest(),
        {"target_detection": {"qcs8550_detected": True}},
        metrics,
    )
    assert report["decision"] == "ready_for_human_release_review"
    assert "not physical-board verification" in report["claim_boundary"]
