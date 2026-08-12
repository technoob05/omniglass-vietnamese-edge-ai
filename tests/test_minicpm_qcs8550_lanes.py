from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "manifests" / "minicpm_qcs8550_lanes.json"
BUILD = ROOT / "scripts" / "build_minicpm_qcs8550_runtime.sh"
DOWNLOAD = ROOT / "scripts" / "download_minicpm_qcs8550_models.sh"
BENCHMARK = ROOT / "scripts" / "benchmark_minicpm_qcs8550.sh"
OMNI_BENCHMARK = ROOT / "scripts" / "benchmark_minicpmo_qcs8550.sh"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_profiles_are_additive_and_preserve_native_english() -> None:
    policy = load_manifest()["profile_policy"]
    assert "preserved unchanged" in policy["native_english_chinese"]
    assert "additive" in policy["vietnamese"]
    assert "prohibited" in policy["model_replacement"]


def test_minicpm_v46_1_3b_is_primary_and_qwen_is_control_only() -> None:
    lanes = {item["id"]: item for item in load_manifest()["lanes"]}
    primary = lanes["A_minicpmv46_modular"]
    assert primary["priority"] == 1
    assert primary["parameters"] == 1_300_428_016
    assert primary["bundle_bytes"] == 1_610_003_840
    assert primary["artifacts"][0]["name"].endswith("Q4_0.gguf")
    assert "Q4_0" in primary["quantization_reason"]
    assert lanes["B_minicpmv45_quality_fallback"]["priority"] == 2
    assert lanes["C_minicpmo45_full"]["native_talker"] is True
    assert lanes["D_qwen3vl4b_control"]["priority"] == 4
    assert "comparison only" in lanes["D_qwen3vl4b_control"]["role"]
    assert all("unverified" in item["status"] for item in lanes.values())


def test_runtime_records_executable_linux_preset_without_opencl() -> None:
    runtime = load_manifest()["runtime"]
    assert runtime["revision"] == "09f5c3f1b484759f17b06fc63574f749c89c8761"
    assert runtime["linux_preset_backends"] == ["CPU ARM64", "Hexagon HTP"]
    assert runtime["linux_preset_opencl"] is False
    assert "not built or measured" in runtime["physical_status"]


def test_host_16gb_smoke_distinguishes_fit_from_realtime() -> None:
    smoke = load_manifest()["host_16gb_budget_smoke"]
    assert smoke["budget_ram_gib"] == 16
    assert smoke["peak_rss_gib"] < smoke["peak_gate_gib"]
    assert smoke["memory_fit_passed"] is True
    assert smoke["realtime_passed"] is False
    assert smoke["language_quality_passed"] is False
    assert "no physical QCS8550" in smoke["claim_boundary"]


def test_build_is_revision_pinned_and_never_mutates_services() -> None:
    source = BUILD.read_text(encoding="utf-8")
    assert "09f5c3f1b484759f17b06fc63574f749c89c8761" in source
    assert 'BRANCH="master"' in source
    assert "arm64-linux-snapdragon-release" in source
    assert "snapdragon-toolchain/arm64-linux:v0.1" in source
    assert "llama-mtmd-cli" in source
    assert "llama-omni-cli" in source
    assert "llama-omni-server" in source
    assert "llama-omni-single-test-omni" in source
    assert "BUILD_EVIDENCE.txt" in source
    for forbidden in ("pkill", "systemctl", "service ", "rm -rf"):
        assert forbidden not in source


def test_full_omni_harness_is_bounded_and_keeps_tts_as_second_stage() -> None:
    source = OMNI_BENCHMARK.read_text(encoding="utf-8")
    assert "llama-omni-single-test-omni" in source
    assert "MiniCPM-o-4_5-Q4_0.gguf" in source
    assert "--no-tts" in source
    assert "omni_with_tts.log" in source
    assert "GGML_HEXAGON_PROFILE=1" in source
    assert "rejecting possible CPU-only fallback" in source
    for forbidden in ("pkill", "systemctl", "service ", "rm -rf"):
        assert forbidden not in source


def test_downloads_pin_official_artifacts_and_support_dry_run() -> None:
    source = DOWNLOAD.read_text(encoding="utf-8")
    assert "openbmb/MiniCPM-V-4.6-gguf" in source
    assert "MiniCPM-V-4_6-Q4_0.gguf" in source
    assert "78e02f066e9819a60573b78a4275df8a0c27f698" in source
    assert "openbmb/MiniCPM-V-4_5-gguf" in source
    assert "cefe1580fe9402b06c4e1b8ed7343809377b8147" in source
    assert "openbmb/MiniCPM-o-4_5-gguf" in source
    assert "MiniCPM-o-4_5-Q4_0.gguf" in source
    assert "f706cc65f45288ef13f18d60834a9141c8e40b8f" in source
    assert "--dry-run" in source
    assert "SHA256SUMS" in source
    assert "qualcomm/Qwen" not in source


def test_board_harness_compares_cpu_and_htp_and_rejects_silent_fallback() -> None:
    source = BENCHMARK.read_text(encoding="utf-8")
    assert "llama-mtmd-cli" in source
    assert "llama-bench" in source
    assert "--list-devices" in source
    assert "GGML_HEXAGON_VERBOSE=1" in source
    assert "GGML_HEXAGON_PROFILE=1" in source
    assert "--no-mmproj-offload" in source
    assert "/usr/bin/time" in source
    assert "thermal_zone" in source
    assert "rejecting silent fallback" in source
    for forbidden in ("pkill", "systemctl", "service ", "rm -rf"):
        assert forbidden not in source
