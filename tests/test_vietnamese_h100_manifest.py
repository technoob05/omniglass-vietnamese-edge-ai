from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_h100_manifest_is_current_and_conservative() -> None:
    manifest = json.loads(
        (ROOT / "manifests/vietnamese_h100_stack.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "deployed_and_e2e_verified"
    assert manifest["verified_results"]["browser_e2e_3turn"]["passed"] is True
    assert manifest["verified_results"]["protocol_soak_10turn"]["passed"] is True
    assert "speech end to first audible response P95 <= 2.0 s" in manifest["known_gates"]["not_yet_passed"]
    assert "physical QCS8550/QCS6490 deployment" in manifest["known_gates"]["not_yet_passed"]

    for artifact in manifest["source_artifacts"]:
        data = (ROOT / artifact["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]


def test_vietnamese_profile_uses_only_pinned_tool_roles() -> None:
    manifest = json.loads(
        (ROOT / "manifests/vietnamese_h100_stack.json").read_text(encoding="utf-8")
    )
    roles = {model["role"]: model for model in manifest["models"]}
    assert roles["vietnamese_asr"]["id"] == "vinai/PhoWhisper-medium"
    assert roles["vietnamese_tts_primary"]["default_voice"] == "Trúc Ly"
    assert roles["vietnamese_tts_fallback"]["fallback_rule"].startswith("used only")
    assert roles["visual_reasoner"]["deployment"] == "H100 primary only"
