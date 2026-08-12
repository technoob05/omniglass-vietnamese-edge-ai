import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_vietnamese_tts_candidates.py"
MANIFEST = ROOT / "manifests" / "vietnamese_tts_candidates.json"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("tts_candidates", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fixed_suite_covers_edge_cases():
    module = load_benchmark_module()
    assert len(module.SENTENCES) == 10
    joined = " ".join(module.SENTENCES)
    assert "8550" in joined
    assert "Nguyễn Thị Minh Khai" in joined
    assert "Cẩn thận" in joined


def test_vietnamese_normalization_and_edit_distance():
    module = load_benchmark_module()
    assert module.normalize_vi("  XIN CHÀO! ") == "xin chào"
    assert module.edit_distance("xin chào".split(), "xin bạn".split()) == 1


def test_candidate_manifest_is_fail_closed():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in manifest["candidates"]}
    assert manifest["decision"]["h100_primary"] == "pnnbao-ump/VieNeu-TTS-v3-Turbo"
    assert by_id["anphunl/Kokoro-Vietnamese"]["status"] == "edge_candidate_not_yet_measured"
    assert by_id["dolly-vn/viterbox"]["status"] == "excluded_from_product_path"
    assert by_id["ResembleAI/chatterbox"]["status"] == "excluded_for_vietnamese"
    assert manifest["benchmark"]["production_restarted"] is False
