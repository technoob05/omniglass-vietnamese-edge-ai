import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "manifests" / "vietnamese_asr_candidate_benchmarks.json"
SCRIPT_PATH = ROOT / "scripts" / "evaluate_asr_candidate_manifest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_asr_candidate_manifest", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_candidate_manifest_is_pinned_and_honest():
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    contract = document["benchmark_contract"]
    assert contract["samples"] == 50
    assert contract["split"] == "test"
    assert len(contract["manifest_sha256"]) == 64
    candidates = {row["key"]: row for row in document["candidates"]}
    assert candidates["qwen3-asr-0.6b"]["license"] == "Apache-2.0"
    assert candidates["qwen3-asr-0.6b"]["supports_vietnamese"] is True
    assert candidates["qwen3-asr-0.6b"]["status"] == "verified_h100_same_manifest"
    assert candidates["qwen3-asr-0.6b"]["normalized_wer"] == 0.091808
    assert candidates["qwen3-asr-0.6b"]["normalized_cer"] == 0.047755
    assert candidates["qwen3-asr-0.6b"]["production_service_after_benchmark"] == "healthy"
    assert candidates["whisper-large-v3-turbo"]["license"] == "MIT"
    assert all(len(row["revision"]) == 40 for row in document["candidates"])
    assert "No candidate is called SOTA" in document["decision_rule"]["do_not_claim_sota"]
    assert all("verified" not in candidates[key]["qcs8550"].split()[0] for key in (
        "qwen3-asr-0.6b", "qwen3-asr-1.7b", "whisper-large-v3-turbo"
    ))


def test_normalization_preserves_vietnamese_diacritics():
    module = load_module()
    assert module.normalize_vi("  Đèn ở BÊN-TRÁI!  ") == "đèn ở bên trái"
    assert module.normalize_vi("Số 17, cửa số 2.") == "số 17 cửa số 2"


def test_levenshtein_and_percentile():
    module = load_module()
    assert module.levenshtein("mở đèn".split(), "mở cửa".split()) == 1
    assert module.percentile([0.1, 0.2, 0.3, 0.4], 0.95) == 0.4


def test_runner_fails_on_duplicate_sample_ids(tmp_path):
    module = load_module()
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"not decoded during validation")
    row = {
        "sample_id": "same",
        "audio_path": str(audio),
        "sentence_id": 1,
        "filename": audio.name,
        "raw_transcription": "xin chào",
    }
    try:
        module.validate_manifest({"samples": [row, dict(row)]}, minimum_samples=2)
    except RuntimeError as error:
        assert "duplicate" in str(error).lower()
    else:
        raise AssertionError("duplicate sample IDs must fail")
