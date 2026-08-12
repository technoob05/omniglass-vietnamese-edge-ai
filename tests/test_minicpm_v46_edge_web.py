from __future__ import annotations

import base64
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "minicpm_v46_edge_web.py"


def _load():
    spec = importlib.util.spec_from_file_location("minicpm_v46_edge_web", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_page_has_camera_upload_and_edge_truth_boundary():
    module = _load()
    assert "getUserMedia" in module.PAGE
    assert "Chọn ảnh" in module.PAGE
    assert "CPU ONLY" in module.PAGE
    assert "chưa phải tốc độ realtime" in module.PAGE


def test_decode_image_separates_supported_media_and_caps_size():
    module = _load()
    payload = base64.b64encode(b"jpeg-test").decode()
    raw, suffix = module._decode_image(f"data:image/jpeg;base64,{payload}")
    assert raw == b"jpeg-test"
    assert suffix == ".jpg"


def test_prompt_preserves_english_control_and_adds_vietnamese_only():
    module = _load()
    assert "Answer only" in module._prompt("What is this?", "en")
    vi = module._prompt("Đây là gì?", "vi")
    assert "BẮT BUỘC chỉ trả lời bằng tiếng Việt" in vi
    assert vi.endswith("Đây là gì?")


def test_inference_command_is_cpu_only_and_bounded():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"-ngl", "0"' in source
    assert '"--no-mmproj-offload"' in source
    assert 'proc = subprocess.run(' in source
    assert 'errors="replace"' in source
    assert "timeout=cfg.timeout" in source
    assert "MAX_BODY_BYTES" in source
