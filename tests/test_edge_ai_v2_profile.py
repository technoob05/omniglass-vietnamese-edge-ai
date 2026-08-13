import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
PROFILE = ROOT / "versions" / "edge-ai-v2" / "device" / "config" / "qwen35-production.json"


def test_qwen35_profile_is_complete_and_local() -> None:
    config = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert config["perception"]["base_url"] == "http://127.0.0.1:8080"
    assert config["vlm"]["base_url"] == "http://127.0.0.1:18181"
    assert config["vlm"]["model"] == "local/Qwen3.5-2B-GGUF:Q4_0"
    assert config["vlm"]["admission_wait_seconds"] <= 20.0
    assert config["vlm"]["max_image_width"] == 512
    assert config["vlm"]["stream"] is True
    assert config["stt"]["backend"] == "whisper_cli_cpu"
    assert config["tts"]["backend"] == "vieneu_onnx"
    assert config["tts"]["voice"] == "Phạm Tuyên"
    assert config["tts"]["tempo"] == 1.5
    assert config["audio"]["half_duplex"] is True


def test_snapshot_documents_the_end_to_end_contract() -> None:
    text = (ROOT / "versions" / "edge-ai-v2" / "README.md").read_text(encoding="utf-8")
    for marker in ("Whisper.cpp", "Qwen3.5-2B-GGUF:Q4_0", "VieNeu", "Hexagon HTP"):
        assert marker in text


def test_qwen_plain_text_is_a_supported_answer() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "versions" / "edge-ai-v2" / "device"))
    from aibox_eye.vlm import parse_answer

    answer = parse_answer("Có một người đứng bên phải.", 123.0)
    assert answer.answer_vi == "Có một người đứng bên phải."
    assert answer.uncertain is False


def test_vlm_frame_is_resized_for_edge_inference() -> None:
    import sys
    import cv2
    import numpy as np
    sys.path.insert(0, str(ROOT / "versions" / "edge-ai-v2" / "device"))
    from aibox_eye.keyframes import prepare_for_vlm

    ok, jpeg = cv2.imencode(".jpg", np.zeros((720, 1280, 3), dtype=np.uint8))
    assert ok
    payload, metadata = prepare_for_vlm(jpeg.tobytes(), max_width=512)
    assert payload.startswith(b"\xff\xd8")
    assert metadata["width"] == 512
    assert metadata["height"] == 288


def test_full_demo_contains_box_camera_and_hold_to_talk_pipeline() -> None:
    text = (ROOT / "versions" / "edge-ai-v2" / "device" / "aibox_eye" / "server.py").read_text(encoding="utf-8")
    for marker in ("stream.mjpg", "/push-to-talk/start", "/push-to-talk/stop", "/stt/test/stop", "Speech-to-Text Lab", "Whisper STT", "audio_seconds", "real_time_factor", "/tts/speak", "/tts/stream", "AudioContext", "pcm_s16le_base64", "playbackRate.value=1.5", "currentTime+.35", "speechSynthesis", "pcSpeaker", "timing", "TTS tới cụm âm đầu", "Từ bấm hỏi → bắt đầu nghe", "RTF", "Detector realtime", "loa ALSA", "Qwen3.5 2B VL"):
        assert marker in text


def test_explicit_visual_request_wins_over_camera_status_keyword() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "versions" / "edge-ai-v2" / "device"))
    from aibox_eye.intents import route

    intent = route("Hãy đọc chữ và mô tả ảnh camera hiện tại")
    assert intent.name == "vlm"
    assert intent.requires_vlm is True


def test_stream_parser_collects_openai_sse_and_first_phrase() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "versions" / "edge-ai-v2" / "device"))
    from aibox_eye.orchestrator import _speakable_prefix
    from aibox_eye.vlm import collect_stream_text

    chunks = [
        b'data: {"choices":[{"delta":{"content":"Toi thay mot "}}]}\n',
        b'data: {"choices":[{"delta":{"content":"nguoi dung phia truoc."}}]}\n',
        b'data: [DONE]\n',
    ]
    partials = []
    text, first_token_ms = collect_stream_text(chunks, 0, lambda value, _ms: partials.append(value))
    assert text == "Toi thay mot nguoi dung phia truoc."
    assert first_token_ms is not None
    assert partials[-1] == text
    assert _speakable_prefix("mot hai ba bon nam sau bay tam chin muoi", 9) == "mot hai ba bon nam sau bay tam chin"


def test_scene_context_never_exposes_uncalibrated_corridor_metres() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "versions" / "edge-ai-v2" / "device"))
    from aibox_eye.answers import scene_facts
    from aibox_eye.models import SceneSnapshot, WalkingCorridor

    scene = SceneSnapshot(7, 1, 2, 3.0, True, True, False, walking_corridor=WalkingCorridor(1.2, 1.3, 1.1, 1.0, True))
    facts = scene_facts(scene)
    assert facts["frame_id"] == 7
    assert facts["walking_corridor"]["center_m"] is None


def test_overlapping_detector_boxes_are_deduplicated_for_qwen_context() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "versions" / "edge-ai-v2" / "device"))
    from aibox_eye.answers import _intersection_over_smaller

    assert _intersection_over_smaller((10, 10, 40, 60), (10, 10, 80, 100)) == 1.0
    assert _intersection_over_smaller((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_vieneu_text_is_localized_and_chunked_for_clear_speech() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "versions" / "edge-ai-v2" / "device"))
    from aibox_eye.tts import normalize_vi_text, phrase_chunks

    spoken = normalize_vi_text("Laptop chạy 24 FPS, NPU 75°C và USB ổn định.")
    assert spoken == "máy tính xách tay chạy 24 khung hình mỗi giây, en pi diu 75 độ C và u ét bê ổn định."
    chunks = phrase_chunks("một hai ba bốn năm sáu bảy tám chín mười mười một mười hai mười ba", maximum_words=12)
    assert chunks[0].endswith(",")
    assert len(chunks) == 2


def test_tts_cache_key_includes_voice_tempo_and_normalized_text() -> None:
    text = (ROOT / "versions" / "edge-ai-v2" / "device" / "aibox_eye" / "tts.py").read_text(encoding="utf-8")
    for marker in ("tts-cache", "cache_hits", "cache_misses", "self.config['voice']", "self.config['tempo']"):
        assert marker in text


def test_vieneu_phrase_stream_is_pcm16_at_48khz() -> None:
    import sys
    import threading
    import numpy as np
    sys.path.insert(0, str(ROOT / "versions" / "edge-ai-v2" / "device"))
    from aibox_eye.tts import VieneuTtsBackend

    class FakeEngine:
        sample_rate = 48_000

        def infer(self, text, voice, style, temperature, top_k, apply_watermark):
            assert text == "Xin chào."
            assert voice == "Phạm Tuyên"
            assert style == "tu_nhien"
            assert temperature == 0.4
            assert top_k == 25
            assert apply_watermark is True
            return np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)

    backend = object.__new__(VieneuTtsBackend)
    backend.config = {"voice": "Phạm Tuyên"}
    backend._engine = FakeEngine()
    backend._engine_lock = threading.RLock()
    chunks = list(backend.stream_pcm("Xin chào."))
    assert len(chunks) == 1
    values = np.frombuffer(chunks[0], dtype="<i2")
    assert values.tolist() == [-32767, 0, 32767]
