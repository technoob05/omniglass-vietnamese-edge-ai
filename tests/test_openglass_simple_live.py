import base64
import io
import json
import wave

import cv2
import numpy as np

from scripts.openglass_simple_live import SimpleVisionAgent


def make_wav(samples: np.ndarray, rate: int = 16000) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(samples.astype(np.int16).tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_capture_keeps_latest_frame():
    engine = SimpleVisionAgent("http://agent.invalid")
    frame = np.full((720, 1280, 3), 42, dtype=np.uint8)

    status = engine.capture(frame)

    assert "frame #1" in status
    snap, frame_id, captured_at = engine._snapshot()
    assert frame_id == 1
    assert captured_at > 0
    np.testing.assert_array_equal(snap, frame)


def test_ask_sends_latest_frame_then_decodes_vietnamese_audio():
    engine = SimpleVisionAgent("http://agent.invalid")
    current = np.zeros((24, 32, 3), dtype=np.uint8)
    samples = np.array([0, 1000, -1000], dtype=np.int16)
    calls = []

    def fake_post(endpoint, payload, timeout=60.0):
        calls.append((endpoint, payload, timeout))
        if endpoint == "analyze":
            return {"answer": "Trước mặt có một chiếc cốc."}
        assert endpoint == "speak"
        return {"audio_wav_base64": make_wav(samples)}

    engine._post = fake_post
    answer, audio, snapshot = engine.ask("Trước mặt có gì?", "webcam", current, None)

    assert answer == "Trước mặt có một chiếc cốc."
    assert audio[0] == 16000
    np.testing.assert_array_equal(audio[1], samples)
    assert snapshot.shape == (24, 32, 3)
    assert calls[0][0] == "analyze"
    assert calls[0][1]["image_jpeg_base64"]
    assert "Câu hỏi: Trước mặt có gì?" in calls[0][1]["prompt"]
    assert calls[1][0] == "speak"


def test_hello_is_answered_locally_in_vietnamese_without_vlm():
    engine = SimpleVisionAgent("http://agent.invalid")
    samples = np.array([0, 1], dtype=np.int16)
    calls = []

    def fake_post(endpoint, payload, timeout=60.0):
        calls.append(endpoint)
        assert endpoint == "speak"
        return {"audio_wav_base64": make_wav(samples)}

    engine._post = fake_post
    answer, audio, snapshot = engine.ask("Hello")

    assert answer.startswith("Xin chào!")
    assert len(answer) < 60
    assert calls == ["speak"]
    assert audio[0] == 16000
    assert snapshot is None


def test_question_uses_browser_frame_instead_of_stale_server_cache():
    engine = SimpleVisionAgent("http://agent.invalid")
    engine.capture(np.zeros((10, 10, 3), dtype=np.uint8))
    current = np.full((12, 16, 3), 77, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(current, cv2.COLOR_RGB2BGR))
    assert ok
    current_data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    samples = np.array([0, 1], dtype=np.int16)

    def fake_post(endpoint, payload, timeout=60.0):
        if endpoint == "analyze":
            return {"answer": "Tôi thấy khung hình hiện tại."}
        return {"audio_wav_base64": make_wav(samples)}

    engine._post = fake_post
    _, _, snapshot = engine.ask("Bạn thấy gì?", "webcam", current_data_url, None)

    assert snapshot.shape == current.shape
    assert abs(float(snapshot.mean()) - 77.0) < 2.0


def test_turn_finishes_with_text_when_tts_fails():
    engine = SimpleVisionAgent("http://agent.invalid")
    current = np.full((12, 16, 3), 55, dtype=np.uint8)

    def fake_post(endpoint, payload, timeout=60.0):
        if endpoint == "analyze":
            return {"answer": "Tôi thấy một chiếc cốc."}
        raise RuntimeError("TTS unavailable")

    engine._post = fake_post
    events = list(engine.ask_turn("Bạn thấy gì?", "webcam", current, None))

    assert len(events) == 2
    assert events[0][0].startswith("⏳")
    assert json.loads(events[0][2])["phase"] == "thinking"
    assert events[1][0] == "Tôi thấy một chiếc cốc."
    assert events[1][1].shape == current.shape
    final_event = json.loads(events[1][2])
    assert final_event["phase"] == "done"
    assert final_event["audio_payload"] == ""


def test_webcam_question_never_falls_back_to_global_cache():
    engine = SimpleVisionAgent("http://agent.invalid")
    engine.capture(np.full((10, 10, 3), 99, dtype=np.uint8))
    engine._post = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not call backend"))

    events = list(engine.ask_turn("Bạn thấy gì?", "webcam", None, None))

    assert len(events) == 1
    assert "đúng phiên webcam này" in events[0][0]
    assert events[0][1] is None
    final_event = json.loads(events[0][2])
    assert final_event["phase"] == "done"
    assert final_event["audio_payload"] == ""


def test_answer_is_hard_limited_for_realtime_tts():
    long_answer = " ".join(f"từ{i}" for i in range(45))
    concise = SimpleVisionAgent._concise_answer(long_answer)

    assert len(concise.split()) == 18
    assert concise.endswith(".")
