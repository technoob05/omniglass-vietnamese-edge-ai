import base64
import io
import threading
import wave

import numpy as np

from scripts.grounded_sam2_live_web import GroundedLiveEngine


class FakeTracker:
    def __init__(self):
        self.prompts = []
        self.last_mask_dict = FakeMaskState({1: "stale-landscape-mask"})
        self.track_dict = FakeMaskState({1: "stale-landscape-track"})
        self.objects_count = 7

    def set_prompt(self, prompt):
        self.prompts.append(prompt)


class FakeMaskState:
    def __init__(self, labels=None):
        self.labels = labels or {}


def build_engine():
    """Build the command router without loading Grounded-SAM-2 or a GPU."""
    engine = GroundedLiveEngine.__new__(GroundedLiveEngine)
    engine.lock = threading.Lock()
    engine.prompt = "person."
    engine.tracker = FakeTracker()
    engine.processed = 0
    engine.started_at = 0.0
    engine.agent_url = "http://agent.invalid"
    engine.latest_frame = np.zeros((12, 18, 3), dtype=np.uint8)
    engine.latest_objects = []
    engine.last_seen = {}
    engine.tracking_enabled = True
    engine.last_answer = ""
    return engine


def test_normalize_and_extract_ascii_target():
    assert GroundedLiveEngine.normalize_prompt(" red bottle ") == "red bottle."
    assert GroundedLiveEngine._extract_target("track red bottle") == "red bottle"
    assert GroundedLiveEngine._extract_target("watch chair!") == "chair"


def test_camera_frames_are_letterboxed_to_one_tracker_resolution():
    landscape = np.zeros((720, 1280, 3), dtype=np.uint8)
    portrait = np.zeros((1920, 1080, 3), dtype=np.uint8)

    normalized_landscape = GroundedLiveEngine.normalize_video_frame(landscape)
    normalized_portrait = GroundedLiveEngine.normalize_video_frame(portrait)

    assert normalized_landscape.shape == (720, 1280, 3)
    assert normalized_portrait.shape == (720, 1280, 3)
    assert normalized_landscape.flags.c_contiguous
    assert normalized_portrait.flags.c_contiguous


def test_normalize_video_frame_handles_1280_and_1920_orientation_changes():
    cases = [
        ((720, 1280), (0, 0, 1280, 720)),
        ((1080, 1920), (0, 0, 1280, 720)),
        ((1280, 720), (437, 0, 842, 720)),
        ((1920, 1080), (437, 0, 842, 720)),
    ]

    for (height, width), (left, top, right, bottom) in cases:
        frame = np.full((height, width, 3), (17, 91, 203), dtype=np.uint8)
        normalized = GroundedLiveEngine.normalize_video_frame(frame)

        assert normalized.shape == (720, 1280, 3)
        assert normalized.dtype == np.uint8
        assert normalized.flags.c_contiguous
        assert np.all(normalized[top:bottom, left:right] == (17, 91, 203))
        if left:
            assert np.all(normalized[:, :left] == 0)
            assert np.all(normalized[:, right:] == 0)


def test_prompt_change_discards_stale_masks_tracks_and_object_ids():
    engine = build_engine()
    old_last_masks = engine.tracker.last_mask_dict
    old_tracks = engine.tracker.track_dict

    _, prompt = engine.set_prompt("red bottle")

    assert prompt == "red bottle."
    assert engine.tracker.last_mask_dict is not old_last_masks
    assert engine.tracker.track_dict is not old_tracks
    assert engine.tracker.last_mask_dict.labels == {}
    assert engine.tracker.track_dict.labels == {}
    assert engine.tracker.objects_count == 0


def test_track_command_changes_tracker_prompt_without_network():
    engine = build_engine()
    engine._post_agent = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("track command must not call an agent service")
    )

    answer, prompt = engine.handle_command("track red bottle")

    assert prompt == "red bottle."
    assert engine.prompt == "red bottle."
    assert engine.tracker.prompts == ["red bottle."]
    assert "red bottle" in answer


def test_stop_and_help_do_not_require_camera_or_remote_service():
    engine = build_engine()
    engine.latest_frame = None
    engine._post_agent = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("local control command must not call an agent service")
    )

    stop_answer, _ = engine.handle_command("stop tracking")
    help_answer, _ = engine.handle_command("help")

    assert engine.tracking_enabled is False
    assert stop_answer
    assert "theo d" in help_answer.casefold()


def test_describe_routes_a_jpeg_snapshot_to_vlm_without_real_network():
    engine = build_engine()
    calls = []

    def fake_post(endpoint, payload, timeout=60.0):
        calls.append((endpoint, payload, timeout))
        return {"answer": "scene description"}

    engine._post_agent = fake_post
    answer, _ = engine.handle_command("describe scene")

    assert answer == "scene description"
    assert len(calls) == 1
    endpoint, payload, timeout = calls[0]
    assert endpoint == "analyze"
    assert timeout == 60.0
    assert payload["image_jpeg_base64"]
    assert isinstance(payload["prompt"], str)


def test_distance_routes_current_tracked_box_to_depth_service():
    engine = build_engine()
    box = [1.0, 2.0, 11.0, 10.0]
    engine.latest_objects = [{"label": "person", "bbox_xyxy": box, "seen_at": 1.0}]
    calls = []

    def fake_post(endpoint, payload, timeout=60.0):
        calls.append((endpoint, payload, timeout))
        return {"answer": "about two metres (uncalibrated)"}

    engine._post_agent = fake_post
    answer, _ = engine.handle_command("distance")

    assert "uncalibrated" in answer
    endpoint, payload, timeout = calls[0]
    assert endpoint == "distance"
    assert payload["bbox_xyxy"] == box
    assert payload["target_name"] == "person"
    assert timeout == 45


def test_last_seen_is_local_and_reports_horizontal_region():
    engine = build_engine()
    engine.latest_objects = [
        {"label": "person", "bbox_xyxy": [0.0, 1.0, 4.0, 8.0], "seen_at": 1.0}
    ]
    engine._post_agent = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("last-seen lookup must not call an agent service")
    )

    answer, _ = engine.handle_command("last seen")

    assert "person" in answer


def test_remote_failure_is_returned_as_a_user_visible_error():
    engine = build_engine()

    def fail(*args, **kwargs):
        raise TimeoutError("agent timed out")

    engine._post_agent = fail
    answer, prompt = engine.handle_command("describe scene")

    assert "agent timed out" in answer
    assert prompt == "person."


def test_ambiguous_command_uses_whitelisted_llm_plan_then_typed_tool():
    engine = build_engine()
    calls = []

    def fake_post(endpoint, payload, timeout=60.0):
        calls.append((endpoint, payload, timeout))
        assert endpoint == "plan"
        return {"action": "track", "target": "red cup", "question": None}

    engine._post_agent = fake_post
    answer, prompt = engine.handle_command("please keep the red cup in view")

    assert calls[0][0] == "plan"
    assert prompt == "red cup."
    assert engine.tracker.prompts == ["red cup."]
    assert "red cup" in answer


def test_h100_tts_wav_is_decoded_for_gradio_audio():
    engine = build_engine()
    samples = np.array([0, 1000, -1000, 2000], dtype=np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())

    def fake_post(endpoint, payload, timeout=60.0):
        assert endpoint == "speak"
        assert payload == {"text": "Xin chào"}
        assert timeout == 45
        return {"audio_wav_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}

    engine._post_agent = fake_post
    sample_rate, decoded = engine._speak_audio("Xin chào")

    assert sample_rate == 16000
    np.testing.assert_array_equal(decoded, samples)


def test_command_returns_answer_prompt_and_neural_audio():
    engine = build_engine()
    expected_audio = (16000, np.array([0, 1], dtype=np.int16))
    engine._speak_audio = lambda text: expected_audio

    answer, prompt, audio = engine.handle_command_with_audio("help")

    assert "theo d" in answer.casefold()
    assert prompt == "person."
    assert audio is expected_audio
