from pathlib import Path

import pytest

from scripts.vlm_agent_service import contains_cjk, looks_vietnamese, strip_cjk


def test_vietnamese_answer_has_no_cjk():
    answer = "Con hà mã đang nổi trên mặt nước."
    assert not contains_cjk(answer)
    assert strip_cjk(answer) == answer
    assert looks_vietnamese(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "This image shows a horse in the water.",
        "There is a person standing on the left side.",
        "A red cup is visible on the table.",
    ],
)
def test_english_only_answers_fail_vietnamese_validation(answer):
    assert not contains_cjk(answer)
    assert not looks_vietnamese(answer)


@pytest.mark.parametrize("answer", ["Xin chào!", "Chào bạn!", "Cảm ơn bạn."])
def test_short_vietnamese_responses_pass_language_validation(answer):
    assert looks_vietnamese(answer)


def test_cjk_leak_is_detected_and_removed_without_damaging_vietnamese():
    leaked = "Con hà mã đang nổi trên 水面 ở giữa khung hình."
    cleaned = strip_cjk(leaked)

    assert contains_cjk(leaked)
    assert not contains_cjk(cleaned)
    assert cleaned == "Con hà mã đang nổi trên ở giữa khung hình."


@pytest.mark.parametrize(
    "leaked",
    [
        "Có một vật ở 水面.",       # Han / Chinese
        "Có chữ カメラ trên hộp.",  # Katakana / Japanese
        "Có chữ ひらがな trên hộp.",  # Hiragana / Japanese
        "Có chữ 카메라 trên hộp.",  # Hangul / Korean
        "Có chữ ㄱ trên hộp.",     # Hangul compatibility Jamo
        "Có chữ ᄀ trên hộp.",     # Hangul Jamo
    ],
)
def test_all_disallowed_east_asian_scripts_are_detected_and_removed(leaked):
    """The Vietnamese-only contract must also cover Korean, not only Han/Japanese."""
    assert contains_cjk(leaked)
    assert not contains_cjk(strip_cjk(leaked))


def test_vietnamese_diacritics_are_preserved_by_language_filter():
    answer = "Ở giữa có một người; phía bên phải có chiếc ghế màu đỏ."
    assert strip_cjk(answer) == answer


def test_prompts_and_browser_voice_contract_are_utf8_vietnamese():
    """Catch accidental mojibake and regressions in the browser language settings."""
    root = Path(__file__).resolve().parents[1]
    service = (root / "scripts" / "vlm_agent_service.py").read_text(encoding="utf-8")
    web = (root / "scripts" / "grounded_sam2_live_web.py").read_text(encoding="utf-8")

    assert "Chỉ trả lời bằng tiếng Việt" in service
    assert "không dùng chữ Hán" in service
    assert "recognition.lang='vi-VN'" in web
    assert "u.lang='vi-VN'" in web
    assert "getVoices()" in web
    assert "toLowerCase()==='vi-vn'" in web
    assert ".startsWith('vi')" in web
    assert "Tôi chưa thể diễn đạt kết quả hoàn toàn bằng tiếng Việt" in service

    for mojibake_marker in ("Báº", "Ä‘", "Ã´", "â€"):
        assert mojibake_marker not in service
        assert mojibake_marker not in web
