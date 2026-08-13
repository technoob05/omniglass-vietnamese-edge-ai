"""On-demand VLM client with grounded prompt construction and strict parsing."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .answers import scene_facts
from .models import SceneSnapshot


SYSTEM_PROMPT = """Bạn là đôi mắt hội thoại tiếng Việt cho người khiếm thị.
Ảnh camera là nguồn để đọc chữ và mô tả cảnh. scene_facts là kết quả YOLO/depth
đã chạy trước trên Qualcomm HTP và là nguồn khoảng cách duy nhất. Dùng cả hai
nguồn, nhưng không bịa vật, chữ hoặc khoảng cách. Không khẳng định đường đi an
toàn; nếu không nhìn rõ thì nói không chắc. Trả lời trực tiếp bằng một câu tiếng
Việt tự nhiên, tối đa 30 từ; không JSON, không Markdown, không giải thích quy trình."""


@dataclass(frozen=True)
class VlmAnswer:
    answer_vi: str
    confidence: float
    uncertain: bool
    evidence: list[str]
    latency_ms: float


class LlamaVlmClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.base_url = str(config["base_url"]).rstrip("/")
        self.model = str(config["model"])
        self.timeout_seconds = float(config["timeout_seconds"])
        self.max_tokens = int(config["max_tokens"])
        self.temperature = float(config["temperature"])

    def ask(self, question_vi: str, scene: SceneSnapshot, jpeg: bytes) -> VlmAnswer:
        encoded = base64.b64encode(jpeg).decode("ascii")
        user_prompt = (
            f"Câu hỏi: {question_vi}\n"
            "Dữ liệu detector/depth đã đồng bộ với ảnh (zone: left/center/right; "
            "distance_m=null nghĩa là depth chưa hiệu chuẩn):\n"
            f"{json.dumps(scene_facts(scene), ensure_ascii=False, separators=(',', ':'))}"
        )
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ]},
            ],
        }
        started = time.monotonic_ns()
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            raise RuntimeError(f"VLM unavailable: {error}") from error
        latency_ms = (time.monotonic_ns() - started) / 1e6
        try:
            content = raw["choices"][0]["message"]["content"]
            return parse_answer(content, latency_ms)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid VLM response: {error}") from error


def parse_answer(content: str, latency_ms: float = 0.0) -> VlmAnswer:
    # GenieX/Qwen3.5 may return a concise natural-language answer even when
    # response_format=json_object is requested. Preserve the conversation
    # instead of treating that valid text as a failed VLM call.
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        answer = " ".join(str(content).strip().split())
        if not answer or len(answer.split()) > 40:
            raise ValueError("VLM plain-text answer missing or too long")
        return VlmAnswer(answer, 0.65, True, ["plain_text_response"], latency_ms)
    if not isinstance(value, dict):
        raise ValueError("VLM output is not an object")
    answer = str(value.get("answer_vi", "")).strip()
    if not answer or len(answer.split()) > 30:
        raise ValueError("VLM answer missing or too long")
    confidence = max(0.0, min(1.0, float(value.get("confidence", 0.0))))
    uncertain = bool(value.get("uncertain", confidence < 0.5))
    evidence_raw = value.get("evidence", [])
    evidence = [str(item)[:80] for item in evidence_raw] if isinstance(evidence_raw, list) else []
    # The VLM never controls risk; nevertheless reject fabricated metric claims
    # when no calibrated depth was supplied in its input.
    return VlmAnswer(answer, confidence, uncertain, evidence, latency_ms)
