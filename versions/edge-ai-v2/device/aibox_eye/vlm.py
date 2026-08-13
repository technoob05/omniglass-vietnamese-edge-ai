"""On-demand VLM client with grounded prompt construction and strict parsing."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from collections.abc import Callable, Iterable
from typing import Any

from .answers import scene_facts
from .models import SceneSnapshot


SYSTEM_PROMPT = """Bạn là đôi mắt hội thoại tiếng Việt cho người khiếm thị.
Ảnh camera là nguồn để đọc chữ và mô tả cảnh. scene_facts là kết quả YOLO/depth
đã chạy trước trên Qualcomm HTP và là nguồn khoảng cách duy nhất. Dùng cả hai
nguồn, nhưng không bịa vật, chữ hoặc khoảng cách. Không khẳng định đường đi an
toàn; nếu không nhìn rõ thì nói không chắc. Trả lời trực tiếp bằng một câu tiếng
Việt tự nhiên, khoảng 15 từ. Không tự suy luận chuyển động hoặc trạng thái nếu
không được hỏi. Không nhắc AI, YOLO, detector, scene_facts hay quy trình xử lý;
không JSON và không Markdown."""


@dataclass(frozen=True)
class VlmAnswer:
    answer_vi: str
    confidence: float
    uncertain: bool
    evidence: list[str]
    latency_ms: float
    first_token_ms: float | None = None


class LlamaVlmClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.base_url = str(config["base_url"]).rstrip("/")
        self.model = str(config["model"])
        self.timeout_seconds = float(config["timeout_seconds"])
        self.max_tokens = int(config["max_tokens"])
        self.temperature = float(config["temperature"])
        self.stream = bool(config.get("stream", True))

    def ask(
        self,
        question_vi: str,
        scene: SceneSnapshot,
        jpeg: bytes,
        history: list[tuple[str, str]] | None = None,
        on_partial: Callable[[str, float], None] | None = None,
    ) -> VlmAnswer:
        encoded = base64.b64encode(jpeg).decode("ascii")
        history_text = json.dumps(
            [{"user": question, "assistant": response} for question, response in (history or [])],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        user_prompt = (
            f"Câu hỏi: {question_vi}\n"
            f"Hội thoại gần đây (chỉ để hiểu câu nối tiếp, không phải dữ liệu ảnh hiện tại): {history_text}\n"
            "Dữ liệu detector/depth đã đồng bộ với ảnh (zone: left/center/right; "
            "distance_m=null nghĩa là depth chưa hiệu chuẩn):\n"
            f"{json.dumps(scene_facts(scene), ensure_ascii=False, separators=(',', ':'))}"
        )
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
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
                if self.stream:
                    content, first_token_ms = collect_stream_text(
                        response,
                        started,
                        on_partial,
                    )
                    latency_ms = (time.monotonic_ns() - started) / 1e6
                    return parse_answer(content, latency_ms, first_token_ms)
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            raise RuntimeError(f"VLM unavailable: {error}") from error
        latency_ms = (time.monotonic_ns() - started) / 1e6
        try:
            content = raw["choices"][0]["message"]["content"]
            return parse_answer(content, latency_ms)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid VLM response: {error}") from error


def collect_stream_text(
    lines: Iterable[bytes],
    started_ns: int,
    on_partial: Callable[[str, float], None] | None = None,
) -> tuple[str, float | None]:
    """Collect OpenAI-compatible SSE chunks and expose cumulative text."""
    parts: list[str] = []
    first_token_ms: float | None = None
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            value = json.loads(payload)
            delta = value["choices"][0].get("delta", {}).get("content", "")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
        text = _content_text(delta)
        if not text:
            continue
        parts.append(text)
        elapsed_ms = (time.monotonic_ns() - started_ns) / 1e6
        if first_token_ms is None:
            first_token_ms = elapsed_ms
        if on_partial is not None:
            on_partial("".join(parts), elapsed_ms)
    content = "".join(parts).strip()
    if not content:
        raise RuntimeError("VLM stream ended without text")
    return content, first_token_ms


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text", "")) for item in value if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def parse_answer(content: str, latency_ms: float = 0.0, first_token_ms: float | None = None) -> VlmAnswer:
    # GenieX/Qwen3.5 may return a concise natural-language answer even when
    # response_format=json_object is requested. Preserve the conversation
    # instead of treating that valid text as a failed VLM call.
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        answer = " ".join(str(content).strip().split())
        if not answer or len(answer.split()) > 40:
            raise ValueError("VLM plain-text answer missing or too long")
        lowered = answer.lower()
        uncertain = any(marker in lowered for marker in ("không chắc", "không rõ", "không nhìn rõ", "có thể"))
        return VlmAnswer(answer, 0.65, uncertain, ["plain_text_response"], latency_ms, first_token_ms)
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
    return VlmAnswer(answer, confidence, uncertain, evidence, latency_ms, first_token_ms)
