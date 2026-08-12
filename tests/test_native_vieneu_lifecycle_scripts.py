from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_native_start_preserves_mms_and_manages_vieneu() -> None:
    source = (ROOT / "scripts/start_native_minicpmo_h100.sh").read_text(encoding="utf-8")
    assert "native_vietnamese_tts_service.py" in source
    assert "--host 127.0.0.1 --port 18781" in source
    assert "native_vieneu_tts_service.py" in source
    assert "--host 127.0.0.1 --port 18782" in source
    assert 'OPENGLASS_VIENEU_VENV' in source
    assert 'vi_tts_vieneu.pid' in source
    assert 'vi_tts_vieneu.log' in source
    assert 'health["backend"] == "pytorch"' in source
    assert "continuing with MMS-TTS fallback" in source


def test_native_stop_owns_both_tts_processes() -> None:
    source = (ROOT / "scripts/stop_native_minicpmo_h100.sh").read_text(encoding="utf-8")
    assert "for name in vi_asr vi_tts_vieneu vi_tts worker backend gateway" in source
    assert "native_vieneu_tts_service.py" in source
    assert "native_vietnamese_tts_service.py" in source


def test_native_lifecycle_manages_phowhisper_per_runtime() -> None:
    start = (ROOT / "scripts/start_native_minicpmo_h100.sh").read_text(encoding="utf-8")
    stop = (ROOT / "scripts/stop_native_minicpmo_h100.sh").read_text(encoding="utf-8")
    assert "phowhisper_asr_service.py" in start
    assert "--host 127.0.0.1 --port 18783" in start
    assert "vi_asr.pid" in start
    assert "vi_asr_phowhisper.log" in start
    assert "phowhisper_asr_service.py" in stop


def test_vieneu_shell_scripts_parse() -> None:
    scripts = [
        ROOT / "scripts/bootstrap_native_vieneu_gpu_h100.sh",
        ROOT / "scripts/start_native_vieneu_tts_h100.sh",
        ROOT / "scripts/start_native_minicpmo_h100.sh",
        ROOT / "scripts/stop_native_minicpmo_h100.sh",
        ROOT / "scripts/start_phowhisper_asr_h100.sh",
        ROOT / "scripts/stop_phowhisper_asr_h100.sh",
        ROOT / "scripts/install_native_vietnamese_profile_h100.sh",
    ]
    result = subprocess.run(
        ["bash", "-n", *(path.relative_to(ROOT).as_posix() for path in scripts)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
