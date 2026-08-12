import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "minicpmo_vi_lora_smoke.json"
LAUNCHER = ROOT / "scripts" / "run_minicpmo_vi_lora_smoke.sh"
UPSTREAM = ROOT / "upstream" / "ms-swift"


def test_manifest_and_launcher_keep_talker_out_of_training():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    scope = manifest["train_scope"]
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert manifest["status"] == "configuration_verified_not_trained"
    assert scope == {
        "init_tts": False,
        "init_audio": True,
        "use_audio_in_video": False,
        "tuner_type": "lora",
        "target_modules": "all-linear",
        "freeze_llm": False,
        "freeze_vit": True,
        "freeze_aligner": True,
        "effective_trainable_scope": "LoRA adapters on the llm component only",
        "explicitly_not_trained": [
            "vpm vision tower",
            "apm audio tower",
            "resampler aligner",
            "tts/talker and speech decoder",
        ],
    }
    for required in (
        "INIT_TTS=false",
        "INIT_AUDIO=true",
        "USE_AUDIO_IN_VIDEO=false",
        "--freeze_llm false",
        "--freeze_vit true",
        "--freeze_aligner true",
        "--model_revision \"${MODEL_REVISION}\"",
    ):
        assert required in launcher


def test_launcher_is_fail_closed_for_production_gpu():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert manifest["hardware_guardrail"]["refuse_if_compute_process_exists"] is True
    assert "TRAIN_GPU is required" in launcher
    assert "Set WORK_ROOT to an isolated private training directory" in launcher
    assert "compute processes already exist" in launcher
    assert "MIN_FREE_MIB:-36000" in launcher
    assert "kill " not in launcher
    assert "pkill" not in launcher
    assert "systemctl" not in launcher


def test_pinned_upstream_contract_if_checkout_is_present():
    model_file = UPSTREAM / "swift" / "model" / "models" / "minicpm.py"
    arch_file = UPSTREAM / "swift" / "model" / "model_arch.py"
    template_file = UPSTREAM / "swift" / "template" / "templates" / "minicpm.py"
    if not model_file.is_file():
        return

    model = model_file.read_text(encoding="utf-8")
    arch = arch_file.read_text(encoding="utf-8")
    template = template_file.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert "OpenBMB/MiniCPM-o-4_5" in model
    assert "transformers==4.51.3" in model
    assert "minicpmo-utils==1.0.6" in model
    assert "config.init_tts" in model and "'false'" in model
    assert "config.init_audio" in model and "'true'" in model
    assert "language_model='llm'" in arch
    assert "aligner='resampler'" in arch
    assert "vision_tower=['vpm', 'apm']" in arch
    assert "MLLMTemplateType.minicpmo4_5" in template
    assert manifest["upstream"]["ms_swift_revision"] == "ca937fbaf8e0c3dc4ea34358889430e36475463b"
