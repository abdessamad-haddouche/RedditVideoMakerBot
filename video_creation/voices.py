"""
video_creation/voices.py
────────────────────────
Thin dispatch layer between main.py and TTS/engine_wrapper.py.

Responsibility: read voice_choice from config, pick the right TTS module,
hand it to TTSEngine. That's it.

Changes vs original:
  - Added "qwen" and "qwen3" cases → Qwen3TTS (local, Apple Silicon)
  - Added "openai" lowercase alias (was "OpenAI" only — case-insensitive fix)
  - Everything else is identical to the original.
"""

from utils import settings
from utils.console import print_substep
from TTS.engine_wrapper import TTSEngine


def save_text_to_mp3(reddit_object) -> tuple:
    """
    Entry point called from main.py.
    Selects TTS engine based on voice_choice, runs the full TTS pipeline.

    Returns:
        (length: float, number_of_clips: int)
    """
    voice_choice = settings.config["settings"]["tts"]["voice_choice"].lower().strip()

    # ── Engine dispatch ───────────────────────────────────────────────────────
    # Adding a new engine: one elif here + one entry in TTS/__init__.py
    # ─────────────────────────────────────────────────────────────────────────

    if voice_choice == "elevenlabs":
        from TTS.elevenlabs import ElevenLabsTTS as tts_module

    elif voice_choice in ("streamlabspolly", "streamlabs"):
        from TTS.streamlabs_polly import StreamlabsPolly as tts_module

    elif voice_choice in ("awspolly", "aws"):
        from TTS.aws_polly import AWSPolly as tts_module

    elif voice_choice == "tiktok":
        from TTS.TikTok import TikTok as tts_module

    elif voice_choice in ("openai", "openaitts"):
        from TTS.openai_tts import OpenAITTS as tts_module

    elif voice_choice in ("pyttsx", "system"):
        from TTS.pyttsx import pyttsx as tts_module

    elif voice_choice == "googletranslate":
        from TTS.GTTS import GTTS as tts_module

    # ── NEW: local Qwen3-TTS (Apple Silicon / iMac M4) ───────────────────────
    elif voice_choice in ("qwen", "qwen3", "qwen3tts", "qwen_tts"):
        from TTS.qwen3_tts import Qwen3TTS as tts_module
    # ─────────────────────────────────────────────────────────────────────────

    else:
        # Unknown engine — warn and fall back to gTTS so the pipeline never
        # crashes on a typo in config.toml
        print_substep(
            f"Unknown voice_choice: '{voice_choice}'. Falling back to googletranslate.",
            style="yellow",
        )
        from TTS.GTTS import GTTS as tts_module

    # ── Run the engine ────────────────────────────────────────────────────────
    engine = TTSEngine(tts_module, reddit_object)
    return engine.run()