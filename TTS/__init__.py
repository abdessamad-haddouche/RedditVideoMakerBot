"""
TTS/__init__.py
───────────────
Engine registry for RedditVideoMakerBot.

This module is the single source of truth for which voice_choice string
maps to which TTS class. Adding a new engine = one line here.

Usage (from video_creation/voices.py):
    from TTS import get_engine_class
    EngineClass = get_engine_class(voice_choice)

Or the legacy dict access pattern (backward compatible):
    from TTS import TTSEngineMap
    EngineClass = TTSEngineMap[voice_choice]
"""

# ─────────────────────────────────────────────────────────────────────────────
# Engine imports — each in its own try/except so a broken engine never
# prevents the others from loading.
# ─────────────────────────────────────────────────────────────────────────────

def _safe_import(module_path: str, class_name: str):
    """Import a TTS engine class safely. Returns None if import fails."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Registry — voice_choice value → engine class
# To add a new engine: add one entry here. Nothing else changes.
# ─────────────────────────────────────────────────────────────────────────────

TTSEngineMap = {
    # ── Existing engines (unchanged) ─────────────────────────────────────────
    "googletranslate":   lambda: _safe_import("TTS.GTTS",              "GTTS"),
    "tiktok":            lambda: _safe_import("TTS.TikTok",            "TikTok"),
    "elevenlabs":        lambda: _safe_import("TTS.elevenlabs",        "TikTok"),
    "streamlabspolly":   lambda: _safe_import("TTS.streamlabs_polly",  "StreamlabsPolly"),
    "awspolly":          lambda: _safe_import("TTS.aws_polly",         "AWSPolly"),
    "pyttsx":            lambda: _safe_import("TTS.pyttsx",            "pyttsx"),
    "openai":            lambda: _safe_import("TTS.openai_tts",        "OpenAITTS"),

    # ── NEW: local Qwen3-TTS on Apple Silicon ─────────────────────────────────
    "qwen":              lambda: _safe_import("TTS.qwen3_tts",         "Qwen3TTS"),
    "qwen3":             lambda: _safe_import("TTS.qwen3_tts",         "Qwen3TTS"),  # alias
}


def get_engine_class(voice_choice: str):
    """
    Returns the TTS engine class for the given voice_choice string.
    Case-insensitive. Returns None if not found.

    Args:
        voice_choice: value from config["settings"]["tts"]["voice_choice"]

    Returns:
        TTS engine class, or None if not registered / import failed.
    """
    key = voice_choice.lower().strip()
    factory = TTSEngineMap.get(key)
    if factory is None:
        return None
    return factory()


__all__ = ["TTSEngineMap", "get_engine_class"]