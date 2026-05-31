"""
whisper_aligner.py
──────────────────
Word-level timestamp extraction using WhisperX.

This module runs after each TTS audio file is saved.
It produces a word-level timestamp JSON for every postaudio-{i}.mp3.

Output format (postaudio-{i}_words.json):
[
  {"word": "I",    "start": 0.00, "end": 0.18},
  {"word": "told", "start": 0.18, "end": 0.42},
  ...
]

WhisperX is used because:
  - Works with ANY TTS engine (Google, OpenAI, ElevenLabs, etc.)
  - Free, runs locally, no API cost
  - Word-level accuracy (not sentence-level)
  - Fast on CPU for short audio clips

If WhisperX is not installed or fails for any reason,
this module returns None and the system falls back to
time_fraction-based sync (single/multi mode).
No crashes, no interruptions.
"""

import json
import os
from typing import List, Optional

from utils.console import print_substep


# ── WhisperX model is loaded once and reused across all audio files ───────────
# Loading is expensive (~2-3s). We cache it as a module-level singleton.
_whisper_model = None
_whisper_model_lang = None


def _get_model(language: str = "en"):
    """
    Lazy-load WhisperX model. Loaded once per run, reused for all audio files.
    Returns None if WhisperX is not installed.
    """
    global _whisper_model, _whisper_model_lang

    if _whisper_model is not None and _whisper_model_lang == language:
        return _whisper_model

    try:
        import whisperx
        print_substep("Loading WhisperX model (first run only)...", style="bold blue")
        _whisper_model = whisperx.load_model(
            "base",          # small enough for CPU, accurate enough for TTS
            device="cpu",
            compute_type="int8",
            language=language,
        )
        _whisper_model_lang = language
        return _whisper_model
    except ImportError:
        return None
    except Exception as e:
        print_substep(f"WhisperX model load failed: {e}", style="yellow")
        return None


def align_audio(audio_path: str, language: str = "en") -> Optional[List[dict]]:
    """
    Run WhisperX on a single audio file and return word-level timestamps.

    Parameters
    ----------
    audio_path : str
        Path to the .mp3 file to align.
    language : str
        Language code (default: "en"). Matches TTS language.

    Returns
    -------
    Optional[List[dict]]
        List of {"word": str, "start": float, "end": float} dicts.
        Returns None if WhisperX is unavailable or alignment fails.
    """
    try:
        import whisperx

        model = _get_model(language)
        if model is None:
            return None

        # Transcribe + align
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=4)

        # Align to get word-level timestamps
        align_model, metadata = whisperx.load_align_model(
            language_code=language,
            device="cpu",
        )
        aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device="cpu",
            return_char_alignments=False,
        )

        # Flatten all words across all segments
        words = []
        for segment in aligned.get("word_segments", []):
            word  = segment.get("word", "").strip()
            start = segment.get("start")
            end   = segment.get("end")
            if word and start is not None and end is not None:
                words.append({
                    "word":  word,
                    "start": round(float(start), 3),
                    "end":   round(float(end),   3),
                })

        return words if words else None

    except Exception as e:
        print_substep(f"WhisperX alignment failed for {audio_path}: {e}", style="yellow")
        return None


def align_and_save(audio_path: str, language: str = "en") -> Optional[str]:
    """
    Align audio and save word timestamps as a JSON file next to the audio.

    Parameters
    ----------
    audio_path : str
        e.g. "assets/temp/abc123/mp3/postaudio-0.mp3"
    language : str
        Language code.

    Returns
    -------
    Optional[str]
        Path to saved JSON file, or None if alignment failed.
    """
    words = align_audio(audio_path, language)

    if words is None:
        return None

    json_path = audio_path.replace(".mp3", "_words.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(words, f, indent=2, ensure_ascii=False)

    return json_path


def load_word_timestamps(audio_path: str) -> Optional[List[dict]]:
    """
    Load previously saved word timestamps for an audio file.
    Returns None if the file doesn't exist.
    """
    json_path = audio_path.replace(".mp3", "_words.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)