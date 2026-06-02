"""
openai_tts.py
─────────────
OpenAI TTS engine for RedditVideoMakerBot.

Uses OpenAI's /v1/audio/speech endpoint (tts-1 or gpt-4o-mini-tts).
Same interface as every other TTS module:
    tts = OpenAITTS()
    tts.run(text, filepath="path/to/output.mp3")

Changes from original:
  - Strips [instruction] prefixes written by DeepSeek before sending to API
    (OpenAI would read "[speak slowly]" literally otherwise)
  - Graceful failure: warns instead of raising in __init__ if no API key
  - Silent fallback to gTTS if API call fails (never crashes pipeline)
  - Reads voice from OPENAI_VOICE_MAP per sentiment if no override in config
"""

import re
import random

import requests

from utils import settings
from utils.console import print_substep


# ─────────────────────────────────────────────────────────────────────────────
# Instruction prefix stripper
# ─────────────────────────────────────────────────────────────────────────────

def _strip_instruction(text: str) -> str:
    """
    Strips [instruction] prefix written by DeepSeek for voice delivery.

    DeepSeek writes these when voice_choice = "openai":
        "[speak slowly, emphasize STOPPED] I opened the door..."
        "[with dread] Something was wrong."

    OpenAI TTS does NOT parse these — it would read the brackets literally.
    The delivery cues (... CAPS ?!) in the rest of the sentence ARE read
    correctly by OpenAI, so we only strip the bracketed prefix.

    Examples:
        "[speak slowly] I opened the door..." → "I opened the door..."
        "I opened the door..."               → "I opened the door..."  (unchanged)
    """
    return re.sub(r'^\[[^\]]+\]\s*', '', text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI TTS engine
# ─────────────────────────────────────────────────────────────────────────────

class OpenAITTS:
    """
    OpenAI TTS engine. Same interface as GTTS, QwenTTS, elevenlabs:
        tts = OpenAITTS()
        tts.run(text, filepath="assets/temp/.../mp3/postaudio-0.mp3")

    Config keys read from settings.config["settings"]["tts"]:
        openai_api_key   : your OpenAI API key (required)
        openai_api_url   : base URL, default "https://api.openai.com/v1/"
        openai_model     : "tts-1" (recommended) or "tts-1-hd" or "gpt-4o-mini-tts"
        openai_voice_name: voice name — overridden per sentiment by OPENAI_VOICE_MAP
        random_voice     : if True, picks a random voice per sentence

    Voice selection priority:
        1. random_voice=True → random from available_voices
        2. OPENAI_VOICE_MAP[current_sentiment] → best voice for the mood
        3. openai_voice_name from config → manual override
        4. "onyx" → hardcoded fallback (best for storytelling)
    """

    max_chars: int = 4096  # OpenAI limit per API call

    def __init__(self):
        self.api_key = settings.config["settings"]["tts"].get("openai_api_key", "").strip()

        if not self.api_key:
            print_substep(
                "No OpenAI API key found in config. "
                "Set openai_api_key in [settings.tts]. "
                "Will fall back to gTTS.",
                style="yellow",
            )

        # Build full endpoint URL
        base_url = settings.config["settings"]["tts"].get(
            "openai_api_url", "https://api.openai.com/v1/"
        ).rstrip("/")
        self.api_url = f"{base_url}/audio/speech"

        self.available_voices = [
            "alloy", "ash", "coral", "echo",
            "fable", "onyx", "nova", "sage", "shimmer",
        ]

    def randomvoice(self) -> str:
        return random.choice(self.available_voices)

    def _get_voice(self, random_voice: bool) -> str:
        """
        Resolve which voice to use for this sentence.
        Priority: random → sentiment map → config → fallback.
        """
        if random_voice:
            return self.randomvoice()

        # Sentiment-aware voice — best voice for the current story mood
        try:
            from utils.sentiment_map import OPENAI_VOICE_MAP
            sentiment = settings.config["settings"].get("sentiment", "dramatic")
            if sentiment in OPENAI_VOICE_MAP:
                return OPENAI_VOICE_MAP[sentiment]
        except Exception:
            pass

        # Manual config override
        voice = settings.config["settings"]["tts"].get("openai_voice_name", "").strip().lower()
        if voice and voice in self.available_voices:
            return voice

        # Hardcoded fallback — onyx is the best storytelling voice
        return "onyx"

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface — called by engine_wrapper.py
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, text: str, filepath: str, random_voice: bool = False) -> None:
        """
        Synthesize text → MP3 at filepath.
        Strips [instruction] prefix before sending to API.
        Falls back to gTTS silently on any failure.
        Never raises. Never crashes the pipeline.
        """
        if not text or not text.strip():
            return

        if not self.api_key:
            self._fallback_gtts(text, filepath)
            return

        # Strip DeepSeek delivery instruction prefix
        # The punctuation cues (..., CAPS, ?!) in the rest of the text
        # are kept — OpenAI reads them naturally for emphasis
        clean_text = _strip_instruction(text)
        if not clean_text:
            return

        try:
            self._call_api(clean_text, filepath, random_voice)
        except Exception as e:
            print_substep(
                f"OpenAI TTS failed: {e}. Falling back to gTTS.",
                style="yellow",
            )
            self._fallback_gtts(text, filepath)

    # ─────────────────────────────────────────────────────────────────────────
    # API call
    # ─────────────────────────────────────────────────────────────────────────

    def _call_api(self, text: str, filepath: str, random_voice: bool) -> None:
        """Makes the OpenAI /v1/audio/speech API call and saves the MP3."""
        voice   = self._get_voice(random_voice)
        model   = settings.config["settings"]["tts"].get("openai_model", "tts-1")

        payload = {
            "model":           model,
            "voice":           voice,
            "input":           text,
            "response_format": "mp3",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI API error {response.status_code}: {response.text[:200]}"
            )

        with open(filepath, "wb") as f:
            f.write(response.content)

    # ─────────────────────────────────────────────────────────────────────────
    # Fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _fallback_gtts(self, text: str, filepath: str) -> None:
        """
        gTTS fallback. Strips [instruction] prefix before speaking.
        Never raises.
        """
        try:
            from gtts import gTTS
            clean = _strip_instruction(text)
            if not clean:
                return
            gTTS(text=clean, lang="en", slow=False).save(filepath)
            print_substep(f"gTTS fallback used for: {clean[:60]}", style="dim")
        except Exception as e:
            print_substep(f"gTTS fallback also failed: {e}", style="red")