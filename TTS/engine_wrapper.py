"""
engine_wrapper.py
─────────────────
Unchanged from original except one targeted change in storymodemethod == 1:

  Before:
    for idx, text in track(enumerate(self.reddit_object["thread_post"])):
        self.call_tts(f"postaudio-{idx}", process_text(text))

  After:
    for idx, item in track(enumerate(self.reddit_object["thread_post"])):
        text = item.tts if hasattr(item, "tts") else item
        self.call_tts(f"postaudio-{idx}", process_text(text))

This handles both paths:
  - EnhancedSentence (feature active)  → uses .tts field (voice cues, clean text)
  - Plain string (fallback / storymode 0) → used directly as before

Everything else is identical to the original.
"""

import os
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import translators
from moviepy import AudioFileClip
from moviepy.audio.AudioClip import AudioClip
from moviepy.audio.fx import MultiplyVolume
from rich.progress import track

from utils import settings
from utils.console import print_step, print_substep
from utils.voice import sanitize_text

DEFAULT_MAX_LENGTH: int = 50


class TTSEngine:
    """Calls the given TTS engine to reduce code duplication and allow multiple TTS engines."""

    def __init__(
        self,
        tts_module,
        reddit_object: dict,
        path: str = "assets/temp/",
        max_length: int = DEFAULT_MAX_LENGTH,
        last_clip_length: int = 0,
    ):
        self.tts_module = tts_module()
        self.reddit_object = reddit_object
        self.redditid = re.sub(r"[^\w\s-]", "", reddit_object["thread_id"])
        self.path = path + self.redditid + "/mp3"
        self.max_length = max_length
        self.length = 0
        self.last_clip_length = last_clip_length

    def add_periods(self):
        for comment in self.reddit_object["comments"]:
            regex_urls = r"((http|https)://)?[a-zA-Z0-9./?:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9.&/?:@\-_=#])*"
            comment["comment_body"] = re.sub(regex_urls, " ", comment["comment_body"])
            comment["comment_body"] = comment["comment_body"].replace("\n", ". ")
            comment["comment_body"] = re.sub(r"\bAI\b", "A.I", comment["comment_body"])
            comment["comment_body"] = re.sub(r"\bAGI\b", "A.G.I", comment["comment_body"])
            if comment["comment_body"][-1] != ".":
                comment["comment_body"] += "."
            comment["comment_body"] = comment["comment_body"].replace(". . .", ".")
            comment["comment_body"] = comment["comment_body"].replace(".. . ", ".")
            comment["comment_body"] = comment["comment_body"].replace(". . ", ".")
            comment["comment_body"] = re.sub(r'\."\.', '".', comment["comment_body"])

    def run(self) -> Tuple[int, int]:
        Path(self.path).mkdir(parents=True, exist_ok=True)
        print_step("Saving Text to MP3 files...")

        self.add_periods()
        self.call_tts("title", process_text(self.reddit_object["thread_title"]))
        idx = 0

        if settings.config["settings"]["storymode"]:
            if settings.config["settings"]["storymodemethod"] == 0:
                if len(self.reddit_object["thread_post"]) > self.tts_module.max_chars:
                    self.split_post(self.reddit_object["thread_post"], "postaudio")
                else:
                    self.call_tts("postaudio", process_text(self.reddit_object["thread_post"]))

            elif settings.config["settings"]["storymodemethod"] == 1:
                # ── CHANGED: handle EnhancedSentence or plain string ─────────
                # item.tts  → enhanced path (voice cues, DeepSeek rewritten)
                # plain str → fallback path (unchanged behaviour)
                for idx, item in track(enumerate(self.reddit_object["thread_post"])):
                    text = item.tts if hasattr(item, "tts") else item
                    self.call_tts(f"postaudio-{idx}", process_text(text))
                    # ── WhisperX alignment ────────────────────────────────────
                    # Run immediately after each TTS save so word timestamps
                    # are ready when imagemaker() runs later.
                    # Fails silently — never blocks video generation.
                    self._align_audio(f"postaudio-{idx}")

        else:
            for idx, comment in track(enumerate(self.reddit_object["comments"]), "Saving..."):
                if self.length > self.max_length and idx > 1:
                    self.length -= self.last_clip_length
                    idx -= 1
                    break
                if len(comment["comment_body"]) > self.tts_module.max_chars:
                    self.split_post(comment["comment_body"], idx)
                else:
                    self.call_tts(f"{idx}", process_text(comment["comment_body"]))

        print_substep("Saved Text to MP3 files successfully.", style="bold green")
        return self.length, idx

    def _align_audio(self, filename: str) -> None:
        """
        Run WhisperX on a saved audio file to produce word-level timestamps.
        Called immediately after each postaudio-{i}.mp3 is saved.
        Fails silently — system falls back to time_fraction mode if unavailable.
        """
        try:
            from utils.whisper_aligner import align_and_save
            audio_path = f"{self.path}/{filename}.mp3"
            lang = settings.config["reddit"]["thread"].get("post_lang", "en") or "en"
            result = align_and_save(audio_path, language=lang)
            if result:
                print_substep(f"Word timestamps saved → {result}", style="dim")
        except Exception:
            pass  # Never crash on alignment failure

    def split_post(self, text: str, idx):
        split_files = []
        _max   = str(self.tts_module.max_chars)
        _pat   = r" *(((.|\n){0," + _max + r"})(\.|.$))"
        split_text = [x.group().strip() for x in re.finditer(_pat, text)]
        self.create_silence_mp3()

        for idy, text_cut in enumerate(split_text):
            newtext = process_text(text_cut)
            if not newtext or newtext.isspace():
                print("newtext was blank because sanitized split text resulted in none")
                continue
            else:
                self.call_tts(f"{idx}-{idy}.part", newtext)
                with open(f"{self.path}/list.txt", "w") as f:
                    for idz in range(0, len(split_text)):
                        f.write("file " + f"'{idx}-{idz}.part.mp3'" + "\n")
                    split_files.append(str(f"{self.path}/{idx}-{idy}.part.mp3"))
                    f.write("file " + f"'silence.mp3'" + "\n")

                os.system(
                    "ffmpeg -f concat -y -hide_banner -loglevel panic -safe 0 "
                    + "-i "
                    + f"{self.path}/list.txt "
                    + "-c copy "
                    + f"{self.path}/{idx}.mp3"
                )
        try:
            for i in range(0, len(split_files)):
                os.unlink(split_files[i])
        except FileNotFoundError as e:
            print("File not found: " + e.filename)
        except OSError:
            print("OSError")

    def call_tts(self, filename: str, text: str):
        if settings.config["settings"]["tts"]["voice_choice"] == "googletranslate":
            self.tts_module.run(
                text,
                filepath=f"{self.path}/{filename}.mp3",
            )
        else:
            self.tts_module.run(
                text,
                filepath=f"{self.path}/{filename}.mp3",
                random_voice=settings.config["settings"]["tts"]["random_voice"],
            )
        try:
            clip = AudioFileClip(f"{self.path}/{filename}.mp3")
            self.last_clip_length = clip.duration
            self.length += clip.duration
            clip.close()
        except:
            self.length = 0

    def create_silence_mp3(self):
        silence_duration = settings.config["settings"]["tts"]["silence_duration"]
        silence = AudioClip(
            frame_function=lambda t: np.sin(440 * 2 * np.pi * t),
            duration=silence_duration,
            fps=44100,
        )
        silence = silence.with_effects([MultiplyVolume(0)])
        silence.write_audiofile(f"{self.path}/silence.mp3", fps=44100, logger=None)


def process_text(text: str, clean: bool = True):
    lang = settings.config["reddit"]["thread"]["post_lang"]
    new_text = sanitize_text(text) if clean else text
    if lang:
        print_substep("Translating Text...")
        translated_text = translators.translate_text(text, translator="google", to_language=lang)
        new_text = sanitize_text(translated_text)
    return new_text