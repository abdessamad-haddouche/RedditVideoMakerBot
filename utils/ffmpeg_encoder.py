"""
ffmpeg_encoder.py
─────────────────
Cross-platform FFmpeg encoder and filter detection.

Detects at runtime what the current system supports and returns
the best available options. Called once per run, result cached.

Encoder priority:
  Linux / Windows with NVIDIA GPU  → h264_nvenc   (hardware, fast)
  macOS (Apple Silicon or Intel)   → h264_videotoolbox (hardware, fast)
  Any platform fallback            → libx264      (software, universal)

drawtext filter:
  Requires FFmpeg built with --enable-libfreetype.
  On macOS Homebrew this is sometimes missing.
  Falls back to skipping the watermark text gracefully.

Usage:
  from utils.ffmpeg_encoder import get_video_codec_args, has_drawtext

  codec_args = get_video_codec_args()   # {"c:v": "...", "b:v": "20M", ...}
  if has_drawtext():
      clip = ffmpeg.drawtext(clip, ...)
"""

import subprocess
import sys
from functools import lru_cache
from typing import Dict

from utils.console import print_substep


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_ffmpeg(*args) -> str:
    """Run ffmpeg with given args, return stdout+stderr combined. Never raises."""
    try:
        result = subprocess.run(
            ["ffmpeg"] + list(args),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout + result.stderr
    except Exception:
        return ""


def _encoder_available(encoder: str) -> bool:
    """Returns True if the named encoder is available in the current FFmpeg build."""
    output = _run_ffmpeg("-hide_banner", "-encoders")
    return encoder in output


def _filter_available(filter_name: str) -> bool:
    """Returns True if the named filter is available in the current FFmpeg build."""
    output = _run_ffmpeg("-hide_banner", "-filters")
    return filter_name in output


# ─────────────────────────────────────────────────────────────────────────────
# Public API — cached so detection runs only once per process
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_video_encoder() -> str:
    """
    Detects and returns the best available H.264 encoder for this platform.

    Returns one of:
      "h264_nvenc"         — NVIDIA hardware encoder (Linux/Windows + GPU)
      "h264_videotoolbox"  — Apple hardware encoder (macOS)
      "libx264"            — Software encoder (universal fallback)
    """
    is_mac = sys.platform == "darwin"

    if not is_mac and _encoder_available("h264_nvenc"):
        print_substep("Video encoder: h264_nvenc (NVIDIA hardware)", style="dim")
        return "h264_nvenc"

    if is_mac and _encoder_available("h264_videotoolbox"):
        print_substep("Video encoder: h264_videotoolbox (Apple hardware)", style="dim")
        return "h264_videotoolbox"

    # Universal fallback — always available if FFmpeg was built with x264
    print_substep("Video encoder: libx264 (software fallback)", style="dim")
    return "libx264"


@lru_cache(maxsize=1)
def get_video_codec_args() -> Dict[str, object]:
    """
    Returns the complete FFmpeg video codec kwargs dict for ffmpeg-python output calls.

    Example usage:
        ffmpeg.output(clip, audio, path, f="mp4", **get_video_codec_args())
    """
    import multiprocessing
    encoder = get_video_encoder()

    args = {
        "c:v":     encoder,
        "b:v":     "20M",
        "b:a":     "192k",
        "threads": multiprocessing.cpu_count(),
    }

    # videotoolbox doesn't support the threads parameter
    if encoder == "h264_videotoolbox":
        del args["threads"]

    return args


@lru_cache(maxsize=1)
def has_drawtext() -> bool:
    """
    Returns True if the drawtext filter is available in the current FFmpeg build.
    Requires FFmpeg built with --enable-libfreetype.

    On macOS, Homebrew FFmpeg sometimes lacks this even in recent builds.
    Fix: brew install ffmpeg --with-libfreetype  (or just accept no watermark)
    """
    available = _filter_available("drawtext")
    if not available:
        print_substep(
            "FFmpeg drawtext filter not available (missing libfreetype). "
            "Background watermark text will be skipped. "
            "To fix: brew install freetype && brew reinstall ffmpeg",
            style="yellow",
        )
    return available