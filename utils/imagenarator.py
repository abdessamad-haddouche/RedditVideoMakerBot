"""
imagenarator.py
───────────────
Thin orchestrator. Does exactly:
  1. Extract sentences from reddit_obj["thread_post"]
  2. Probe audio durations + compute audio start times
  3. Call caption_renderer.get_render_jobs()
  4. Apply SLOW_DURATION_MULT to [SLOW]-flagged jobs
  5. Render each job to PNG
  6. Save timing_map.json for final_video.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Changes (feature/deepseek-text-enhancement)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_extract_sentences():
  Reads item.display if item is an EnhancedSentence.
  Falls back to plain string / dict["text"] path unchanged.
  Also collects is_slow flags from item.is_slow.

get_render_jobs() call:
  Now receives: sentiment, is_slow_flags
  (caption_renderer uses these for correct tag styling + SLOW timing)

SLOW timing:
  After get_render_jobs() returns, jobs with is_slow=True get their
  time_fraction multiplied by SLOW_DURATION_MULT[sentiment].
  Absolute-timed (aligned) jobs: clip_end extended proportionally.

Everything else is identical to the original.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import glob
import json
import os
import re
from typing import List, Optional, Tuple

import ffmpeg
from rich.progress import track

from TTS.engine_wrapper import process_text
from utils import settings
from utils.id import extract_id
from utils.sentiment_map import STYLE_MAP, DEFAULT_STYLE, SLOW_DURATION_MULT
from utils.caption_renderer import get_render_jobs, render_job_to_image, RenderJob


LINE_SPACING: int = 20


# ─────────────────────────────────────────────────────────────────────────────
# Sentence extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_sentences(
    reddit_obj: dict,
    style:      dict,
) -> Tuple[List[str], List[bool]]:
    """
    Extract display sentences and is_slow flags from thread_post.

    Handles three item types:
      EnhancedSentence → uses .display and .is_slow   (feature active)
      dict             → uses ["text"]                (original format)
      str              → used directly                (plain fallback)

    Returns:
        sentences  : List[str]  — display text per sentence (may have [TAG] markup)
        is_slow    : List[bool] — one bool per sentence
    """
    raw_items  = reddit_obj["thread_post"]
    sentences: List[str]  = []
    is_slow:   List[bool] = []

    for item in raw_items:
        # ── EnhancedSentence path ─────────────────────────────────────────────
        if hasattr(item, "display"):
            text  = item.display
            slow  = getattr(item, "is_slow", False)

        # ── Original dict path ────────────────────────────────────────────────
        elif isinstance(item, dict):
            text  = item.get("text", "")
            slow  = False

        # ── Plain string fallback ─────────────────────────────────────────────
        elif isinstance(item, str):
            text  = item
            slow  = False

        else:
            text  = str(item)
            slow  = False

        # process_text with clean=False: don't sanitize display tags out of the text.
        # sanitize_text would strip [, ], / characters which are part of our tag syntax.
        # The tags are stripped by parse_visual_tags in caption_renderer, not here.
        text = text.strip()

        # Apply uppercase AFTER we have the text, before tag parsing in caption_renderer.
        # Tags are uppercase-safe ([EMPHASIZE] vs [emphasize] both handled).
        if style.get("uppercase", False):
            text = text.upper()

        if text:
            sentences.append(text)
            is_slow.append(slow)

    if not sentences:
        sentences = ["..."]
        is_slow   = [False]

    return sentences, is_slow


# ─────────────────────────────────────────────────────────────────────────────
# Audio timing
# ─────────────────────────────────────────────────────────────────────────────

def _get_audio_info(mp3_dir: str) -> Tuple[List[str], List[float], List[float]]:
    """
    Discover postaudio files and compute:
      - postaudio_files: sorted list of postaudio-N.mp3 paths
      - durations:       one float per file
      - start_times:     absolute seconds in video (after title card)

    Returns (postaudio_files, durations, start_times)
    """
    postaudio_files = sorted(
        glob.glob(os.path.join(mp3_dir, "postaudio-*.mp3")),
        key=lambda x: int(re.search(r"postaudio-(\d+)", x).group(1)),
    )

    title_path = os.path.join(mp3_dir, "title.mp3")
    try:
        title_duration = float(ffmpeg.probe(title_path)["format"]["duration"])
    except Exception:
        title_duration = 0.0

    durations:   List[float] = []
    start_times: List[float] = []
    current = title_duration

    for f in postaudio_files:
        try:
            dur = float(ffmpeg.probe(f)["format"]["duration"])
        except Exception:
            dur = 0.0
        start_times.append(current)
        durations.append(dur)
        current += dur

    return postaudio_files, durations, start_times


# ─────────────────────────────────────────────────────────────────────────────
# SLOW timing adjustment
# ─────────────────────────────────────────────────────────────────────────────

def _apply_slow_timing(jobs: List[RenderJob], sentiment: str) -> None:
    """
    Multiplies the display duration of [SLOW]-flagged jobs.

    Fraction-based jobs: time_fraction *= mult
    Absolute-based jobs: clip_end extended by (duration * (mult - 1))

    Mutates jobs in-place. Safe to call even if no jobs are slow.
    """
    mult = SLOW_DURATION_MULT.get(sentiment, 1.0)
    if mult == 1.0:
        return

    for job in jobs:
        if not job.is_slow:
            continue

        if job.timing_type == "fraction":
            job.time_fraction = min(job.time_fraction * mult, 1.0)

        elif job.timing_type == "absolute":
            duration = job.clip_end - job.clip_start
            job.clip_end = round(job.clip_end + duration * (mult - 1.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def imagemaker(
    theme,
    reddit_obj: dict,
    txtclr,
    padding:     int  = 5,
    transparent: bool = False,
) -> int:
    """
    Render caption images for the video.

    Flow:
        1. Extract sentences + is_slow flags from reddit_obj["thread_post"]
        2. Get audio timing info (needed for aligned mode)
        3. Call caption_renderer.get_render_jobs()
        4. Apply SLOW_DURATION_MULT to flagged jobs
        5. Render each job to transparent PNG (img{idx}.png)
        6. Save timing_map.json for final_video.py

    Returns:
        int: total number of images generated
    """
    # ── Style + canvas ───────────────────────────────────────────────────────
    sentiment = settings.config["settings"].get("sentiment", "dramatic")
    style     = STYLE_MAP.get(sentiment, DEFAULT_STYLE)
    CANVAS_W  = int(settings.config["settings"]["resolution_w"])
    CANVAS_H  = int(settings.config["settings"]["resolution_h"])
    reddit_id = extract_id(reddit_obj)
    mp3_dir   = f"assets/temp/{reddit_id}/mp3"

    # ── 1. Extract sentences ─────────────────────────────────────────────────
    sentences, is_slow_flags = _extract_sentences(reddit_obj, style)

    # ── 2. Audio timing ──────────────────────────────────────────────────────
    _, durations, start_times = _get_audio_info(mp3_dir)

    # ── 3. Get render jobs ───────────────────────────────────────────────────
    jobs: List[RenderJob] = get_render_jobs(
        sentences         = sentences,
        style             = style,
        sentiment         = sentiment,
        mp3_dir           = mp3_dir,
        audio_start_times = start_times  if start_times  else None,
        audio_durations   = durations    if durations    else None,
        is_slow_flags     = is_slow_flags,
    )

    # ── 4. Apply SLOW timing multiplier ─────────────────────────────────────
    _apply_slow_timing(jobs, sentiment)

    # ── 5. Render each job to transparent PNG ────────────────────────────────
    png_dir = f"assets/temp/{reddit_id}/png"
    os.makedirs(png_dir, exist_ok=True)

    for job in track(jobs, description="Rendering caption images"):
        image = render_job_to_image(job, style, CANVAS_W, CANVAS_H, LINE_SPACING)
        image.save(os.path.join(png_dir, f"img{job.idx}.png"))

    # ── 6. Save timing map ───────────────────────────────────────────────────
    timing_map = []
    for job in jobs:
        if job.timing_type == "absolute":
            timing_map.append({
                "timing_type": "absolute",
                "clip_start":  job.clip_start,
                "clip_end":    job.clip_end,
            })
        else:
            timing_map.append({
                "timing_type":   "fraction",
                "audio_idx":     job.audio_idx,
                "time_fraction": job.time_fraction,
            })

    timing_map_path = f"assets/temp/{reddit_id}/timing_map.json"
    with open(timing_map_path, "w") as f:
        json.dump(timing_map, f, indent=2)

    return len(jobs)