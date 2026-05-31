"""
imagenarator.py
───────────────
Thin orchestrator. Does exactly:
  1. Extract sentences from reddit_obj
  2. Probe audio durations + compute audio start times (needed for aligned mode)
  3. Call caption_renderer.get_render_jobs()
  4. Render each job to PNG
  5. Save timing_map.json for final_video.py
"""

import glob
import json
import os
import re
from typing import List, Optional

import ffmpeg
from rich.progress import track

from TTS.engine_wrapper import process_text
from utils import settings
from utils.id import extract_id
from utils.sentiment_map import STYLE_MAP, DEFAULT_STYLE
from utils.caption_renderer import get_render_jobs, render_job_to_image, RenderJob


LINE_SPACING: int = 20


def _extract_sentences(reddit_obj: dict, style: dict) -> List[str]:
    """
    Extract sentences from thread_post.
    One sentence per postaudio-{i}.mp3 — order preserved.
    """
    raw_texts = reddit_obj["thread_post"]
    sentences: List[str] = []
    for item in raw_texts:
        if isinstance(item, dict):
            text = item.get("text", "")
        elif isinstance(item, str):
            text = item
        else:
            text = str(item)
        text = process_text(text, False).strip()
        if style.get("uppercase", False):
            text = text.upper()
        if text:
            sentences.append(text)
    return sentences if sentences else ["..."]


def _get_audio_info(mp3_dir: str) -> tuple:
    """
    Discover postaudio files and compute:
      - durations list (one per postaudio file)
      - start times list (absolute seconds in video, after title card)

    Returns (postaudio_files, durations, start_times)
    """
    postaudio_files = sorted(
        glob.glob(os.path.join(mp3_dir, "postaudio-*.mp3")),
        key=lambda x: int(re.search(r"postaudio-(\d+)", x).group(1))
    )

    title_path = os.path.join(mp3_dir, "title.mp3")
    try:
        title_duration = float(ffmpeg.probe(title_path)["format"]["duration"])
    except Exception:
        title_duration = 0.0

    durations   = []
    start_times = []
    current     = title_duration

    for f in postaudio_files:
        try:
            dur = float(ffmpeg.probe(f)["format"]["duration"])
        except Exception:
            dur = 0.0
        start_times.append(current)
        durations.append(dur)
        current += dur

    return postaudio_files, durations, start_times


def imagemaker(theme, reddit_obj: dict, txtclr, padding=5, transparent=False) -> int:
    """
    Render caption images for the video.

    Flow:
        sentences + audio info
            → caption_renderer.get_render_jobs()
            → List[RenderJob]
        each RenderJob → transparent PNG (img{idx}.png)
        timing_map.json → saved for final_video.py

    timing_map.json entry for fraction-based jobs:
        {"timing_type": "fraction", "audio_idx": N, "time_fraction": F}

    timing_map.json entry for absolute-based jobs (aligned mode):
        {"timing_type": "absolute", "clip_start": S, "clip_end": E}

    Returns:
        int: total number of images generated
    """
    # 1. Style
    sentiment = settings.config["settings"].get("sentiment", "dramatic")
    style     = STYLE_MAP.get(sentiment, DEFAULT_STYLE)
    CANVAS_W: int = int(settings.config["settings"]["resolution_w"])
    CANVAS_H: int = int(settings.config["settings"]["resolution_h"])
    reddit_id     = extract_id(reddit_obj)
    mp3_dir       = f"assets/temp/{reddit_id}/mp3"

    # 2. Extract sentences
    sentences = _extract_sentences(reddit_obj, style)

    # 3. Get audio timing info (needed for aligned mode)
    _, durations, start_times = _get_audio_info(mp3_dir)

    # 4. Get render jobs
    jobs: List[RenderJob] = get_render_jobs(
        sentences=sentences,
        style=style,
        mp3_dir=mp3_dir,
        audio_start_times=start_times if start_times else None,
        audio_durations=durations   if durations   else None,
    )

    # 5. Render each job to a transparent PNG
    for job in track(jobs, description="Rendering caption images"):
        image = render_job_to_image(job, style, CANVAS_W, CANVAS_H, LINE_SPACING)
        image.save(f"assets/temp/{reddit_id}/png/img{job.idx}.png")

    # 6. Save timing map
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
                "timing_type":  "fraction",
                "audio_idx":    job.audio_idx,
                "time_fraction": job.time_fraction,
            })

    timing_map_path = f"assets/temp/{reddit_id}/timing_map.json"
    with open(timing_map_path, "w") as f:
        json.dump(timing_map, f, indent=2)

    return len(jobs)