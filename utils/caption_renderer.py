"""
caption_renderer.py
───────────────────
All caption rendering logic. Three display modes:

  multi    → full sentence on one image (1 RenderJob per sentence)
  single   → sentence split into word chunks (N RenderJobs per sentence)
  aligned  → word-level timestamps from WhisperX (perfect sync, any TTS)

RenderJob is the contract between this module and final_video.py.
Two types of timing:

  FRACTION-based (multi, single):
    audio_idx + time_fraction → final_video computes absolute time
    time_fraction = fraction of audio_clips_durations[audio_idx+1]

  ABSOLUTE-based (aligned):
    clip_start + clip_end → final_video uses directly
    These are absolute seconds in the video timeline (after title card)

final_video.py checks job["timing_type"] to know which to use.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from utils.fonts import getsize


# ─────────────────────────────────────────────────────────────────────────────
# RenderJob — the contract
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RenderJob:
    """
    Describes exactly one output image (img{idx}.png).

    timing_type = "fraction":
        audio_idx + time_fraction used by final_video to compute display time.
        time_fraction = 1.0 means shown for full audio file duration.
        time_fraction = 0.25 means shown for 25% of audio file duration.

    timing_type = "absolute":
        clip_start + clip_end are absolute seconds in the video timeline.
        final_video uses these directly — no calculation needed.
    """
    idx:          int
    lines:        List[str]
    timing_type:  str          # "fraction" or "absolute"

    # fraction-based fields
    audio_idx:    int   = 0
    time_fraction: float = 1.0

    # absolute-based fields
    clip_start:   float = 0.0
    clip_end:     float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Display modes
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_MODES = {"single", "multi", "aligned"}


def render_multi_mode(
    sentence: str,
    style: dict,
    audio_idx: int,
    start_idx: int,
) -> List[RenderJob]:
    """
    Full sentence on one image, wrapped into lines.
    One RenderJob, time_fraction = 1.0.
    Best for: funny, sad, wholesome, happy.
    """
    words = sentence.split()
    wpl   = style["words_per_chunk"]
    lines = [" ".join(words[i:i + wpl]) for i in range(0, len(words), wpl)]
    if not lines:
        lines = [sentence]

    return [RenderJob(
        idx=start_idx,
        lines=lines,
        timing_type="fraction",
        audio_idx=audio_idx,
        time_fraction=1.0,
    )]


def render_single_mode(
    sentence: str,
    style: dict,
    audio_idx: int,
    start_idx: int,
) -> List[RenderJob]:
    """
    Sentence split into word chunks, one per image.
    Each shown for (1/N) of the audio duration.
    Best for: scary, dramatic, angry, mysterious.
    """
    wpc   = style["words_per_chunk"]
    words = sentence.split()
    raw   = [" ".join(words[i:i + wpc]) for i in range(0, len(words), wpc)]
    raw   = [c for c in raw if c.strip()] or [sentence]

    n        = len(raw)
    fraction = 1.0 / n

    return [
        RenderJob(
            idx=start_idx + i,
            lines=[chunk],
            timing_type="fraction",
            audio_idx=audio_idx,
            time_fraction=fraction,
        )
        for i, chunk in enumerate(raw)
    ]


def render_aligned_mode(
    sentence: str,
    style: dict,
    audio_idx: int,
    start_idx: int,
    word_timestamps: List[dict],
    audio_start_time: float,
    audio_duration: float,
) -> List[RenderJob]:
    """
    Word-level aligned mode using WhisperX timestamps.

    Groups consecutive words into chunks of words_per_chunk words.
    Each chunk's clip_start = timestamp of first word in chunk.
    Each chunk's clip_end   = timestamp of last word in chunk + its duration.

    audio_start_time: absolute time in video when this audio file starts.
    audio_duration:   duration of this audio file (used as fallback end time).

    Falls back to single mode if timestamps are empty or malformed.
    """
    wpc = style["words_per_chunk"]

    if not word_timestamps:
        return render_single_mode(sentence, style, audio_idx, start_idx)

    # Group word timestamps into chunks of wpc words
    jobs = []
    n    = len(word_timestamps)

    for chunk_start in range(0, n, wpc):
        chunk_words = word_timestamps[chunk_start:chunk_start + wpc]
        if not chunk_words:
            continue

        text       = " ".join(w["word"] for w in chunk_words)
        clip_start = audio_start_time + chunk_words[0]["start"]

        # clip_end = end of last word in chunk,
        # or start of next chunk if available, capped at audio end
        if chunk_start + wpc < n:
            clip_end = audio_start_time + word_timestamps[chunk_start + wpc]["start"]
        else:
            last_end = chunk_words[-1].get("end", chunk_words[-1]["start"] + 0.3)
            clip_end = audio_start_time + last_end

        # Safety: never exceed audio boundary
        audio_end = audio_start_time + audio_duration
        clip_end  = min(clip_end, audio_end)
        clip_end  = max(clip_end, clip_start + 0.1)  # minimum 100ms visibility

        jobs.append(RenderJob(
            idx=start_idx + len(jobs),
            lines=[text],
            timing_type="absolute",
            clip_start=round(clip_start, 3),
            clip_end=round(clip_end,   3),
        ))

    return jobs if jobs else render_single_mode(sentence, style, audio_idx, start_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

def get_render_jobs(
    sentences: List[str],
    style: dict,
    mp3_dir: Optional[str] = None,
    audio_start_times: Optional[List[float]] = None,
    audio_durations: Optional[List[float]] = None,
) -> List[RenderJob]:
    """
    Route each sentence to the correct renderer.
    Returns flat ordered list of all RenderJobs.

    For "aligned" mode, loads word timestamps from
    {mp3_dir}/postaudio-{i}_words.json written by engine_wrapper.
    Falls back to "single" mode per sentence if timestamps missing.

    Parameters
    ----------
    sentences         : one per postaudio-{i}.mp3
    style             : STYLE_MAP entry for current sentiment
    mp3_dir           : path to mp3 folder (needed for aligned mode)
    audio_start_times : absolute start time of each audio in video (needed for aligned)
    audio_durations   : duration of each audio file (needed for aligned)
    """
    mode = style.get("display_mode", "multi")

    if mode not in DISPLAY_MODES:
        print(f"[caption_renderer] Unknown display_mode '{mode}', using 'multi'")
        mode = "multi"

    all_jobs:    List[RenderJob] = []
    img_counter: int             = 0

    for audio_idx, sentence in enumerate(sentences):

        if mode == "aligned" and mp3_dir and audio_start_times and audio_durations:
            # Try to load word timestamps for this sentence
            from utils.whisper_aligner import load_word_timestamps
            audio_path = os.path.join(mp3_dir, f"postaudio-{audio_idx}.mp3")
            word_ts    = load_word_timestamps(audio_path)

            if word_ts:
                jobs = render_aligned_mode(
                    sentence=sentence,
                    style=style,
                    audio_idx=audio_idx,
                    start_idx=img_counter,
                    word_timestamps=word_ts,
                    audio_start_time=audio_start_times[audio_idx],
                    audio_duration=audio_durations[audio_idx],
                )
            else:
                # WhisperX not available or failed — fall back to single mode
                print(f"[caption_renderer] No timestamps for sentence {audio_idx}, using single mode")
                jobs = render_single_mode(sentence, style, audio_idx, img_counter)

        elif mode == "single":
            jobs = render_single_mode(sentence, style, audio_idx, img_counter)

        else:
            jobs = render_multi_mode(sentence, style, audio_idx, img_counter)

        all_jobs.extend(jobs)
        img_counter += len(jobs)

    return all_jobs


# ─────────────────────────────────────────────────────────────────────────────
# Drawing primitives
# ─────────────────────────────────────────────────────────────────────────────

def measure_text_block(
    draw: ImageDraw.ImageDraw,
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    line_spacing: int,
) -> tuple:
    max_w = 0
    total_h = 0
    for i, line in enumerate(lines):
        w, h = getsize(font, line)
        if w > max_w:
            max_w = w
        total_h += h
        if i < len(lines) - 1:
            total_h += line_spacing
    return max_w, total_h


def draw_stroked_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    line: str,
    font: ImageFont.FreeTypeFont,
    fill_color: tuple,
    stroke_color: tuple,
    stroke_width: int,
) -> None:
    sw   = stroke_width
    half = max(1, sw // 2)
    offsets = [
        (-sw, 0), (sw, 0), (0, -sw), (0, sw),
        (-sw, -sw), (sw, -sw), (-sw, sw), (sw, sw),
        (-sw, -half), (sw, -half), (-sw, half), (sw, half),
        (-half, -sw), (half, -sw), (-half, sw), (half, sw),
    ]
    for ox, oy in offsets:
        draw.text((x + ox, y + oy), line, font=font, fill=stroke_color)
    draw.text((x, y), line, font=font, fill=fill_color)


def fit_font(
    style: dict,
    lines: List[str],
    canvas_w: int,
    canvas_h: int,
    line_spacing: int,
    max_width_ratio: float = 0.88,
    max_height_ratio: float = 0.45,
) -> ImageFont.FreeTypeFont:
    font_size = style["font_size"]
    font_path = os.path.join("fonts", style["font_file"])
    if not os.path.exists(font_path):
        font_path = os.path.join("fonts", "Roboto-Bold.ttf")
    max_w = int(canvas_w * max_width_ratio)
    max_h = int(canvas_h * max_height_ratio)
    while font_size > 30:
        font      = ImageFont.truetype(font_path, font_size)
        dummy     = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        dummy_d   = ImageDraw.Draw(dummy)
        bw, bh    = measure_text_block(dummy_d, lines, font, line_spacing)
        if bw <= max_w and bh <= max_h:
            return font
        font_size -= 4
    return ImageFont.truetype(font_path, 30)


def render_job_to_image(
    job: RenderJob,
    style: dict,
    canvas_w: int,
    canvas_h: int,
    line_spacing: int,
) -> Image.Image:
    font    = fit_font(style, job.lines, canvas_w, canvas_h, line_spacing)
    image   = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(image)
    bw, bh  = measure_text_block(draw, job.lines, font, line_spacing)
    anchor_y = int(canvas_h * style["y_position"]) - (bh // 2)
    cy = anchor_y
    for line in job.lines:
        w, h = getsize(font, line)
        x    = (canvas_w - w) // 2
        draw_stroked_text(draw, x, cy, line, font,
                          style["fill_color"], style["stroke_color"], style["stroke_width"])
        cy += h + line_spacing
    return image