"""
caption_renderer.py
───────────────────
All caption rendering logic. Three display modes:

  multi    → full sentence on one image (1 RenderJob per sentence)
  single   → sentence split into word chunks (N RenderJobs per sentence)
  aligned  → word-level timestamps from WhisperX (perfect sync, any TTS)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW (feature/deepseek-text-enhancement)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WordSpan dataclass:
  One word with its complete visual style attached.
  Tags supported: "emphasize", "shock", "hook", None (untagged).

parse_visual_tags(display_text, style, sentiment):
  Parses [TAG]...[/TAG] markup from display text.
  Returns List[WordSpan] — one span per word.
  Unknown tags → treated as untagged (never crashes).
  Malformed tags → rest of sentence treated as untagged.

RenderJob.spans (replaces RenderJob.lines):
  List[WordSpan] instead of List[str].
  Each span carries its own color, font, size_mult.

render_job_to_image():
  Updated to render per-word using WordSpan styles.
  Baseline-anchored mixed-size rendering:
    - All words in a line share the same visual baseline.
    - Taller (SHOCK) words anchor the line height.
    - Shorter words are shifted down so their bottoms align.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RenderJob timing types (unchanged):

  FRACTION-based (multi, single):
    audio_idx + time_fraction → final_video computes absolute time

  ABSOLUTE-based (aligned):
    clip_start + clip_end → final_video uses directly

final_video.py checks job["timing_type"] — zero changes there.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from utils.fonts import getsize


# ─────────────────────────────────────────────────────────────────────────────
# WordSpan — one word with its complete visual style
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WordSpan:
    """
    Represents one word (or token) with its full rendering style.

    tag:
        None        → base style (most words)
        "emphasize" → [EMPHASIZE] tag — different color/font, same size
        "shock"     → [SHOCK] tag — different color/font, +20% size
        "hook"      → [HOOK] tag — accent color, +10% size (sentence[0] only)

    color:     RGBA tuple, drawn as fill
    font_file: filename only (e.g. "Montserrat-ExtraBold.ttf")
               full path prepended at render time
    size_mult: multiplier on the base font_size from STYLE_MAP
               1.0 = normal, 1.10 = hook, 1.20 = shock
    """
    word:      str
    tag:       Optional[str]  # None / "emphasize" / "shock" / "hook"
    color:     Tuple[int, int, int, int]
    font_file: str
    size_mult: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# RenderJob — the contract between this module and final_video.py
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RenderJob:
    """
    Describes exactly one output image (img{idx}.png).

    spans:
        List[WordSpan] — replaces the old List[str] lines.
        Each word carries its own color, font, size_mult.

    is_slow:
        Pre-computed from [SLOW] tag in display text.
        imagenarator.py uses this to apply SLOW_DURATION_MULT.

    timing_type = "fraction":
        audio_idx + time_fraction used by final_video to compute display time.

    timing_type = "absolute":
        clip_start + clip_end are absolute seconds in the video timeline.
    """
    idx:           int
    spans:         List[WordSpan]
    timing_type:   str           # "fraction" or "absolute"
    is_slow:       bool  = False

    # fraction-based fields
    audio_idx:     int   = 0
    time_fraction: float = 1.0

    # absolute-based fields
    clip_start:    float = 0.0
    clip_end:      float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Visual tag parser
# ─────────────────────────────────────────────────────────────────────────────

# Maps tag name (lowercase) → style map key in sentiment_map
_TAG_TO_STYLE_KEY = {
    "emphasize": "EMPHASIS_STYLE",
    "shock":     "SHOCK_STYLE",
    "hook":      "HOOK_STYLE",
}

# Regex: matches [TAG]content[/TAG] — case-insensitive, non-greedy
_TAG_PATTERN = re.compile(
    r"\[(?P<tag>EMPHASIZE|SHOCK|HOOK|SLOW)\]"
    r"(?P<content>.*?)"
    r"\[/(?P=tag)\]",
    re.IGNORECASE | re.DOTALL,
)


def parse_visual_tags(
    display_text: str,
    style: dict,
    sentiment: str,
) -> List[WordSpan]:
    """
    Parses [TAG]...[/TAG] markup in display_text.
    Returns List[WordSpan] — one span per word.

    Base style (untagged words):
        color     = style["fill_color"]
        font_file = style["font_file"]
        size_mult = 1.0

    Tagged words get overridden per EMPHASIS_STYLE / SHOCK_STYLE / HOOK_STYLE.

    Edge cases:
        - Unknown tags  → stripped, words treated as untagged
        - [SLOW] tag    → wraps entire sentence, not per-word styling,
                          so [SLOW] is stripped and words are untagged
                          (is_slow is handled at RenderJob level, not here)
        - Nested tags   → outer tag wins (inner stripped cleanly)
        - Empty text    → returns []
        - No tags       → all words are untagged WordSpans

    Args:
        display_text: raw display string with optional [TAG] markup
        style:        STYLE_MAP entry for current sentiment
        sentiment:    current sentiment label (for EMPHASIS/SHOCK/HOOK lookups)

    Returns:
        List[WordSpan]
    """
    from utils.sentiment_map import EMPHASIS_STYLE, SHOCK_STYLE, HOOK_STYLE

    # Style lookup for each tag type
    tag_styles = {
        "emphasize": EMPHASIS_STYLE.get(sentiment, EMPHASIS_STYLE.get("dramatic", {})),
        "shock":     SHOCK_STYLE.get(sentiment,     SHOCK_STYLE.get("dramatic", {})),
        "hook":      HOOK_STYLE.get(sentiment,       HOOK_STYLE.get("dramatic", {})),
    }

    # Base (untagged) style
    base_color     = style.get("fill_color", (255, 255, 255, 255))
    base_font_file = style.get("font_file",  "Montserrat-ExtraBold.ttf")

    # ── Build a list of (word, tag_name) tuples ───────────────────────────────
    # Strategy: scan the text left-to-right. When we hit a [TAG]...[/TAG],
    # record each word inside it with the tag name. Everything else is untagged.

    word_tag_pairs: List[Tuple[str, Optional[str]]] = []
    last_end = 0

    for match in _TAG_PATTERN.finditer(display_text):
        tag_name = match.group("tag").lower()
        content  = match.group("content")
        start    = match.start()
        end      = match.end()

        # Text before this tag → untagged
        before = display_text[last_end:start]
        for word in before.split():
            if word:
                word_tag_pairs.append((word, None))

        # [SLOW] is sentence-level, not word-level styling → treat as untagged
        if tag_name == "slow":
            for word in content.split():
                if word:
                    word_tag_pairs.append((word, None))
        elif tag_name in _TAG_TO_STYLE_KEY:
            # Strip any nested tags inside this one
            inner = re.sub(r"\[[^\]]+\]", "", content)
            for word in inner.split():
                if word:
                    word_tag_pairs.append((word, tag_name))
        else:
            # Unknown tag → strip markup, keep words as untagged
            for word in content.split():
                if word:
                    word_tag_pairs.append((word, None))

        last_end = end

    # Remaining text after last tag
    after = display_text[last_end:]
    for word in after.split():
        if word:
            word_tag_pairs.append((word, None))

    # ── Convert to WordSpan list ──────────────────────────────────────────────
    spans: List[WordSpan] = []
    for word, tag in word_tag_pairs:
        if tag and tag in tag_styles and tag_styles[tag]:
            ts = tag_styles[tag]
            spans.append(WordSpan(
                word      = word,
                tag       = tag,
                color     = ts.get("color",     base_color),
                font_file = ts.get("font_file",  base_font_file),
                size_mult = ts.get("size_mult",  1.0),
            ))
        else:
            spans.append(WordSpan(
                word      = word,
                tag       = None,
                color     = base_color,
                font_file = base_font_file,
                size_mult = 1.0,
            ))

    return spans


# ─────────────────────────────────────────────────────────────────────────────
# Span helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_font(font_file: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a font by filename. Falls back to Roboto-Bold if not found."""
    path = os.path.join("fonts", font_file)
    if not os.path.exists(path):
        path = os.path.join("fonts", "Roboto-Bold.ttf")
    return ImageFont.truetype(path, size)


# Shared dummy draw for all measurements — avoids re-creating per call.
_DUMMY_IMG  = Image.new("RGBA", (1, 1))
_DUMMY_DRAW = ImageDraw.Draw(_DUMMY_IMG)


def _measure_span(span: WordSpan, base_size: int) -> Tuple[int, int]:
    """
    Returns (advance_width, bbox_height) using textbbox.
    advance_width  = pixel width to advance x cursor after this word.
    bbox_height    = height of the glyph bounding box (excludes top bearing).

    Using textbbox with anchor="ls" (left, baseline) gives us metrics
    relative to the baseline, so all fonts align correctly regardless of
    their internal metrics or top-bearing differences.
    """
    size = max(12, int(base_size * span.size_mult))
    font = _get_font(span.font_file, size)
    # anchor="ls": x=left edge, y=baseline
    # bbox = (left, top_above_baseline, right, bottom_below_baseline)
    # top is negative (above baseline), bottom is positive (descender)
    bbox = _DUMMY_DRAW.textbbox((0, 0), span.word, font=font, anchor="ls")
    w    = bbox[2] - bbox[0]   # right - left = advance width
    # Full visual height = ascender + descender
    # ascender  = -bbox[1]  (top is negative above baseline)
    # descender =  bbox[3]  (bottom is positive below baseline)
    ascender  = -bbox[1]
    descender =  bbox[3]
    h         = ascender + descender
    return w, h


def _get_span_baseline_offset(span: WordSpan, base_size: int) -> Tuple[int, int, int]:
    """
    Returns (advance_width, ascender, descender) for a span.

    ascender:  pixels the glyph rises ABOVE the typographic baseline (positive)
    descender: pixels the glyph falls BELOW the typographic baseline (positive)

    These are used ONLY for:
      - Computing line_height = max_ascender + max_descender (layout)
      - Computing baseline_y position (vertical centering)

    They are NOT used to compute per-word y offsets.
    All words are rendered with draw.text(..., anchor="ls") at the SAME
    baseline_y — Pillow handles per-glyph placement internally.
    """
    size = max(12, int(base_size * span.size_mult))
    font = _get_font(span.font_file, size)
    # anchor="ls": x=left edge, y=typographic baseline
    # bbox[1] is negative (ascender above baseline)
    # bbox[3] is positive (descender below baseline)
    bbox      = _DUMMY_DRAW.textbbox((0, 0), span.word, font=font, anchor="ls")
    w         = bbox[2] - bbox[0]
    ascender  = -bbox[1]   # how far above baseline (positive int)
    descender =  bbox[3]   # how far below baseline (positive int)
    return w, ascender, descender


def _wrap_spans_into_lines(
    spans:       List[WordSpan],
    base_size:   int,
    max_width:   int,
    space_width: int = 14,
) -> List[List[WordSpan]]:
    """
    Groups spans into lines respecting max_width.
    Each word is measured at its actual (potentially scaled) size.
    Returns List[List[WordSpan]] — one inner list per line.
    """
    if not spans:
        return []

    lines:        List[List[WordSpan]] = []
    current_line: List[WordSpan]       = []
    current_w:    int                  = 0

    for span in spans:
        w, _ = _measure_span(span, base_size)
        needed = w + (space_width if current_line else 0)

        if current_line and current_w + needed > max_width:
            lines.append(current_line)
            current_line = [span]
            current_w    = w
        else:
            current_line.append(span)
            current_w += needed

    if current_line:
        lines.append(current_line)

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Word-level renderer
# ─────────────────────────────────────────────────────────────────────────────

def _draw_span(
    draw:       ImageDraw.ImageDraw,
    x:          int,
    y:          int,
    span:       WordSpan,
    base_size:  int,
    stroke_color: Tuple[int, int, int, int],
    stroke_width: int,
) -> int:
    """
    Renders one WordSpan at (x, y).
    Uses 8-direction stroke for clean thick outlines.
    Returns the pixel width consumed (caller advances x cursor).
    """
    size = max(12, int(base_size * span.size_mult))
    font = _get_font(span.font_file, size)

    sw   = stroke_width
    half = max(1, sw // 2)
    offsets = [
        (-sw,   0), (sw,   0), (0,  -sw), (0,   sw),
        (-sw, -sw), (sw,  -sw), (-sw,  sw), (sw,   sw),
        (-sw, -half), (sw, -half), (-sw,  half), (sw,  half),
        (-half, -sw), (half, -sw), (-half,  sw), (half,  sw),
    ]
    for ox, oy in offsets:
        draw.text((x + ox, y + oy), span.word, font=font, fill=stroke_color)
    draw.text((x, y), span.word, font=font, fill=span.color)

    w, _ = getsize(font, span.word)
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Display modes — now produce List[WordSpan] via parse_visual_tags
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_MODES = {"single", "multi", "aligned"}


def _sentence_to_spans(
    sentence:  str,
    style:     dict,
    sentiment: str,
) -> List[WordSpan]:
    """
    Converts a display sentence (possibly with [TAG] markup) to List[WordSpan].
    Central helper used by all three display modes.
    """
    return parse_visual_tags(sentence, style, sentiment)


def render_multi_mode(
    sentence:  str,
    style:     dict,
    sentiment: str,
    audio_idx: int,
    start_idx: int,
    is_slow:   bool,
) -> List[RenderJob]:
    """
    Full sentence on one image. 1 RenderJob, time_fraction = 1.0.
    Best for: funny, sad, wholesome, happy.
    """
    spans = _sentence_to_spans(sentence, style, sentiment)
    if not spans:
        spans = _sentence_to_spans("...", style, sentiment)

    return [RenderJob(
        idx          = start_idx,
        spans        = spans,
        timing_type  = "fraction",
        is_slow      = is_slow,
        audio_idx    = audio_idx,
        time_fraction = 1.0,
    )]


def render_single_mode(
    sentence:  str,
    style:     dict,
    sentiment: str,
    audio_idx: int,
    start_idx: int,
    is_slow:   bool,
) -> List[RenderJob]:
    """
    Sentence split into word chunks, one RenderJob per chunk.
    Each chunk shown for (1/N) of the audio duration.
    Best for: scary, dramatic, angry, mysterious.
    """
    wpc   = style["words_per_chunk"]
    spans = _sentence_to_spans(sentence, style, sentiment)
    if not spans:
        spans = _sentence_to_spans("...", style, sentiment)

    # Split spans into chunks of wpc words
    chunks = [spans[i:i + wpc] for i in range(0, len(spans), wpc)]
    chunks = [c for c in chunks if c]
    if not chunks:
        chunks = [spans]

    n        = len(chunks)
    fraction = 1.0 / n

    return [
        RenderJob(
            idx           = start_idx + i,
            spans         = chunk,
            timing_type   = "fraction",
            is_slow       = is_slow,
            audio_idx     = audio_idx,
            time_fraction = fraction,
        )
        for i, chunk in enumerate(chunks)
    ]


def render_aligned_mode(
    sentence:         str,
    style:            dict,
    sentiment:        str,
    audio_idx:        int,
    start_idx:        int,
    is_slow:          bool,
    word_timestamps:  List[dict],
    audio_start_time: float,
    audio_duration:   float,
) -> List[RenderJob]:
    """
    Word-level aligned mode using WhisperX timestamps.

    Groups consecutive word timestamps into chunks of words_per_chunk.
    Each chunk's clip_start = timestamp of first word in chunk.
    Each chunk's clip_end   = timestamp of last word + its duration.

    Falls back to single mode per sentence if timestamps missing.
    """
    wpc = style["words_per_chunk"]

    if not word_timestamps:
        return render_single_mode(
            sentence, style, sentiment, audio_idx, start_idx, is_slow,
        )

    all_spans = _sentence_to_spans(sentence, style, sentiment)
    if not all_spans:
        all_spans = _sentence_to_spans("...", style, sentiment)

    # Build a span lookup by word text (best-effort — handles repeated words
    # by consuming spans in order rather than matching by text)
    span_queue = list(all_spans)

    jobs: List[RenderJob] = []
    n = len(word_timestamps)

    for chunk_start in range(0, n, wpc):
        chunk_ts = word_timestamps[chunk_start:chunk_start + wpc]
        if not chunk_ts:
            continue

        # Pull the next wpc spans from the queue (order-preserving)
        chunk_spans = span_queue[:wpc]
        span_queue  = span_queue[wpc:]

        # If DeepSeek added/removed words vs WhisperX, queue may run out.
        # Fall back to untagged spans for the remaining timestamps.
        if not chunk_spans:
            from utils.sentiment_map import STYLE_MAP, DEFAULT_STYLE
            base_style = STYLE_MAP.get(sentiment, DEFAULT_STYLE)
            chunk_spans = [
                WordSpan(
                    word      = ts["word"],
                    tag       = None,
                    color     = base_style["fill_color"],
                    font_file = base_style["font_file"],
                    size_mult = 1.0,
                )
                for ts in chunk_ts
            ]

        clip_start = audio_start_time + chunk_ts[0]["start"]

        if chunk_start + wpc < n:
            clip_end = audio_start_time + word_timestamps[chunk_start + wpc]["start"]
        else:
            last_end = chunk_ts[-1].get("end", chunk_ts[-1]["start"] + 0.3)
            clip_end = audio_start_time + last_end

        audio_end = audio_start_time + audio_duration
        clip_end  = min(clip_end, audio_end)
        clip_end  = max(clip_end, clip_start + 0.1)

        jobs.append(RenderJob(
            idx        = start_idx + len(jobs),
            spans      = chunk_spans,
            timing_type = "absolute",
            is_slow    = is_slow,
            clip_start = round(clip_start, 3),
            clip_end   = round(clip_end,   3),
        ))

    return jobs if jobs else render_single_mode(
        sentence, style, sentiment, audio_idx, start_idx, is_slow,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

def get_render_jobs(
    sentences:         List[str],
    style:             dict,
    sentiment:         str        = "dramatic",
    mp3_dir:           Optional[str]        = None,
    audio_start_times: Optional[List[float]] = None,
    audio_durations:   Optional[List[float]] = None,
    is_slow_flags:     Optional[List[bool]]  = None,
) -> List[RenderJob]:
    """
    Route each sentence to the correct renderer.
    Returns a flat ordered list of all RenderJobs.

    New parameter vs original:
      sentiment     → passed to parse_visual_tags for correct tag styling
      is_slow_flags → one bool per sentence, from EnhancedSentence.is_slow

    For "aligned" mode, loads word timestamps from
    {mp3_dir}/postaudio-{i}_words.json written by engine_wrapper.
    Falls back to "single" mode per sentence if timestamps missing.
    """
    mode = style.get("display_mode", "multi")
    if mode not in DISPLAY_MODES:
        print(f"[caption_renderer] Unknown display_mode '{mode}', using 'multi'")
        mode = "multi"

    all_jobs:    List[RenderJob] = []
    img_counter: int             = 0

    for audio_idx, sentence in enumerate(sentences):

        is_slow = bool(is_slow_flags[audio_idx]) if is_slow_flags else False

        if mode == "aligned" and mp3_dir and audio_start_times and audio_durations:
            from utils.whisper_aligner import load_word_timestamps
            audio_path = os.path.join(mp3_dir, f"postaudio-{audio_idx}.mp3")
            word_ts    = load_word_timestamps(audio_path)

            if word_ts:
                jobs = render_aligned_mode(
                    sentence         = sentence,
                    style            = style,
                    sentiment        = sentiment,
                    audio_idx        = audio_idx,
                    start_idx        = img_counter,
                    is_slow          = is_slow,
                    word_timestamps  = word_ts,
                    audio_start_time = audio_start_times[audio_idx],
                    audio_duration   = audio_durations[audio_idx],
                )
            else:
                print(
                    f"[caption_renderer] No timestamps for sentence {audio_idx},"
                    f" using single mode"
                )
                jobs = render_single_mode(
                    sentence, style, sentiment, audio_idx, img_counter, is_slow,
                )

        elif mode == "single":
            jobs = render_single_mode(
                sentence, style, sentiment, audio_idx, img_counter, is_slow,
            )

        else:
            jobs = render_multi_mode(
                sentence, style, sentiment, audio_idx, img_counter, is_slow,
            )

        all_jobs.extend(jobs)
        img_counter += len(jobs)

    return all_jobs


# ─────────────────────────────────────────────────────────────────────────────
# Fit font — determines base size that fits the canvas
# ─────────────────────────────────────────────────────────────────────────────

def fit_base_size(
    style:           dict,
    spans:           List[WordSpan],
    canvas_w:        int,
    canvas_h:        int,
    line_spacing:    int,
    max_width_ratio: float = 0.88,
    max_height_ratio: float = 0.45,
) -> int:
    """
    Binary-search the largest base_size where all spans fit on canvas.
    Returns the base font size (int).
    Individual spans multiply this by their size_mult.
    """
    max_w = int(canvas_w * max_width_ratio)
    max_h = int(canvas_h * max_height_ratio)

    font_size = style.get("font_size", 90)

    while font_size > 28:
        lines = _wrap_spans_into_lines(spans, font_size, max_w)
        # Measure total height using proper ascender+descender metrics
        total_h = 0
        for i, line in enumerate(lines):
            # line height = max_ascender + max_descender across all spans
            max_asc = max((_get_span_baseline_offset(sp, font_size)[1] for sp in line), default=0)
            max_dsc = max((_get_span_baseline_offset(sp, font_size)[2] for sp in line), default=0)
            line_h  = max_asc + max_dsc
            total_h += line_h
            if i < len(lines) - 1:
                total_h += line_spacing
        # Measure max line width
        max_line_w = 0
        for line in lines:
            lw = sum(_get_span_baseline_offset(sp, font_size)[0] for sp in line)
            lw += 14 * max(0, len(line) - 1)
            if lw > max_line_w:
                max_line_w = lw

        if max_line_w <= max_w and total_h <= max_h:
            return font_size
        font_size -= 4

    return 28


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render_job_to_image(
    job:          RenderJob,
    style:        dict,
    canvas_w:     int,
    canvas_h:     int,
    line_spacing: int,
) -> Image.Image:
    """
    Renders one RenderJob to a transparent RGBA PNG.

    Per-word rendering with baseline-anchored mixed-size text:
      1. Determine base_size via fit_base_size()
      2. Wrap spans into lines respecting canvas_w * 0.88
      3. For each line:
           a. Find max_height = tallest word in line (at its size_mult)
           b. y_baseline = current y + max_height
           c. For each word: render at (x, y_baseline - word_height)
              → all word bottoms align on the same baseline
           d. x cursor advances by word_width + space_width
      4. Centered horizontally per line.

    Stroke color comes from style["stroke_color"] (sentiment-level, consistent).
    Fill color comes from each WordSpan individually.
    """
    if not job.spans:
        # Empty job — return blank transparent image
        return Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    image  = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw   = ImageDraw.Draw(image)

    stroke_color = style.get("stroke_color", (0, 0, 0, 255))
    stroke_width = style.get("stroke_width", 4)
    space_width  = 14  # pixels between words

    max_w     = int(canvas_w * 0.88)
    base_size = fit_base_size(style, job.spans, canvas_w, canvas_h, line_spacing)
    lines     = _wrap_spans_into_lines(job.spans, base_size, max_w, space_width)

    if not lines:
        return image

    # ── Pre-compute per-line metrics using true ascender/descender ────────────
    #
    # For each line:
    #   line_ascender  = max ascender  across all spans in that line
    #   line_descender = max descender across all spans in that line
    #   line_height    = line_ascender + line_descender
    #
    # Rendering a word:
    #   baseline_y = cy + line_ascender          (shared for all words in line)
    #   word_y     = baseline_y - span_ascender  (each word placed from its own ascender)
    #
    # This guarantees all glyphs sit on the same optical baseline regardless of
    # font family, weight, or size_mult differences between spans.
    line_metrics = []
    for line in lines:
        metrics = [_get_span_baseline_offset(sp, base_size) for sp in line]
        max_asc = max(m[1] for m in metrics) if metrics else base_size
        max_dsc = max(m[2] for m in metrics) if metrics else 0
        line_metrics.append((max_asc, max_dsc, metrics))

    total_h  = sum(asc + dsc for asc, dsc, _ in line_metrics)
    total_h += line_spacing * max(0, len(lines) - 1)
    anchor_y = int(canvas_h * style.get("y_position", 0.65)) - (total_h // 2)

    # ── Render line by line ───────────────────────────────────────────────────
    cy = anchor_y

    for line_idx, line in enumerate(lines):
        line_asc, line_dsc, span_metrics = line_metrics[line_idx]
        line_h = line_asc + line_dsc

        # Shared baseline for every word in this line
        baseline_y = cy + line_asc

        # Measure total line width for horizontal centering
        line_w = sum(m[0] for m in span_metrics)
        line_w += space_width * max(0, len(line) - 1)
        x = (canvas_w - line_w) // 2

        for span_idx, span in enumerate(line):
            span_w, _, _ = span_metrics[span_idx]
            size = max(12, int(base_size * span.size_mult))
            font = _get_font(span.font_file, size)

            # ── THE FIX ──────────────────────────────────────────────────────
            # Pass the SAME baseline_y to draw.text for EVERY word in the line.
            # anchor="ls" tells Pillow: "x=left edge, y=typographic baseline".
            # Pillow then places each glyph correctly relative to ITS OWN font
            # metrics — so all words sit on the same optical baseline line,
            # regardless of font family, weight, or size_mult differences.
            #
            # DO NOT compute word_y = baseline_y - span_ascender.
            # That reintroduces per-font ascender differences (the original bug).
            # ─────────────────────────────────────────────────────────────────

            # Draw 8-direction stroke
            sw   = stroke_width
            half = max(1, sw // 2)
            offsets = [
                (-sw,   0), (sw,   0), (0,  -sw), (0,   sw),
                (-sw, -sw), (sw,  -sw), (-sw,  sw), (sw,   sw),
                (-sw, -half), (sw, -half), (-sw,  half), (sw,  half),
                (-half, -sw), (half, -sw), (-half,  sw), (half,  sw),
            ]
            for ox, oy in offsets:
                draw.text(
                    (x + ox, baseline_y + oy),
                    span.word, font=font,
                    fill=stroke_color, anchor="ls",
                )

            # Draw fill — same baseline_y for every word
            draw.text((x, baseline_y), span.word, font=font,
                      fill=span.color, anchor="ls")

            x += span_w + space_width

        cy += line_h + line_spacing

    return image