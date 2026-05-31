"""
sentiment.py
────────────
Single responsibility: one DeepSeek call that does everything.

Public API
──────────
  apply_sentiment_config(reddit_object)
      Called once from main.py before TTS runs.
      Mutates reddit_object in-place:
        - reddit_object["thread_post"]  → List[EnhancedSentence]
        - reddit_object["thread_title"] → rewritten title
      Also sets in-memory config overrides (background, voice, sentiment label)
      and saves metadata.json.

  EnhancedSentence
      Dataclass consumed by engine_wrapper.py (.tts)
      and imagenarator.py (.display).

Internal
────────
  enhance_post()               → single DeepSeek call, returns raw dict
  get_emphasis_instruction()   → engine-aware voice cue prompt block
  _build_prompt()              → assembles the full prompt
  _parse_response()            → validates + returns parsed dict
  _fallback_enhance()          → safe fallback if API fails
  _apply_config_overrides()    → sets background/voice in-memory
  _save_metadata()             → saves metadata.json to results/

Design decisions
────────────────
  - Two separate DeepSeek calls (old detect_sentiment + generate_metadata)
    are merged into ONE call. Half the latency, half the cost, one failure point.
  - reddit_object mutation is intentional: both engine_wrapper and imagenarator
    read thread_post directly. Mutating here means zero changes downstream.
  - EnhancedSentence is a dataclass, not a dict, so callers can use hasattr()
    to detect enhanced vs plain-string fallback cleanly.
  - All failures fall back silently — video generation NEVER crashes because
    of the enhancement layer.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from utils import settings
from utils.console import print_step, print_substep
from utils.sentiment_map import (
    BACKGROUND_MAP,
    OPENAI_VOICE_MAP,
    ELEVENLABS_VOICE_MAP,
    VALID_SENTIMENTS,
    DEFAULT_SENTIMENT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public dataclass — consumed by engine_wrapper.py and imagenarator.py
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnhancedSentence:
    """
    One sentence of the Reddit post, enhanced by DeepSeek.

    tts:
        Text sent to the TTS engine.
        Contains voice delivery cues appropriate for the active engine:
          - OpenAI/Qwen: CAPS for stress, ... for pauses
          - ElevenLabs:  <break/> and <emphasis> SSML tags
          - Google/other: clean text only (no cues)
        Never contains visual tags like [EMPHASIZE] or [SHOCK].

    display:
        Text sent to caption_renderer.py.
        Contains visual tags:
          [HOOK]...[/HOOK]           sentence[0] only
          [EMPHASIZE]...[/EMPHASIZE] 1-2 key words per sentence
          [SHOCK]...[/SHOCK]         reveals/twists only
          [SLOW]...[/SLOW]           sentence-level pacing hint
        Never contains voice delivery cues.

    is_slow:
        True when display contains a [SLOW] tag.
        Pre-computed here so imagenarator doesn't need to re-parse.
    """
    tts:     str
    display: str
    is_slow: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Voice delivery instruction — engine-aware
# ─────────────────────────────────────────────────────────────────────────────

def get_emphasis_instruction(voice_choice: str) -> str:
    """
    Returns the voice delivery rules block for the DeepSeek prompt.
    Injected into SECTION 3 of the prompt depending on active TTS engine.

    Pure function — no side effects, no config access.
    Adding a new engine: add an elif branch below.

    Args:
        voice_choice: value of settings.config["settings"]["tts"]["voice_choice"]

    Returns:
        str: Multi-line instruction block for inclusion in the DeepSeek prompt.
    """
    vc = voice_choice.lower().strip()

    if vc in ("openai", "qwen"):
        return (
            "VOICE DELIVERY RULES (OpenAI / Qwen TTS — punctuation-based):\n"
            "- Use ... for dramatic pauses. Maximum 1 per sentence.\n"
            "- Use ALL CAPS for words that need vocal stress. Maximum 2 per sentence.\n"
            "- Use ?! for shocked or disbelieving delivery.\n"
            "- Do NOT use em dashes (—). They are stripped by the pipeline.\n"
            "- Do NOT put any of these in the display field. tts field only.\n"
            "- Example tts: 'I opened the door... and my heart STOPPED.'\n"
            "- Example display: 'I opened the door and my heart [SHOCK]STOPPED[/SHOCK].'"
        )

    if vc == "elevenlabs":
        return (
            "VOICE DELIVERY RULES (ElevenLabs — SSML tags):\n"
            "- Use <break time=\"0.5s\"/> for short pauses.\n"
            "- Use <break time=\"1s\"/> for dramatic pauses. Maximum 1 per sentence.\n"
            "- Use <emphasis level=\"strong\">word</emphasis> for stressed delivery.\n"
            "- Do NOT put SSML tags in the display field. tts field only.\n"
            "- Example tts: 'I opened the door<break time=\"0.8s\"/> "
            "and my heart <emphasis level=\"strong\">stopped</emphasis>.'\n"
            "- Example display: 'I opened the door and my heart [SHOCK]STOPPED[/SHOCK].'"
        )

    # googletranslate, tiktok, awspolly, streamlabspolly, pyttsx, and any future engine
    return (
        "VOICE DELIVERY RULES (clean text engine):\n"
        "- Write clean, natural sentences only.\n"
        "- No special formatting, punctuation tricks, or markup.\n"
        "- The tts field should be the display field with all visual tags stripped.\n"
        "- Example tts: 'I opened the door and my heart stopped.'\n"
        "- Example display: 'I opened the door and my heart [SHOCK]stopped[/SHOCK].'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek client
# ─────────────────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    api_key = settings.config["deepseek"]["api_key"]
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


def _extract_raw_text(reddit_object: dict) -> tuple:
    """
    Pull title and body from the reddit_object.
    Handles both List[dict] and plain str for thread_post.

    Title-only posts (no body text) are common on some subreddits.
    In that case post falls back to the title so DeepSeek still runs
    and produces proper sentences + metadata instead of crashing.

    Returns (title: str, post: str) — post is NEVER empty if title exists.
    """
    title = reddit_object.get("thread_title", "").strip()
    raw   = reddit_object.get("thread_post", "")

    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict):
                t = item.get("text", "").strip()
                if t:
                    parts.append(t)
            elif isinstance(item, str):
                t = item.strip()
                if t:
                    parts.append(t)
        post = " ".join(parts).strip()
    elif isinstance(raw, str):
        post = raw.strip()
    else:
        post = ""

    # Title-only post: use the title as the post body so DeepSeek always
    # has content to work with and produces real sentences.
    if not post and title:
        post = title

    return title, post


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(title: str, post: str, voice_choice: str, channel_name: str) -> str:
    """
    Assembles the complete DeepSeek prompt.
    Separated from the API call so it can be tested and logged independently.
    """
    emphasis_block = get_emphasis_instruction(voice_choice)

    return f"""You are a Reddit story video script enhancer.
You receive a raw Reddit post and return enhanced text optimized for
faceless story videos on YouTube and TikTok.
Return ONLY valid JSON. No markdown. No ``` fences. No explanation.
Just the raw JSON object.

═══════════════════════════════════════════════════
SECTION 1 — HUMANIZATION RULES (always applied)
═══════════════════════════════════════════════════
Rewrite the post into clean, punchy sentences optimized for video narration.

Expand common Reddit shorthands:
  20F → "a 20-year-old woman"
  30M → "a 30-year-old man"
  AITA → "Am I wrong here"
  NTA → "you're not wrong"
  YTA → "you're at fault"
  ESH → "everyone's at fault here"
  NAH → "nobody's wrong here"
  OP → "the original poster"
  SO → "significant other"
  BF/GF → "boyfriend" / "girlfriend"
  MIL/FIL → "mother-in-law" / "father-in-law"

Sentence rules:
  - Maximum ~15 words per sentence. Split longer sentences at natural breath points.
  - Split at: conjunctions (and, but, so, because), commas, semicolons.
  - Never split in the middle of a proper name or quoted speech.
  - Add minimal context only where it genuinely helps story clarity. Never invent facts.
  - Preserve the original meaning, voice, and emotional tone exactly.
  - Remove Reddit formatting: asterisks, bullet points, excessive line breaks.

═══════════════════════════════════════════════════
SECTION 2 — SENTIMENT CLASSIFICATION
═══════════════════════════════════════════════════
Classify the overall post into exactly one of:
  sad, happy, angry, mysterious, funny, dramatic, wholesome, scary

═══════════════════════════════════════════════════
SECTION 3 — VOICE DELIVERY RULES
═══════════════════════════════════════════════════
{emphasis_block}

═══════════════════════════════════════════════════
SECTION 4 — VISUAL TAGGING RULES
═══════════════════════════════════════════════════
Add visual emphasis tags to the display field only.

Available tags:
  [HOOK]sentence[/HOOK]
    - sentence[0] ONLY. Use when the opening line is genuinely gripping.
    - Do NOT use if the opener is weak. Leave untagged.
    - Wraps the entire sentence.

  [EMPHASIZE]word[/EMPHASIZE]
    - 1 to 2 key words per sentence. Maximum.
    - Use for words that carry the emotional weight of the sentence.
    - Never tag filler words (the, a, I, was, is, etc.).

  [SHOCK]phrase[/SHOCK]
    - Reveals, plot twists, genuinely shocking facts.
    - Use SPARINGLY — maximum 1 per 4 sentences across the whole post.
    - Can wrap 1-3 words. Never an entire sentence.

  [SLOW]sentence[/SLOW]
    - The single sentence with the most emotional weight in the post.
    - Use ONCE per post, maximum.
    - Wraps the entire sentence.

Hard rules:
  - tts field: voice cues ONLY. Zero visual tags.
  - display field: visual tags ONLY. Zero voice cues.
  - Never stack multiple tags on the same word.
  - Unknown or invented tag names are forbidden.

═══════════════════════════════════════════════════
SECTION 5 — SOCIAL MEDIA METADATA
═══════════════════════════════════════════════════
Channel name: {channel_name}

Generate:
  - youtube_title: under 70 characters, click-worthy, no clickbait lies
  - youtube_description: 2-3 sentences + call to action + hashtags
  - tiktok_caption: under 150 characters total including hashtags
  - instagram_caption: engaging, 1-2 sentences + hashtags
  - facebook_caption: conversational, 1-2 sentences
  - hashtags: list of 8-12 relevant hashtag strings

═══════════════════════════════════════════════════
SECTION 6 — OUTPUT FORMAT (strict)
═══════════════════════════════════════════════════
Return exactly this JSON structure. No extra keys. No missing keys.

{{
  "sentiment": "one of: sad/happy/angry/mysterious/funny/dramatic/wholesome/scary",
  "rewritten_title": "rewritten version of the post title, optimized for CTR",
  "sentences": [
    {{
      "tts":     "sentence with voice cues if applicable, no visual tags",
      "display": "sentence with visual tags if applicable, no voice cues"
    }}
  ],
  "metadata": {{
    "youtube_title": "...",
    "youtube_description": "...",
    "tiktok_caption": "...",
    "instagram_caption": "...",
    "facebook_caption": "...",
    "hashtags": ["#reddit", "..."]
  }}
}}

═══════════════════════════════════════════════════
INPUT
═══════════════════════════════════════════════════
Title: {title}

Post:
{post[:2000]}"""


# ─────────────────────────────────────────────────────────────────────────────
# Response parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict:
    """
    Validates and parses the DeepSeek JSON response.

    Strips markdown fences if present.
    Validates all required keys.
    Raises ValueError with a descriptive message on any validation failure
    so the caller can fall back cleanly.
    """
    # Strip markdown fences
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e}") from e

    # Validate top-level keys
    required_top = {"sentiment", "rewritten_title", "sentences", "metadata"}
    missing = required_top - set(data.keys())
    if missing:
        raise ValueError(f"Missing top-level keys: {missing}")

    # Validate sentiment
    if data["sentiment"] not in VALID_SENTIMENTS:
        raise ValueError(
            f"Invalid sentiment '{data['sentiment']}'. "
            f"Must be one of: {VALID_SENTIMENTS}"
        )

    # Validate sentences
    if not isinstance(data["sentences"], list) or len(data["sentences"]) == 0:
        raise ValueError("'sentences' must be a non-empty list")

    for i, s in enumerate(data["sentences"]):
        if not isinstance(s, dict):
            raise ValueError(f"sentences[{i}] is not a dict")
        if "tts" not in s or "display" not in s:
            raise ValueError(f"sentences[{i}] missing 'tts' or 'display' key")
        if not isinstance(s["tts"], str) or not isinstance(s["display"], str):
            raise ValueError(f"sentences[{i}] tts/display must be strings")

    # Validate metadata — fill missing keys with empty strings rather than failing
    required_meta = {
        "youtube_title", "youtube_description",
        "tiktok_caption", "instagram_caption",
        "facebook_caption", "hashtags",
    }
    if not isinstance(data.get("metadata"), dict):
        data["metadata"] = {}
    for key in required_meta:
        if key not in data["metadata"]:
            data["metadata"][key] = [] if key == "hashtags" else ""

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Fallback — never crashes, always returns something usable
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_enhance(reddit_object: dict) -> dict:
    """
    Safe fallback when DeepSeek fails entirely.
    Returns a dict in the same shape as a successful _parse_response() result,
    but with raw text — no enhancement, no visual tags.
    """
    title = reddit_object.get("thread_title", "Reddit Story")
    raw_post = reddit_object.get("thread_post", [])

    # Extract raw sentences from whatever format thread_post is in
    raw_sentences: List[str] = []
    if isinstance(raw_post, list):
        for item in raw_post:
            if isinstance(item, dict):
                text = item.get("text", "").strip()
            elif isinstance(item, str):
                text = item.strip()
            else:
                text = str(item).strip()
            if text:
                raw_sentences.append(text)
    elif isinstance(raw_post, str) and raw_post.strip():
        raw_sentences = [raw_post.strip()]

    # Final safety net: if thread_post was completely empty (title-only post),
    # use the title itself so TTS always gets real text, never an empty string.
    if not raw_sentences:
        if title and title.strip():
            raw_sentences = [title.strip()]
        else:
            raw_sentences = ["This is a Reddit story."]

    # Strip out any empty strings that slipped through
    raw_sentences = [s for s in raw_sentences if s.strip()]
    if not raw_sentences:
        raw_sentences = ["This is a Reddit story."]

    sentences = [
        {"tts": s, "display": s}
        for s in raw_sentences
    ]

    return {
        "sentiment":       DEFAULT_SENTIMENT,
        "rewritten_title": title,
        "sentences":       sentences,
        "metadata": {
            "youtube_title":       title[:70],
            "youtube_description": f"{title}\n\n#reddit #stories",
            "tiktok_caption":      f"{title[:100]} #reddit #storytime",
            "instagram_caption":   f"{title[:100]} #reddit #stories",
            "facebook_caption":    title,
            "hashtags":            ["#reddit", "#storytime", "#stories", "#redditstories"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core enhancement function — single DeepSeek call
# ─────────────────────────────────────────────────────────────────────────────

def enhance_post(reddit_object: dict) -> dict:
    """
    Single DeepSeek call that does everything:
      - Humanizes and rewrites the post for video narration
      - Adds voice delivery cues appropriate for the active TTS engine
      - Adds visual emphasis tags for caption_renderer
      - Detects sentiment
      - Generates social media metadata

    Returns a parsed dict (same shape as _parse_response output).
    Falls back to _fallback_enhance() on any failure — never raises.

    This function replaces the old detect_sentiment() + generate_metadata()
    two-call pattern.
    """
    api_key = settings.config["deepseek"].get("api_key", "")
    if not api_key:
        print_substep(
            "No DeepSeek API key. Using fallback (no enhancement).",
            style="yellow",
        )
        return _fallback_enhance(reddit_object)

    voice_choice  = settings.config["settings"]["tts"].get("voice_choice", "googletranslate")
    channel_name  = settings.config["settings"].get("channel_name", "Reddit Tales")
    title, post   = _extract_raw_text(reddit_object)

    # post is guaranteed non-empty because _extract_raw_text falls back to title
    # for title-only posts. Guard kept for truly pathological edge cases only.
    if not post.strip():
        print_substep("Post has no content at all. Using fallback.", style="yellow")
        return _fallback_enhance(reddit_object)

    prompt = _build_prompt(title, post, voice_choice, channel_name)

    try:
        client   = _get_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role":    "system",
                    "content": (
                        "You are a Reddit story video script enhancer. "
                        "Return ONLY valid JSON. No markdown, no explanation, "
                        "no ``` fences. Just the raw JSON object."
                    ),
                },
                {
                    "role":    "user",
                    "content": prompt,
                },
            ],
            max_tokens=4000,
            temperature=0.7,
        )

        raw = response.choices[0].message.content
        return _parse_response(raw)

    except Exception as e:
        print_substep(
            f"DeepSeek enhancement failed: {e}. Using fallback.",
            style="yellow",
        )
        return _fallback_enhance(reddit_object)


# ─────────────────────────────────────────────────────────────────────────────
# Config overrides — applied in-memory, never written to disk
# ─────────────────────────────────────────────────────────────────────────────

def _apply_config_overrides(sentiment: str, voice_choice: str) -> str:
    """
    Overrides background video/audio and voice settings in-memory for this run.
    Returns the voice name that was set (for logging).
    """
    # Sentiment label stored in memory — STYLE_MAP lookups depend on this
    settings.config["settings"]["sentiment"] = sentiment

    # Background
    bg_video, bg_audio = BACKGROUND_MAP[sentiment]
    settings.config["settings"]["background"]["background_video"] = bg_video
    settings.config["settings"]["background"]["background_audio"] = bg_audio

    # Voice
    vc = voice_choice.lower()
    if vc == "elevenlabs":
        voice = ELEVENLABS_VOICE_MAP[sentiment]
        settings.config["settings"]["tts"]["elevenlabs_voice_name"] = voice
    elif vc in ("openai", "qwen"):
        voice = OPENAI_VOICE_MAP[sentiment]
        settings.config["settings"]["tts"]["openai_voice_name"] = voice
    else:
        voice = f"(no override for {voice_choice})"

    return voice


# ─────────────────────────────────────────────────────────────────────────────
# Metadata persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_metadata(metadata: dict, reddit_object: dict, sentiment: str) -> None:
    """
    Saves metadata.json to the per-video results folder.
    Path: results/{subreddit}/{thread_id}_{background_video}/metadata.json
    """
    try:
        subreddit  = reddit_object.get(
            "thread_subreddit",
            settings.config["reddit"]["thread"]["subreddit"],
        )
        thread_id  = reddit_object.get("thread_id", "unknown")
        bg_video   = settings.config["settings"]["background"].get("background_video", "unknown")
        folder     = f"results/{subreddit}/{thread_id}_{bg_video}"

        os.makedirs(folder, exist_ok=True)
        filepath = f"{folder}/metadata.json"

        payload = {**metadata, "sentiment": sentiment}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print_substep(f"Metadata saved → {filepath}", style="bold green")
    except Exception as e:
        print_substep(f"Failed to save metadata: {e}", style="yellow")


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced post persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_enhanced_post(enhanced: dict, reddit_object: dict) -> None:
    """
    Saves enhanced_post.json to the per-video results folder.

    Structure:
      {
        "original_title":   "...",        ← raw Reddit title before rewrite
        "rewritten_title":  "...",        ← DeepSeek rewritten title
        "sentiment":        "dramatic",
        "sentences": [
          {
            "index":   0,
            "tts":     "...",             ← what TTS engine receives
            "display": "...",             ← what caption renderer receives (with [TAGS])
            "is_slow": false
          },
          ...
        ]
      }

    This file is for inspection/debugging only.
    It has no effect on the pipeline.
    """
    try:
        subreddit = reddit_object.get(
            "thread_subreddit",
            settings.config["reddit"]["thread"]["subreddit"],
        )
        thread_id = reddit_object.get("thread_id", "unknown")
        bg_video  = settings.config["settings"]["background"].get("background_video", "unknown")
        folder    = f"results/{subreddit}/{thread_id}_{bg_video}"

        os.makedirs(folder, exist_ok=True)
        filepath = f"{folder}/enhanced_post.json"

        payload = {
            "original_title":  reddit_object.get("thread_title", ""),
            "rewritten_title": enhanced.get("rewritten_title", ""),
            "sentiment":       enhanced.get("sentiment", "dramatic"),
            "sentences": [
                {
                    "index":   i,
                    "tts":     s.get("tts",     ""),
                    "display": s.get("display", ""),
                    "is_slow": "[SLOW]" in s.get("display", ""),
                }
                for i, s in enumerate(enhanced.get("sentences", []))
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print_substep(f"Enhanced post saved → {filepath}", style="bold green")
    except Exception as e:
        print_substep(f"Failed to save enhanced post: {e}", style="yellow")


# ─────────────────────────────────────────────────────────────────────────────
# reddit_object mutation
# ─────────────────────────────────────────────────────────────────────────────

def _build_enhanced_sentences(raw_sentences: list) -> List[EnhancedSentence]:
    """
    Converts the raw sentences list from the DeepSeek response
    into List[EnhancedSentence].

    Also pre-computes is_slow by checking for [SLOW] tag in display text —
    saves imagenarator from doing a regex pass later.
    """
    result = []
    for s in raw_sentences:
        tts     = s.get("tts",     "").strip()
        display = s.get("display", "").strip()

        # Defensive: if either field is empty, use the other
        if not tts and display:
            tts = display
        if not display and tts:
            display = tts

        is_slow = "[SLOW]" in display

        result.append(EnhancedSentence(
            tts=tts,
            display=display,
            is_slow=is_slow,
        ))
    return result


def _mutate_reddit_object(reddit_object: dict, enhanced: dict) -> None:
    """
    Replaces reddit_object["thread_post"] with List[EnhancedSentence]
    and reddit_object["thread_title"] with the rewritten title.

    This is the single mutation point. After this call:
      - engine_wrapper.py reads item.tts  (via hasattr check)
      - imagenarator.py  reads item.display (via hasattr check)
    Both files have zero changes beyond those hasattr checks.
    """
    reddit_object["thread_post"]  = _build_enhanced_sentences(enhanced["sentences"])
    reddit_object["thread_title"] = enhanced.get(
        "rewritten_title",
        reddit_object.get("thread_title", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — called from main.py
# ─────────────────────────────────────────────────────────────────────────────

def apply_sentiment_config(reddit_object: dict) -> None:
    """
    Main entry point. Called once from main.py before TTS runs.

    Flow:
      1. Single DeepSeek call → enhanced post data
      2. Apply in-memory config overrides (background, voice, sentiment label)
      3. Mutate reddit_object in-place with enhanced sentences + rewritten title
      4. Save metadata.json

    On any failure: falls back to raw text, no enhancement, video still generates.
    Never raises. Never crashes the pipeline.
    """
    print_step("Enhancing post with DeepSeek... ✍️")

    voice_choice = settings.config["settings"]["tts"].get("voice_choice", "googletranslate")

    # ── Single DeepSeek call ─────────────────────────────────────────────────
    enhanced  = enhance_post(reddit_object)
    sentiment = enhanced["sentiment"]

    # ── In-memory config overrides ───────────────────────────────────────────
    voice = _apply_config_overrides(sentiment, voice_choice)

    # ── Mutate reddit_object ─────────────────────────────────────────────────
    _mutate_reddit_object(reddit_object, enhanced)

    # ── Save metadata + enhanced post ───────────────────────────────────────
    _save_metadata(enhanced["metadata"], reddit_object, sentiment)
    _save_enhanced_post(enhanced, reddit_object)

    # ── Log ──────────────────────────────────────────────────────────────────
    bg_video, bg_audio = BACKGROUND_MAP[sentiment]
    metadata           = enhanced["metadata"]

    print_substep(f"Sentiment detected  : {sentiment} 🎯",                   style="bold green")
    print_substep(f"Sentences enhanced  : {len(reddit_object['thread_post'])}", style="bold green")
    print_substep(f"Background video    : {bg_video}",                        style="bold blue")
    print_substep(f"Background audio    : {bg_audio or 'none'}",              style="bold blue")
    print_substep(f"Voice               : {voice}",                           style="bold blue")
    print_substep(f"YouTube title       : {metadata['youtube_title']}",       style="bold blue")
    print_substep(f"TikTok caption      : {metadata['tiktok_caption']}",      style="bold blue")
    print_substep(
        f"Rewritten title     : {enhanced.get('rewritten_title', '')}",
        style="bold blue",
    )