"""
sentiment_map.py
────────────────
All per-sentiment configuration maps.

Maps:
  BACKGROUND_MAP       → (background_video, background_audio) per sentiment
  OPENAI_VOICE_MAP     → OpenAI voice name per sentiment
  ELEVENLABS_VOICE_MAP → ElevenLabs voice name per sentiment
  STYLE_MAP            → full visual style config per sentiment
  DEFAULT_STYLE        → fallback style (dramatic)

  ── NEW (feature/deepseek-text-enhancement) ──────────────────────────────
  EMPHASIS_STYLE       → [EMPHASIZE] tag styling per sentiment
  SHOCK_STYLE          → [SHOCK] tag styling per sentiment
  HOOK_STYLE           → [HOOK] tag styling per sentiment (sentence-level)
  SLOW_DURATION_MULT   → time multiplier for [SLOW] tagged sentences
  ─────────────────────────────────────────────────────────────────────────

Style entry shape for EMPHASIS_STYLE / SHOCK_STYLE / HOOK_STYLE:
  {
      "color":     (R, G, B, A),   # RGBA tuple
      "font_file": "Filename.ttf", # must exist in /fonts/
      "size_mult": float,          # multiplier on base font_size
  }
"""

# ─────────────────────────────────────────────────────────────────────────────
# Core maps (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

BACKGROUND_MAP = {
    "sad":        ("minecraft",     "lofi"),
    "happy":      ("fall-guys",     "chill-summer"),
    "angry":      ("gta",           "lofi"),
    "mysterious": ("csgo-surf",     "lofi-2"),
    "funny":      ("cluster-truck", "chill-summer"),
    "dramatic":   ("rocket-league", "lofi"),
    "wholesome":  ("steep",         "chill-summer"),
    "scary":      ("minecraft-2",   "lofi-2"),
}

OPENAI_VOICE_MAP = {
    "sad":        "nova",
    "happy":      "shimmer",
    "angry":      "onyx",
    "mysterious": "echo",
    "funny":      "fable",
    "dramatic":   "alloy",
    "wholesome":  "nova",
    "scary":      "onyx",
}

ELEVENLABS_VOICE_MAP = {
    "sad":        "Brian - Deep, Resonant and Comforting",
    "happy":      "Jessica - Playful, Bright, Warm",
    "angry":      "Adam - Dominant, Firm",
    "mysterious": "Callum - Husky Trickster",
    "funny":      "Laura - Enthusiast, Quirky Attitude",
    "dramatic":   "George - Warm, Captivating Storyteller",
    "wholesome":  "Matilda - Knowledgable, Professional",
    "scary":      "Harry - Fierce Warrior",
}

VALID_SENTIMENTS = list(BACKGROUND_MAP.keys())
DEFAULT_SENTIMENT = "dramatic"


# ─────────────────────────────────────────────────────────────────────────────
# STYLE_MAP (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
#
# display_mode options:
#
#   "aligned"  → WhisperX word timestamps — perfect sync with any TTS.
#                Falls back to "single" per sentence if timestamps unavailable.
#
#   "single"   → Split sentence into word chunks, equal time per chunk.
#                Good fallback when WhisperX is not installed.
#
#   "multi"    → Full sentence on one image. No splitting.
#                Best for slow TTS or wholesome/sad content.
#
STYLE_MAP = {

    "dramatic": {
        "font_file":       "Montserrat-ExtraBold.ttf",
        "font_size":       95,
        "fill_color":      (255, 255, 255, 255),
        "stroke_color":    (0,   0,   0,   255),
        "stroke_width":    4,
        "words_per_chunk": 4,
        "y_position":      0.65,
        "uppercase":       False,
        "display_mode":    "aligned",
    },

    "scary": {
        "font_file":       "Oswald-Bold.ttf",
        "font_size":       95,
        "fill_color":      (232, 244, 248, 255),
        "stroke_color":    (0,   0,   0,   255),
        "stroke_width":    5,
        "words_per_chunk": 3,
        "y_position":      0.65,
        "uppercase":       False,
        "display_mode":    "aligned",
    },

    "angry": {
        "font_file":       "Anton-Regular.ttf",
        "font_size":       105,
        "fill_color":      (255, 69,  0,   255),
        "stroke_color":    (0,   0,   0,   255),
        "stroke_width":    5,
        "words_per_chunk": 3,
        "y_position":      0.65,
        "uppercase":       True,
        "display_mode":    "aligned",
    },

    "mysterious": {
        "font_file":       "Raleway-Bold.ttf",
        "font_size":       90,
        "fill_color":      (184, 212, 232, 255),
        "stroke_color":    (0,   0,   0,   255),
        "stroke_width":    4,
        "words_per_chunk": 3,
        "y_position":      0.65,
        "uppercase":       False,
        "display_mode":    "aligned",
    },

    "funny": {
        "font_file":       "Nunito-ExtraBold.ttf",
        "font_size":       90,
        "fill_color":      (255, 230, 0,   255),
        "stroke_color":    (0,   0,   0,   255),
        "stroke_width":    4,
        "words_per_chunk": 5,
        "y_position":      0.65,
        "uppercase":       False,
        "display_mode":    "aligned",
    },

    "sad": {
        "font_file":       "Lato-Bold.ttf",
        "font_size":       88,
        "fill_color":      (220, 225, 255, 255),
        "stroke_color":    (10,  10,  46,  255),
        "stroke_width":    3,
        "words_per_chunk": 5,
        "y_position":      0.65,
        "uppercase":       False,
        "display_mode":    "aligned",
    },

    "wholesome": {
        "font_file":       "Nunito-ExtraBold.ttf",
        "font_size":       88,
        "fill_color":      (255, 248, 231, 255),
        "stroke_color":    (26,  10,  0,   255),
        "stroke_width":    3,
        "words_per_chunk": 5,
        "y_position":      0.65,
        "uppercase":       False,
        "display_mode":    "aligned",
    },

    "happy": {
        "font_file":       "Nunito-ExtraBold.ttf",
        "font_size":       90,
        "fill_color":      (255, 230, 0,   255),
        "stroke_color":    (0,   0,   0,   255),
        "stroke_width":    4,
        "words_per_chunk": 5,
        "y_position":      0.65,
        "uppercase":       False,
        "display_mode":    "aligned",
    },
}

DEFAULT_STYLE = STYLE_MAP["dramatic"]


# ─────────────────────────────────────────────────────────────────────────────
# NEW — Visual emphasis styles (feature/deepseek-text-enhancement)
# ─────────────────────────────────────────────────────────────────────────────
#
# Used by caption_renderer.py when it encounters visual tags in display text.
# Each style entry: { "color": RGBA, "font_file": str, "size_mult": float }
#
# EMPHASIS_STYLE → [EMPHASIZE]word[/EMPHASIZE]
#   Highlights key words. Same size as base, different color + heavier font.
#   Max 1-2 words per sentence — DeepSeek is prompted to be conservative.
#
# SHOCK_STYLE → [SHOCK]phrase[/SHOCK]
#   Reveals, twists, shocking facts. +20% size, aggressive colors.
#   Used sparingly — DeepSeek is prompted: only for genuine story beats.
#
# HOOK_STYLE → [HOOK]sentence[/HOOK]
#   Applied to sentence[0] only when it genuinely earns it.
#   +10% size, accent color. Designed to maximize first-3-seconds retention.
#
# font_file must exist in /fonts/ — all fonts listed here are already present.
# ─────────────────────────────────────────────────────────────────────────────

EMPHASIS_STYLE = {
    "dramatic": {
        "color":     (255, 200, 50,  255),   # gold
        "font_file": "Montserrat-ExtraBold.ttf",
        "size_mult": 1.0,
    },
    "scary": {
        "color":     (200, 220, 255, 255),   # cold blue-white
        "font_file": "Oswald-Bold.ttf",
        "size_mult": 1.0,
    },
    "angry": {
        "color":     (255, 255, 255, 255),   # pure white (pops on orange base)
        "font_file": "Anton-Regular.ttf",
        "size_mult": 1.0,
    },
    "mysterious": {
        "color":     (200, 160, 255, 255),   # soft purple
        "font_file": "Raleway-Bold.ttf",
        "size_mult": 1.0,
    },
    "funny": {
        "color":     (255, 120, 0,   255),   # bright orange
        "font_file": "Nunito-ExtraBold.ttf",
        "size_mult": 1.0,
    },
    "sad": {
        "color":     (150, 180, 255, 255),   # pale blue
        "font_file": "Lato-Bold.ttf",
        "size_mult": 1.0,
    },
    "wholesome": {
        "color":     (100, 220, 120, 255),   # warm green
        "font_file": "Nunito-ExtraBold.ttf",
        "size_mult": 1.0,
    },
    "happy": {
        "color":     (255, 100, 200, 255),   # hot pink (contrast on yellow base)
        "font_file": "Nunito-ExtraBold.ttf",
        "size_mult": 1.0,
    },
}

SHOCK_STYLE = {
    "dramatic": {
        "color":     (255, 50,  50,  255),   # hot red
        "font_file": "Montserrat-ExtraBold.ttf",
        "size_mult": 1.20,
    },
    "scary": {
        "color":     (200, 0,   0,   255),   # blood red
        "font_file": "Oswald-Bold.ttf",
        "size_mult": 1.20,
    },
    "angry": {
        "color":     (255, 255, 0,   255),   # electric yellow (pops on dark bg)
        "font_file": "Anton-Regular.ttf",
        "size_mult": 1.25,
    },
    "mysterious": {
        "color":     (255, 0,   200, 255),   # bright magenta
        "font_file": "Raleway-Bold.ttf",
        "size_mult": 1.20,
    },
    "funny": {
        "color":     (255, 230, 0,   255),   # electric yellow
        "font_file": "Nunito-ExtraBold.ttf",
        "size_mult": 1.25,
    },
    "sad": {
        "color":     (255, 255, 255, 255),   # bright white (breaks through sadness)
        "font_file": "Lato-Bold.ttf",
        "size_mult": 1.15,
    },
    "wholesome": {
        "color":     (255, 210, 80,  255),   # warm gold
        "font_file": "Nunito-ExtraBold.ttf",
        "size_mult": 1.15,
    },
    "happy": {
        "color":     (255, 50,  50,  255),   # hot red (surprise contrast)
        "font_file": "Nunito-ExtraBold.ttf",
        "size_mult": 1.20,
    },
}

HOOK_STYLE = {
    # sentence[0] only — slightly larger, accent color, same font family as base
    "dramatic": {
        "color":     (255, 200, 50,  255),   # same gold as EMPHASIS
        "font_file": "Montserrat-ExtraBold.ttf",
        "size_mult": 1.10,
    },
    "scary": {
        "color":     (200, 220, 255, 255),
        "font_file": "Oswald-Bold.ttf",
        "size_mult": 1.10,
    },
    "angry": {
        "color":     (255, 255, 255, 255),
        "font_file": "Anton-Regular.ttf",
        "size_mult": 1.10,
    },
    "mysterious": {
        "color":     (200, 160, 255, 255),
        "font_file": "Raleway-Bold.ttf",
        "size_mult": 1.10,
    },
    "funny": {
        "color":     (255, 120, 0,   255),
        "font_file": "Nunito-ExtraBold.ttf",
        "size_mult": 1.10,
    },
    "sad": {
        "color":     (150, 180, 255, 255),
        "font_file": "Lato-Bold.ttf",
        "size_mult": 1.10,
    },
    "wholesome": {
        "color":     (100, 220, 120, 255),
        "font_file": "Nunito-ExtraBold.ttf",
        "size_mult": 1.10,
    },
    "happy": {
        "color":     (255, 100, 200, 255),
        "font_file": "Nunito-ExtraBold.ttf",
        "size_mult": 1.10,
    },
}

# How much longer to hold a [SLOW] tagged sentence's caption.
# Applied as a multiplier on time_fraction (fraction mode)
# or on the clip duration (absolute/aligned mode).
# 1.0 = no change, 1.5 = held 50% longer than normal.
SLOW_DURATION_MULT = {
    "dramatic":   1.4,
    "scary":      1.6,
    "mysterious": 1.5,
    "sad":        1.5,
    "wholesome":  1.3,
    "angry":      1.1,
    "funny":      1.0,
    "happy":      1.1,
}