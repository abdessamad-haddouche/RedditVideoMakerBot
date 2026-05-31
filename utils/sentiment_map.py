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
# STYLE_MAP
# ─────────────────────────────────────────────────────────────────────────────
#
# display_mode options:
#
#   "aligned"  → WhisperX word timestamps — perfect sync with any TTS.
#                Falls back to "single" per sentence if timestamps unavailable.
#                USE THIS for all sentiments once WhisperX is installed.
#
#   "single"   → Split sentence into word chunks, equal time per chunk.
#                Good fallback when WhisperX is not installed.
#
#   "multi"    → Full sentence on one image. No splitting.
#                Best for slow TTS or wholesome/sad content.
#
# words_per_chunk:
#   In "aligned" mode: words grouped per visible chunk (3-5 recommended)
#   In "single" mode:  words per chunk (higher = fewer chunks = slower pace)
#   In "multi" mode:   words per line in the wrapped text block
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