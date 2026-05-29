# Maps sentiment → (background_video, background_audio)
BACKGROUND_MAP = {
    "sad":        ("minecraft",     "lofi"),        # slow, melancholic
    "happy":      ("fall-guys",     "chill-summer"),# upbeat, fun
    "angry":      ("gta",           "lofi"),        # lofi keeps intensity without distraction
    "mysterious": ("csgo-surf",     "lofi-2"),      # lofi-2 is more atmospheric
    "funny":      ("cluster-truck", "chill-summer"),# light and playful
    "dramatic":   ("rocket-league", "lofi"),        # lofi under dramatic = tension
    "wholesome":  ("steep",         "chill-summer"),# warm and positive
    "scary":      ("minecraft-2",   "lofi-2"),      # lofi-2 is darker/moodier
}

# Maps sentiment → OpenAI voice name
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

# Maps sentiment → ElevenLabs voice name
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

# All valid sentiment labels
VALID_SENTIMENTS = list(BACKGROUND_MAP.keys())

# Fallback if detection fails — maps to rocket-league + lofi + alloy
DEFAULT_SENTIMENT = "dramatic"