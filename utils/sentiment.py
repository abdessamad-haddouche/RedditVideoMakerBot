import json
import os
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


def _get_client() -> OpenAI:
    api_key = settings.config["deepseek"]["api_key"]
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


def _extract_text(reddit_object: dict) -> tuple:
    title = reddit_object.get("thread_title", "")
    post = reddit_object.get("thread_post", "")
    if isinstance(post, list):
        post = " ".join([p.get("text", "") for p in post if isinstance(p, dict)])
    return title, post


def detect_sentiment(reddit_object: dict) -> str:
    """
    Sends the post title + body to DeepSeek and returns a sentiment label.
    Falls back to DEFAULT_SENTIMENT on any error.
    """
    try:
        api_key = settings.config["deepseek"]["api_key"]
        if not api_key:
            print_substep("No DeepSeek API key found. Using default sentiment.", style="yellow")
            return DEFAULT_SENTIMENT

        title, post = _extract_text(reddit_object)
        text = f"Title: {title}\nPost: {post[:500]}"

        client = _get_client()

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment classifier for Reddit stories. "
                        "Classify the post into exactly one of these labels: "
                        "sad, happy, angry, mysterious, funny, dramatic, wholesome, scary. "
                        "Respond with only the label, nothing else. No punctuation, no explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            max_tokens=10,
            temperature=0,
        )

        label = response.choices[0].message.content.strip().lower()

        if label not in VALID_SENTIMENTS:
            print_substep(
                f"DeepSeek returned unexpected label '{label}'. Using default: {DEFAULT_SENTIMENT}",
                style="yellow",
            )
            return DEFAULT_SENTIMENT

        return label

    except Exception as e:
        print_substep(f"Sentiment detection failed: {e}. Using default: {DEFAULT_SENTIMENT}", style="yellow")
        return DEFAULT_SENTIMENT


def generate_metadata(reddit_object: dict, sentiment: str) -> dict:
    """
    Generates YouTube title, description, TikTok/Instagram/Facebook captions,
    and hashtags in a single DeepSeek call.
    Saves output as JSON next to the video in results/.
    Falls back to basic metadata on any error.
    """
    try:
        api_key = settings.config["deepseek"]["api_key"]
        if not api_key:
            return _fallback_metadata(reddit_object, sentiment)

        title, post = _extract_text(reddit_object)
        text = f"Title: {title}\nPost: {post[:800]}"
        channel_name = settings.config["settings"].get("channel_name", "Reddit Tales")

        client = _get_client()

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a social media content creator specializing in Reddit story videos. "
                        "Generate engaging titles, captions, and hashtags for a Reddit story video. "
                        "Return ONLY a valid JSON object with these exact keys: "
                        "youtube_title, youtube_description, tiktok_caption, instagram_caption, facebook_caption, hashtags. "
                        "hashtags must be a list of strings. "
                        "Keep youtube_title under 70 characters. "
                        "Keep tiktok_caption under 150 characters including hashtags. "
                        "Make content engaging and click-worthy. "
                        f"The channel name is '{channel_name}'. "
                        f"The story mood is: {sentiment}. "
                        "No markdown, no explanation, just the JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            max_tokens=600,
            temperature=0.7,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        metadata = json.loads(raw)

        # Validate all required keys exist
        required_keys = [
            "youtube_title", "youtube_description",
            "tiktok_caption", "instagram_caption",
            "facebook_caption", "hashtags"
        ]
        for key in required_keys:
            if key not in metadata:
                metadata[key] = ""

        metadata["sentiment"] = sentiment
        return metadata

    except Exception as e:
        print_substep(f"Metadata generation failed: {e}. Using fallback.", style="yellow")
        return _fallback_metadata(reddit_object, sentiment)


def _fallback_metadata(reddit_object: dict, sentiment: str) -> dict:
    """Basic fallback metadata if DeepSeek fails."""
    title = reddit_object.get("thread_title", "Reddit Story")
    return {
        "sentiment": sentiment,
        "youtube_title": title[:70],
        "youtube_description": f"{title}\n\n#reddit #stories",
        "tiktok_caption": f"{title[:100]} #reddit #storytime",
        "instagram_caption": f"{title[:100]} #reddit #stories",
        "facebook_caption": title,
        "hashtags": ["#reddit", "#storytime", "#stories", "#redditstories"],
    }


def save_metadata(metadata: dict, reddit_object: dict) -> None:
    """Saves metadata JSON inside the per-video folder."""
    try:
        subreddit = reddit_object.get("thread_subreddit", settings.config["reddit"]["thread"]["subreddit"])
        thread_id = reddit_object.get("thread_id", "unknown")
        sentiment_bg = settings.config["settings"]["background"].get("background_video", "unknown")
        video_folder = f"results/{subreddit}/{thread_id}_{sentiment_bg}"
        os.makedirs(video_folder, exist_ok=True)
        filepath = f"{video_folder}/metadata.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print_substep(f"Metadata saved → {filepath}", style="bold green")
    except Exception as e:
        print_substep(f"Failed to save metadata: {e}", style="yellow")


def apply_sentiment_config(reddit_object: dict) -> None:
    """
    Detects sentiment, overrides in-memory config for background/voice,
    generates metadata, and saves it to disk.
    Does NOT write to config.toml — changes are per-run only.
    """
    print_step("Detecting sentiment and generating metadata... 🎭")

    sentiment = detect_sentiment(reddit_object)

    # ── Background ───────────────────────────────────────────
    bg_video, bg_audio = BACKGROUND_MAP[sentiment]
    settings.config["settings"]["background"]["background_video"] = bg_video
    settings.config["settings"]["background"]["background_audio"] = bg_audio

    # ── Voice ────────────────────────────────────────────────
    voice_choice = settings.config["settings"]["tts"]["voice_choice"].lower()

    if voice_choice == "elevenlabs":
        voice = ELEVENLABS_VOICE_MAP[sentiment]
        settings.config["settings"]["tts"]["elevenlabs_voice_name"] = voice
    elif voice_choice == "openai":
        voice = OPENAI_VOICE_MAP[sentiment]
        settings.config["settings"]["tts"]["openai_voice_name"] = voice
    else:
        voice = f"(voice override not supported for {voice_choice})"

    # ── Metadata ─────────────────────────────────────────────
    print_substep("Generating titles, captions and hashtags... ✍️", style="bold blue")
    metadata = generate_metadata(reddit_object, sentiment)
    save_metadata(metadata, reddit_object)

    # ── Log ──────────────────────────────────────────────────
    print_substep(f"Sentiment detected  : {sentiment} 🎯", style="bold green")
    print_substep(f"Background video    : {bg_video}", style="bold blue")
    print_substep(f"Background audio    : {bg_audio if bg_audio else 'none'}", style="bold blue")
    print_substep(f"Voice               : {voice}", style="bold blue")
    print_substep(f"YouTube title       : {metadata['youtube_title']}", style="bold blue")
    print_substep(f"TikTok caption      : {metadata['tiktok_caption']}", style="bold blue")