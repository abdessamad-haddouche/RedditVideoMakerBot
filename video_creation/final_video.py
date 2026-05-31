import multiprocessing
import os
import re
import tempfile
import textwrap
import threading
import time
from os.path import exists
from pathlib import Path
from typing import Dict, Final, Tuple
import glob
import json

import ffmpeg
import translators
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.progress import track

from utils import settings
from utils.cleanup import cleanup
from utils.console import print_step, print_substep
from utils.ffmpeg_encoder import get_video_codec_args, has_drawtext
from utils.fonts import getheight
from utils.id import extract_id
from utils.thumbnail import create_thumbnail
from utils.videos import save_data

console = Console()


class ProgressFfmpeg(threading.Thread):
    def __init__(self, vid_duration_seconds, progress_update_callback):
        threading.Thread.__init__(self, name="ProgressFfmpeg")
        self.stop_event = threading.Event()
        self.output_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        self.vid_duration_seconds = vid_duration_seconds
        self.progress_update_callback = progress_update_callback

    def run(self):
        while not self.stop_event.is_set():
            latest_progress = self.get_latest_ms_progress()
            if latest_progress is not None:
                completed_percent = latest_progress / self.vid_duration_seconds
                self.progress_update_callback(completed_percent)
            time.sleep(1)

    def get_latest_ms_progress(self):
        lines = self.output_file.readlines()
        if lines:
            for line in lines:
                if "out_time_ms" in line:
                    out_time_ms_str = line.split("=")[1].strip()
                    if out_time_ms_str.isnumeric():
                        return float(out_time_ms_str) / 1000000.0
                    else:
                        return None
        return None

    def stop(self):
        self.stop_event.set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()


def name_normalize(name: str) -> str:
    name = re.sub(r'[?\\"%*:|<>]', "", name)
    name = re.sub(r"( [w,W]\s?\/\s?[o,O,0])", r" without", name)
    name = re.sub(r"( [w,W]\s?\/)", r" with", name)
    name = re.sub(r"(\d+)\s?\/\s?(\d+)", r"\1 of \2", name)
    name = re.sub(r"(\w+)\s?\/\s?(\w+)", r"\1 or \2", name)
    name = re.sub(r"\/", r"", name)
    lang = settings.config["reddit"]["thread"]["post_lang"]
    if lang:
        print_substep("Translating filename...")
        translated_name = translators.translate_text(name, translator="google", to_language=lang)
        return translated_name
    else:
        return name


def prepare_background(reddit_id: str, W: int, H: int) -> str:
    output_path = f"assets/temp/{reddit_id}/background_noaudio.mp4"
    output = (
        ffmpeg.input(f"assets/temp/{reddit_id}/background.mp4")
        .filter("crop", f"ih*({W}/{H})", "ih")
        .output(
            output_path,
            an=None,
            **get_video_codec_args(),
        )
        .overwrite_output()
    )
    try:
        output.run(quiet=True)
    except ffmpeg.Error as e:
        print(e.stderr.decode("utf8"))
        exit(1)
    return output_path


def get_text_height(draw, text, font, max_width):
    lines = textwrap.wrap(text, width=max_width)
    total_height = 0
    for line in lines:
        _, _, _, height = draw.textbbox((0, 0), line, font=font)
        total_height += height
    return total_height


def create_fancy_thumbnail(image, text, text_color, padding, wrap=35):
    print_step(f"Creating fancy thumbnail for: {text}")
    font_title_size = 47
    font = ImageFont.truetype(os.path.join("fonts", "Roboto-Bold.ttf"), font_title_size)
    image_width, image_height = image.size

    draw = ImageDraw.Draw(image)
    text_height = get_text_height(draw, text, font, wrap)
    lines = textwrap.wrap(text, width=wrap)
    new_image_height = image_height + text_height + padding * (len(lines) - 1) - 50

    top_part_height    = image_height // 2
    middle_part_height = 1
    bottom_part_height = image_height - top_part_height - middle_part_height

    top_part    = image.crop((0, 0, image_width, top_part_height))
    middle_part = image.crop((0, top_part_height, image_width, top_part_height + middle_part_height))
    bottom_part = image.crop((0, top_part_height + middle_part_height, image_width, image_height))

    new_middle_height = max(1, new_image_height - top_part_height - bottom_part_height)
    middle_part = middle_part.resize((image_width, new_middle_height))

    new_image = Image.new("RGBA", (image_width, new_image_height))
    new_image.paste(top_part,    (0, 0))
    new_image.paste(middle_part, (0, top_part_height))
    new_image.paste(bottom_part, (0, top_part_height + new_middle_height))

    draw = ImageDraw.Draw(new_image)
    y = top_part_height + padding
    for line in lines:
        draw.text((120, y), line, font=font, fill=text_color, align="left")
        y += get_text_height(draw, line, font, wrap) + padding

    username_font = ImageFont.truetype(os.path.join("fonts", "Roboto-Bold.ttf"), 30)
    draw.text(
        (205, 825),
        settings.config["settings"]["channel_name"],
        font=username_font,
        fill=text_color,
        align="left",
    )
    return new_image


def merge_background_audio(audio: ffmpeg, reddit_id: str):
    background_audio_volume = settings.config["settings"]["background"]["background_audio_volume"]
    if background_audio_volume == 0:
        return audio
    bg_audio = ffmpeg.input(f"assets/temp/{reddit_id}/background.mp3").filter(
        "volume", background_audio_volume,
    )
    return ffmpeg.filter([audio, bg_audio], "amix", duration="longest")


def _load_timing_map(reddit_id: str, img_files: list, postaudio_files: list,
                     audio_clips_durations: list, title_duration: float) -> list:
    """
    Load timing_map.json written by imagemaker().

    Each entry is one of:
      {"timing_type": "absolute", "clip_start": S, "clip_end": E}
        → used directly as FFmpeg enable times

      {"timing_type": "fraction", "audio_idx": N, "time_fraction": F}
        → clip time computed as: audio_start[N] + accumulated_fraction * audio_dur[N]

    Falls back to 1:1 mapping if file missing.
    """
    timing_map_path = f"assets/temp/{reddit_id}/timing_map.json"
    if os.path.exists(timing_map_path):
        with open(timing_map_path) as f:
            return json.load(f)

    # Fallback: 1:1
    print_substep("timing_map.json not found — using 1:1 fallback", style="yellow")
    return [
        {"timing_type": "fraction", "audio_idx": i, "time_fraction": 1.0}
        for i in range(len(img_files))
    ]


def make_final_video(
    number_of_clips: int,
    length: int,
    reddit_obj: dict,
    background_config: Dict[str, Tuple],
):
    W: Final[int] = int(settings.config["settings"]["resolution_w"])
    H: Final[int] = int(settings.config["settings"]["resolution_h"])
    opacity       = settings.config["settings"]["opacity"]
    reddit_id     = extract_id(reddit_obj)

    allowOnlyTTSFolder: bool = (
        settings.config["settings"]["background"]["enable_extra_audio"]
        and settings.config["settings"]["background"]["background_audio_volume"] != 0
    )

    print_step("Creating the final video 🎥")
    background_clip = ffmpeg.input(prepare_background(reddit_id, W=W, H=H))

    # ── Audio clips ───────────────────────────────────────────────────────────
    audio_clips = list()
    if number_of_clips == 0 and settings.config["settings"]["storymode"] == "false":
        print("No audio clips to gather.")
        exit()

    if settings.config["settings"]["storymode"]:
        if settings.config["settings"]["storymodemethod"] == 0:
            audio_clips = [ffmpeg.input(f"assets/temp/{reddit_id}/mp3/title.mp3")]
            audio_clips.insert(1, ffmpeg.input(f"assets/temp/{reddit_id}/mp3/postaudio.mp3"))
        elif settings.config["settings"]["storymodemethod"] == 1:
            postaudio_files = sorted(
                glob.glob(f"assets/temp/{reddit_id}/mp3/postaudio-*.mp3"),
                key=lambda x: int(re.search(r"postaudio-(\d+)", x).group(1))
            )
            audio_clips = [ffmpeg.input(f) for f in postaudio_files]
            audio_clips.insert(0, ffmpeg.input(f"assets/temp/{reddit_id}/mp3/title.mp3"))
    else:
        audio_clips = [
            ffmpeg.input(f"assets/temp/{reddit_id}/mp3/{i}.mp3")
            for i in range(number_of_clips)
        ]
        audio_clips.insert(0, ffmpeg.input(f"assets/temp/{reddit_id}/mp3/title.mp3"))
        audio_clips_durations = [
            float(ffmpeg.probe(f"assets/temp/{reddit_id}/mp3/{i}.mp3")["format"]["duration"])
            for i in range(number_of_clips)
        ]
        audio_clips_durations.insert(
            0,
            float(ffmpeg.probe(f"assets/temp/{reddit_id}/mp3/title.mp3")["format"]["duration"]),
        )

    audio_concat = ffmpeg.concat(*audio_clips, a=1, v=0)
    ffmpeg.output(
        audio_concat, f"assets/temp/{reddit_id}/audio.mp3", **{"b:a": "192k"}
    ).overwrite_output().run(quiet=False)

    console.log(f"[bold green] Video Will Be: {length} Seconds Long")

    screenshot_width = int((W * 45) // 100)
    audio       = ffmpeg.input(f"assets/temp/{reddit_id}/audio.mp3")
    final_audio = merge_background_audio(audio, reddit_id)
    image_clips = list()

    Path(f"assets/temp/{reddit_id}/png").mkdir(parents=True, exist_ok=True)

    # ── Title card ────────────────────────────────────────────────────────────
    title_template = Image.open("assets/title_template.png")
    title          = name_normalize(reddit_obj["thread_title"])
    title_img      = create_fancy_thumbnail(title_template, title, "#000000", 5)
    title_img.save(f"assets/temp/{reddit_id}/png/title.png")
    image_clips.insert(
        0,
        ffmpeg.input(f"assets/temp/{reddit_id}/png/title.png")["v"].filter(
            "scale", screenshot_width, -1
        ),
    )

    current_time = 0

    if settings.config["settings"]["storymode"]:

        if settings.config["settings"]["storymodemethod"] == 0:
            audio_clips_durations = [
                float(ffmpeg.probe(f"assets/temp/{reddit_id}/mp3/postaudio.mp3")["format"]["duration"])
            ]
            audio_clips_durations.insert(
                0,
                float(ffmpeg.probe(f"assets/temp/{reddit_id}/mp3/title.mp3")["format"]["duration"]),
            )
            image_clips.insert(
                1,
                ffmpeg.input(f"assets/temp/{reddit_id}/png/story_content.png").filter(
                    "scale", screenshot_width, -1
                ),
            )
            background_clip = background_clip.overlay(
                image_clips[0],
                enable=f"between(t,{current_time},{current_time + audio_clips_durations[0]})",
                x="(main_w-overlay_w)/2",
                y="(main_h-overlay_h)/2",
            )
            current_time += audio_clips_durations[0]

        elif settings.config["settings"]["storymodemethod"] == 1:

            # ── Discover postaudio files ──────────────────────────────────────
            postaudio_files = sorted(
                glob.glob(f"assets/temp/{reddit_id}/mp3/postaudio-*.mp3"),
                key=lambda x: int(re.search(r"postaudio-(\d+)", x).group(1))
            )

            # ── Build durations ───────────────────────────────────────────────
            audio_clips_durations = [
                float(ffmpeg.probe(f)["format"]["duration"])
                for f in postaudio_files
            ]
            title_duration = float(
                ffmpeg.probe(f"assets/temp/{reddit_id}/mp3/title.mp3")["format"]["duration"]
            )
            audio_clips_durations.insert(0, title_duration)

            # ── Pre-compute absolute start time per audio file ────────────────
            audio_start_times = []
            t = title_duration
            for dur in audio_clips_durations[1:]:
                audio_start_times.append(t)
                t += dur

            # ── Title card overlay ────────────────────────────────────────────
            background_clip = background_clip.overlay(
                image_clips[0],
                enable=f"between(t,0,{title_duration})",
                x="(main_w-overlay_w)/2",
                y="(main_h-overlay_h)/2",
            )
            current_time = title_duration

            # ── Load image files ──────────────────────────────────────────────
            img_files = sorted(
                glob.glob(f"assets/temp/{reddit_id}/png/img*.png"),
                key=lambda x: int(re.search(r"img(\d+)", x).group(1))
            )

            # ── Load timing map ───────────────────────────────────────────────
            timing_map = _load_timing_map(
                reddit_id, img_files, postaudio_files,
                audio_clips_durations, title_duration
            )

            # ── Overlay each image ────────────────────────────────────────────
            audio_time_used = {}

            for i, img_file in enumerate(img_files):
                if i >= len(timing_map):
                    break

                entry       = timing_map[i]
                timing_type = entry.get("timing_type", "fraction")

                if timing_type == "absolute":
                    clip_start = entry["clip_start"]
                    clip_end   = entry["clip_end"]
                else:
                    audio_idx     = entry["audio_idx"]
                    time_fraction = entry["time_fraction"]
                    if audio_idx + 1 >= len(audio_clips_durations):
                        break
                    audio_dur     = audio_clips_durations[audio_idx + 1]
                    display_dur   = audio_dur * time_fraction
                    offset        = audio_time_used.get(audio_idx, 0.0)
                    clip_start    = audio_start_times[audio_idx] + offset
                    clip_end      = clip_start + display_dur
                    audio_time_used[audio_idx] = offset + display_dur

                img_clip = ffmpeg.input(img_file)["v"].filter(
                    "scale", screenshot_width, -1
                )
                image_clips.append(img_clip)
                background_clip = background_clip.overlay(
                    img_clip,
                    enable=f"between(t,{clip_start:.3f},{clip_end:.3f})",
                    x="(main_w-overlay_w)/2",
                    y="(main_h-overlay_h)/2",
                )

            current_time = t

    else:
        for i in range(0, number_of_clips + 1):
            image_clips.append(
                ffmpeg.input(f"assets/temp/{reddit_id}/png/comment_{i}.png")["v"].filter(
                    "scale", screenshot_width, -1
                )
            )
            image_overlay = image_clips[i].filter("colorchannelmixer", aa=opacity)
            assert audio_clips_durations is not None
            background_clip = background_clip.overlay(
                image_overlay,
                enable=f"between(t,{current_time},{current_time + audio_clips_durations[i]})",
                x="(main_w-overlay_w)/2",
                y="(main_h-overlay_h)/2",
            )
            current_time += audio_clips_durations[i]

    # ── Output ────────────────────────────────────────────────────────────────
    title_str    = extract_id(reddit_obj, "thread_title")
    idx          = extract_id(reddit_obj)
    title_thumb  = reddit_obj["thread_title"]
    subreddit    = reddit_obj.get("thread_subreddit", settings.config["reddit"]["thread"]["subreddit"])
    sentiment    = settings.config["settings"]["background"].get("background_video", "unknown")
    video_folder = f"./results/{subreddit}/{idx}_{sentiment}"
    os.makedirs(video_folder, exist_ok=True)

    if allowOnlyTTSFolder:
        os.makedirs(f"{video_folder}/OnlyTTS", exist_ok=True)

    settingsbackground = settings.config["settings"]["background"]
    if settingsbackground["background_thumbnail"]:
        first_image = next(
            (f for f in os.listdir("assets/backgrounds") if f.endswith(".png")), None
        )
        if first_image is None:
            print_substep("No png files found in assets/backgrounds", "red")
        else:
            thumbnail = Image.open(f"assets/backgrounds/{first_image}")
            w, h = thumbnail.size
            thumbnailSave = create_thumbnail(
                thumbnail,
                settingsbackground["background_thumbnail_font_family"],
                settingsbackground["background_thumbnail_font_size"],
                settingsbackground["background_thumbnail_font_color"],
                w, h, title_thumb,
            )
            thumbnailSave.save(f"{video_folder}/thumbnail.png")

    # ── drawtext watermark — skipped gracefully if filter unavailable ─────────
    if has_drawtext():
        background_clip = ffmpeg.drawtext(
            background_clip,
            text=f"Background by {background_config['video'][2]}",
            x="(w-text_w)", y="(h-text_h)",
            fontsize=5, fontcolor="White",
            fontfile=os.path.join("fonts", "Roboto-Regular.ttf"),
        )
    background_clip = background_clip.filter("scale", W, H)

    print_step("Rendering the video 🎥")
    from tqdm import tqdm
    pbar = tqdm(total=100, desc="Progress: ", bar_format="{l_bar}{bar}", unit=" %")

    def on_update_example(progress) -> None:
        status = round(progress * 100, 2)
        old_percentage = pbar.n
        pbar.update(status - old_percentage)

    with ProgressFfmpeg(length, on_update_example) as progress:
        path = f"{video_folder}/video.mp4"
        try:
            ffmpeg.output(
                background_clip, final_audio, path,
                f="mp4",
                **get_video_codec_args(),
            ).overwrite_output().global_args("-progress", progress.output_file.name).run(
                quiet=True, overwrite_output=True,
                capture_stdout=False, capture_stderr=False,
            )
        except ffmpeg.Error as e:
            print(e.stderr.decode("utf8"))
            exit(1)

    old_percentage = pbar.n
    pbar.update(100 - old_percentage)

    if allowOnlyTTSFolder:
        path = f"{video_folder}/OnlyTTS/video.mp4"
        print_step("Rendering the Only TTS Video 🎥")
        with ProgressFfmpeg(length, on_update_example) as progress:
            try:
                ffmpeg.output(
                    background_clip, audio, path,
                    f="mp4",
                    **get_video_codec_args(),
                ).overwrite_output().global_args("-progress", progress.output_file.name).run(
                    quiet=True, overwrite_output=True,
                    capture_stdout=False, capture_stderr=False,
                )
            except ffmpeg.Error as e:
                print(e.stderr.decode("utf8"))
                exit(1)
        old_percentage = pbar.n
        pbar.update(100 - old_percentage)

    pbar.close()
    save_data(subreddit, f"{idx}_{sentiment}/video.mp4", title_str, idx, background_config["video"][2])
    print_step("Removing temporary files 🗑")
    cleanups = cleanup(reddit_id)
    print_substep(f"Removed {cleanups} temporary files 🗑")
    print_step("Done! 🎉 The video is in the results folder 📁")