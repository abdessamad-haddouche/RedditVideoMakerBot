import os
import shutil
from os.path import exists


def _listdir(d):  # listdir with full path
    return [os.path.join(d, f) for f in os.listdir(d)]


def cleanup(reddit_id: str) -> int:
    """
    Deletes temporary assets for a specific reddit_id after video generation.

    Returns:
        int: 1 if the directory was deleted, 0 if it didn't exist.
    """
    directory = f"assets/temp/{reddit_id}/"
    if exists(directory):
        shutil.rmtree(directory)
        return 1
    return 0


def cleanup_temp() -> int:
    """
    Wipes the ENTIRE assets/temp/ folder before a new run starts.
    Prevents stale MP3/PNG files from previous runs (different TTS engine,
    different sentence count, etc.) from bleeding into the new video.

    Called at the START of each run in main.py — not at the end.

    Returns:
        int: Number of reddit_id folders deleted.
    """
    temp_dir = "assets/temp/"
    if not exists(temp_dir):
        return 0

    count = 0
    for entry in os.listdir(temp_dir):
        full_path = os.path.join(temp_dir, entry)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
            count += 1

    return count