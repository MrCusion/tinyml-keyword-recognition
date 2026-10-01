"""
Organize the downloaded Google Speech Commands dataset into your dataset/ folders.

Run this AFTER torchaudio.datasets.SPEECHCOMMANDS(..., download=True) has finished.

Usage:
    python organize_dataset.py
"""

import os
import shutil
import random

from tqdm import tqdm

SOURCE_DIR = "./data/SpeechCommands/speech_commands_v0.02"
DEST_DIR = "dataset"

# How many UNKNOWN clips to take per leftover word (keeps UNKNOWN roughly balanced)
CLIPS_PER_UNKNOWN_WORD = 40

random.seed(42)  # reproducible selection


def choose_keywords():
    """List the words actually available in the dataset and let the user pick which ones to keep."""
    available = sorted(
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d)) and not d.startswith("_")
    )

    print("Available words in the dataset:")
    for i, word in enumerate(available, 1):
        print(f"  {i:2d}. {word}")

    raw = input(
        "\nType the words you want as keyword classes (comma-separated, e.g. 'stop,go,yes'): "
    )
    chosen = [w.strip().lower() for w in raw.split(",") if w.strip()]
    chosen = [w for w in chosen if w in available]

    if not chosen:
        print("No valid words selected — defaulting to 'stop'.")
        chosen = ["stop"]

    leftover = [w for w in available if w not in chosen]
    return chosen, leftover


def copy_all(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    files = [f for f in os.listdir(src_folder) if f.endswith(".wav")]
    for f in tqdm(files, desc=f"Copying {os.path.basename(dst_folder)}", unit="file"):
        shutil.copy(os.path.join(src_folder, f), os.path.join(dst_folder, f))
    return len(files)


def copy_sample(src_folder, dst_folder, n, prefix):
    os.makedirs(dst_folder, exist_ok=True)
    files = [f for f in os.listdir(src_folder) if f.endswith(".wav")]
    chosen = random.sample(files, min(n, len(files)))
    for f in tqdm(chosen, desc=f"Sampling '{prefix}' -> UNKNOWN", unit="file"):
        # prefix avoids filename collisions across different source words
        new_name = f"{prefix}_{f}"
        shutil.copy(os.path.join(src_folder, f), os.path.join(dst_folder, new_name))
    return len(chosen)


def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"Can't find {SOURCE_DIR} — has the download finished?")
        return

    chosen_words, leftover_words = choose_keywords()

    # One class folder per chosen keyword, named after the word (e.g. dataset/STOP)
    for word in chosen_words:
        word_src = os.path.join(SOURCE_DIR, word)
        word_dst = os.path.join(DEST_DIR, word.upper())
        n = copy_all(word_src, word_dst)
        print(f"{word.upper()}: copied {n} files -> {word_dst}")

    # UNKNOWN class, sampled from every word that wasn't chosen
    unknown_dst = os.path.join(DEST_DIR, "UNKNOWN")
    total_unknown = 0
    for word in leftover_words:
        word_src = os.path.join(SOURCE_DIR, word)
        n = copy_sample(word_src, unknown_dst, CLIPS_PER_UNKNOWN_WORD, prefix=word)
        total_unknown += n
    print(f"UNKNOWN: copied {total_unknown} files -> {unknown_dst}")

    print("\nDone. Folder counts:")
    for cls in [w.upper() for w in chosen_words] + ["UNKNOWN"]:
        d = os.path.join(DEST_DIR, cls)
        count = len([f for f in os.listdir(d) if f.endswith(".wav")]) if os.path.isdir(d) else 0
        print(f"  {cls}: {count} files")


if __name__ == "__main__":
    main()