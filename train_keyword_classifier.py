"""
Stage 1 starter: train a tiny keyword classifier on your laptop.

Classes are auto-detected from whatever subfolders exist in --data_dir,
e.g. if organize_dataset.py made:

dataset/
├── STOP/
│   ├── stop_001.wav
│   └── ...
├── GO/
│   └── ...
└── UNKNOWN/
    └── ...

...those become the classes automatically (UNKNOWN is always sorted last).

Each .wav should be ~1 second, mono, any sample rate (we resample to 16kHz).

Usage:
    python train_keyword_classifier.py --data_dir ./dataset --epochs 20
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

SAMPLE_RATE = 16000
DURATION_S = 1.0
N_MFCC = 13


def detect_classes(data_dir):
    """Find class names from whatever subfolders exist in data_dir, UNKNOWN sorted last."""
    data_dir = Path(data_dir)
    found = sorted(
        d.name for d in data_dir.iterdir() if d.is_dir()
    )
    if "UNKNOWN" in found:
        found.remove("UNKNOWN")
        found.append("UNKNOWN")
    return found


def extract_mfcc_from_array(y):
    """Turn a raw audio array (any length) into the fixed-size MFCC feature grid.
    This is the single source of truth for feature extraction — used for both
    training (from files) and live inference (from the microphone), so they
    can never drift apart."""
    target_len = int(SAMPLE_RATE * DURATION_S)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return mfcc.astype(np.float32)  # shape: (N_MFCC, time_frames)


def extract_mfcc(path):
    """Load a wav file and return a fixed-size MFCC feature array."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return extract_mfcc_from_array(y)


class KeywordDataset(Dataset):
    def __init__(self, data_dir, classes):
        paths_and_labels = []  # list of (filepath, label_idx)
        data_dir = Path(data_dir)
        for label_idx, class_name in enumerate(classes):
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue
            for wav_path in class_dir.glob("*.wav"):
                paths_and_labels.append((wav_path, label_idx))

        if not paths_and_labels:
            raise RuntimeError(
                f"No .wav files found under {data_dir}. "
                f"Expected subfolders named: {classes}"
            )

        # Compute every file's MFCC features once, up front, instead of
        # recomputing them from scratch on every epoch — this is normally
        # the biggest speed win available, since MFCC extraction (not the
        # tiny model) is what actually takes the time.
        self.samples = []  # list of (mfcc_tensor, label_idx), ready to use
        for path, label in tqdm(paths_and_labels, desc="Precomputing features", unit="file"):
            mfcc = extract_mfcc(path)
            x = torch.from_numpy(mfcc).unsqueeze(0)  # add channel dim -> (1, N_MFCC, T)
            self.samples.append((x, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TinyKeywordCNN(nn.Module):
    """Deliberately small: this is the shape that will later get quantized for the ESP32."""

    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)


def train(data_dir, epochs, batch_size, lr, out_path):
    classes = detect_classes(data_dir)
    print("Detected classes (index order):", classes)

    dataset = KeywordDataset(data_dir, classes)
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyKeywordCNN(n_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for x, y in progress:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / n_train

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
        val_acc = correct / n_val

        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_acc {val_acc:.2%}")

    torch.save(model.state_dict(), out_path)
    classes_path = os.path.splitext(out_path)[0] + ".classes.json"
    with open(classes_path, "w") as f:
        json.dump(classes, f)
    print(f"\nSaved model weights to {out_path}")
    print(f"Saved class order to {classes_path}")
    print("Classes (index order):", classes)


def prompt_with_default(message, default, cast=str):
    """Ask for a value, press Enter to keep the default."""
    raw = input(f"{message} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"  Couldn't parse that, using default: {default}")
        return default


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="keyword_model.pt")
    args = parser.parse_args()

    # Only epochs is asked interactively; everything else stays at its default
    # unless overridden with a --flag.
    if args.epochs is None:
        args.epochs = prompt_with_default("Epochs", 20, int)

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.out)