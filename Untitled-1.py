"""
Stage 1 starter: train a tiny keyword classifier on your laptop.

Expected folder layout:

dataset/
├── START/
│   ├── start_001.wav
│   └── ...
├── STOP/
│   └── ...
├── OKAY/
│   └── ...
└── UNKNOWN/
    └── ...

Each .wav should be ~1 second, mono, any sample rate (we resample to 16kHz).

Usage:
    python train_keyword_classifier.py --data_dir ./dataset --epochs 20
"""

import argparse
import os
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

SAMPLE_RATE = 16000
DURATION_S = 1.0
N_MFCC = 13
CLASSES = ["UNKNOWN", "START", "STOP", "OKAY"]  # index order = label order


def extract_mfcc(path):
    """Load a wav file and return a fixed-size MFCC feature array."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    target_len = int(SAMPLE_RATE * DURATION_S)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return mfcc.astype(np.float32)  # shape: (N_MFCC, time_frames)


class KeywordDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []  # list of (filepath, label_idx)
        data_dir = Path(data_dir)
        for label_idx, class_name in enumerate(CLASSES):
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue
            for wav_path in class_dir.glob("*.wav"):
                self.samples.append((wav_path, label_idx))

        if not self.samples:
            raise RuntimeError(
                f"No .wav files found under {data_dir}. "
                f"Expected subfolders named: {CLASSES}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mfcc = extract_mfcc(path)
        x = torch.from_numpy(mfcc).unsqueeze(0)  # add channel dim -> (1, N_MFCC, T)
        return x, label


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
    dataset = KeywordDataset(data_dir)
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyKeywordCNN(n_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

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
    print(f"\nSaved model weights to {out_path}")
    print("Classes (index order):", CLASSES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="keyword_model.pt")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.out)