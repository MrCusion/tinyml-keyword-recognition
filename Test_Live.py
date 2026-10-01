"""
Live-test your trained keyword model using the microphone.

Records 1-second clips back-to-back and prints the model's confidence
for every class after each clip, so you can see in real time whether it
recognizes your keywords.

Requires: pip install sounddevice

Usage:
    python test_live.py
    python test_live.py --model keyword_model.pt
"""

import argparse
import json

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F

# Reuse the exact same constants, feature extraction, and model definition
# used during training, so live inference matches training exactly.
import train_keyword_classifier as tkc


def load_classes(model_path):
    classes_path = model_path.rsplit(".", 1)[0] + ".classes.json"
    with open(classes_path) as f:
        return json.load(f)


def load_model(model_path, n_classes):
    model = tkc.TinyKeywordCNN(n_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def record_clip():
    """Record one DURATION_S-long clip from the default microphone."""
    audio = sd.rec(
        int(tkc.SAMPLE_RATE * tkc.DURATION_S),
        samplerate=tkc.SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def predict(model, classes, audio_array):
    mfcc = tkc.extract_mfcc_from_array(audio_array)
    x = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0)  # add batch + channel dims
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="keyword_model.pt")
    args = parser.parse_args()

    classes = load_classes(args.model)
    model = load_model(args.model, len(classes))

    print("Classes:", classes)
    print("Listening — say a keyword after each prompt. Press Ctrl+C to stop.\n")

    try:
        while True:
            input("Press Enter, then speak...")
            audio = record_clip()
            probs = predict(model, classes, audio)

            top_idx = int(np.argmax(probs))
            line = "  ".join(f"{c}: {p:.1%}" for c, p in zip(classes, probs))
            print(f"  -> predicted: {classes[top_idx]}  ({probs[top_idx]:.1%})")
            print(f"     all scores: {line}\n")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()