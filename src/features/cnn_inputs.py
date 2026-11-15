"""Stack the preprocessed image, SURF map and skeleton as CNN input for the label-noise splits."""

import argparse
import csv
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREPROCESSED_DIR = ROOT / "Preprocessed_Data_Final"
FEATURE_DIR = ROOT / "Feature_Inputs"
SPLIT_DIR = ROOT / "Label_Noise_Experiments"
SPLITS = ("train_noise_0", "train_noise_10", "train_noise_20", "test_clean")


def channel_paths(signer, label, stem):
    name = Path(signer, label, f"{stem}.png")
    return [
        PREPROCESSED_DIR / name,
        FEATURE_DIR / "surf_map" / name,
        FEATURE_DIR / "skeleton" / name,
    ]


def load_split(csv_path, size=100):
    """Return X (N, size, size, 3), the assigned labels, and per-image info."""
    images, labels = [], []
    info = {"signer": [], "stem": [], "original_label": [], "is_noisy": []}

    with open(csv_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            stem = Path(PureWindowsPath(row["image_path"]).name).stem.split("__")[-1]

            channels = []
            for path in channel_paths(row["signer"], row["original_label"], stem):
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(path)
                channels.append(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA))

            images.append(np.stack(channels, axis=-1))
            labels.append(row["assigned_label"])
            info["signer"].append(row["signer"])
            info["stem"].append(stem)
            info["original_label"].append(row["original_label"])
            info["is_noisy"].append(row["is_noisy"] == "True")

    return np.array(images), np.array(labels), {k: np.array(v) for k, v in info.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true",
                        help="write Feature_Inputs/stacked_100/<split>.npz for every split")
    args = parser.parse_args()

    if not args.export:
        parser.print_help()
        return

    output = FEATURE_DIR / "stacked_100"
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        X, y, info = load_split(SPLIT_DIR / f"{split}.csv")
        np.savez_compressed(output / f"{split}.npz", X=X, y=y, **info)
        print(f"{split}: X={X.shape}, noisy labels={int(info['is_noisy'].sum())}")


if __name__ == "__main__":
    main()
