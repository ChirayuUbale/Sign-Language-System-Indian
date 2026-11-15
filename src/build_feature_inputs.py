import argparse
import csv
import json
import random
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

import preprocess_pipeline as pp

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PREPROCESSED_DIR = ROOT / "Preprocessed_Data_Final"
OUTPUT_DIR = ROOT / "Feature_Inputs"
PREVIEW_COUNT = 10


def create_surf(hessian_threshold):
    try:
        return cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold, extended=False)
    except (AttributeError, cv2.error):
        raise SystemExit(
            "SURF is not available in this OpenCV build. "
            "Rebuild OpenCV with OPENCV_ENABLE_NONFREE=ON (see README)."
        )


def read_landmarks(path):
    landmarks = {}
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            signer, label, name = PureWindowsPath(row["image"]).parts[-3:]
            key = f"{signer}/{label}/{Path(name).stem}"
            landmarks[key] = (name, json.loads(row["landmarks_json"]))
    return landmarks


def letterbox_mask(cropped_shape, scale, offset_x, offset_y, margin=3):
    height, width = cropped_shape[:2]
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    mask = np.zeros((pp.IMAGE_SIZE, pp.IMAGE_SIZE), dtype=np.uint8)
    mask[offset_y + margin:offset_y + new_height - margin,
         offset_x + margin:offset_x + new_width - margin] = 255
    return mask


def surf_heatmap(keypoints, shape):
    height, width = shape
    heat = np.zeros(shape, dtype=np.float32)

    for kp in keypoints:
        x, y = kp.pt
        sigma = max(1.0, kp.size / 4)
        radius = int(3 * sigma) + 1
        x0, x1 = max(0, int(x) - radius), min(width, int(x) + radius + 1)
        y0, y1 = max(0, int(y) - radius), min(height, int(y) + radius + 1)
        xs = np.arange(x0, x1) - x
        ys = np.arange(y0, y1) - y
        heat[y0:y1, x0:x1] += kp.response * np.exp(
            -(ys[:, None] ** 2 + xs[None, :] ** 2) / (2 * sigma ** 2)
        )

    if heat.max() > 0:
        heat *= 255 / heat.max()
    return heat.astype(np.uint8)


def build_channels(original, preprocessed, hands, surf):
    cropped, crop_x, crop_y, _ = pp.calculate_safe_crop(original, hands)
    _, scale, offset_x, offset_y = pp.preprocess_main_image(cropped)

    skeleton = pp.draw_landmarks_on_image(
        np.zeros_like(preprocessed), hands, crop_x, crop_y, scale, offset_x, offset_y
    )
    mask = letterbox_mask(cropped.shape, scale, offset_x, offset_y)
    keypoints, descriptors = surf.detectAndCompute(preprocessed, mask)

    return surf_heatmap(keypoints, preprocessed.shape), skeleton, keypoints, descriptors


def save_keypoints(path, keypoints, descriptors):
    if descriptors is None:
        descriptors = np.zeros((0, 64), dtype=np.float32)
    np.savez_compressed(
        path,
        xy=np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2),
        size=np.array([kp.size for kp in keypoints], dtype=np.float32),
        angle=np.array([kp.angle for kp in keypoints], dtype=np.float32),
        response=np.array([kp.response for kp in keypoints], dtype=np.float32),
        descriptors=descriptors,
    )


def save_preview(samples, path):
    rows = []
    for key, preprocessed, keypoints, surf_map, skeleton in samples:
        with_keypoints = cv2.drawKeypoints(
            cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR), keypoints, None,
            (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        row = np.hstack([
            with_keypoints,
            cv2.applyColorMap(surf_map, cv2.COLORMAP_INFERNO),
            cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR),
        ])
        cv2.putText(row, f"{key}  keypoints={len(keypoints)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    parser = argparse.ArgumentParser(
        description="Build SURF heat-map and skeleton channels from Preprocessed_Data_Final."
    )
    parser.add_argument("--limit", type=int, help="process a random sample of N images")
    parser.add_argument("--hessian", type=float, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    surf = create_surf(args.hessian)
    landmarks = read_landmarks(PREPROCESSED_DIR / "mediapipe_landmarks.csv")

    keys = sorted(landmarks)
    if args.limit:
        keys = sorted(random.Random(args.seed).sample(keys, args.limit))
    preview_keys = set(keys[::max(1, len(keys) // PREVIEW_COUNT)][:PREVIEW_COUNT])

    OUTPUT_DIR.mkdir(exist_ok=True)
    log, preview = [], []

    for key in keys:
        name, hands = landmarks[key]
        signer, label, _ = key.split("/")
        original = cv2.imread(str(DATA_DIR / signer / label / name))
        preprocessed = cv2.imread(str(PREPROCESSED_DIR / f"{key}.png"), cv2.IMREAD_GRAYSCALE)
        if original is None or preprocessed is None:
            print(f"Skipping {key}: image not found")
            continue

        surf_map, skeleton, keypoints, descriptors = build_channels(
            original, preprocessed, hands, surf
        )

        for folder, image in (("surf_map", surf_map), ("skeleton", skeleton)):
            destination = OUTPUT_DIR / folder / f"{key}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), image)

        destination = OUTPUT_DIR / "surf_keypoints" / f"{key}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_keypoints(destination, keypoints, descriptors)

        log.append([key, len(hands), len(keypoints)])
        if key in preview_keys:
            preview.append((key, preprocessed, keypoints, surf_map, skeleton))

    with open(OUTPUT_DIR / "feature_log.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "detected_hands", "surf_keypoints"])
        writer.writerows(log)

    if preview:
        save_preview(preview, OUTPUT_DIR / "preview.png")

    print(f"Processed {len(log)} of {len(keys)} images. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
