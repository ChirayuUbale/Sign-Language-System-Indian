"""Detect visible hands and crop them with padding, preserving dataset layout.

By default this reads data/ and writes a separate data_hand_crops/ tree.
Images where MediaPipe finds no hands are copied unchanged and counted.
"""
from __future__ import annotations

import argparse
import inspect
import math
import random
import re
import shutil
import textwrap
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def create_hands_detector(max_hands: int, min_confidence: float):
    """Initialize MediaPipe Hands, including a runtime shim for protobuf 7."""
    import google.protobuf.message_factory as message_factory

    # MediaPipe Solutions in the installed environment use APIs removed by
    # protobuf 7. Apply this compatibility shim only when needed.
    if not hasattr(message_factory.MessageFactory, "GetPrototype"):
        message_factory.MessageFactory.GetPrototype = (
            lambda self, descriptor: message_factory.GetMessageClass(descriptor)
        )
        import mediapipe.python.solution_base as solution_base

        source = textwrap.dedent(
            inspect.getsource(solution_base.SolutionBase._modify_calculator_options)
        )
        source = re.sub(
            r"field_label = calculator_options\.DESCRIPTOR\.fields_by_name\[\s*field_name\]\.label",
            "field_label = calculator_options.DESCRIPTOR.fields_by_name[\n                field_name].is_repeated",
            source,
        )
        source = source.replace(
            "if field_label == descriptor.FieldDescriptor.LABEL_REPEATED:",
            "if field_label:",
        )
        if ".label" in source:
            raise RuntimeError(
                "Installed MediaPipe is incompatible with protobuf. "
                "Install the protobuf version required by requirements.txt."
            )
        namespace = solution_base.__dict__
        exec(source, namespace)
        solution_base.SolutionBase._modify_calculator_options = namespace[
            "_modify_calculator_options"
        ]

    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=max_hands,
        min_detection_confidence=min_confidence,
    )


def padded_hand_box(landmarks, image_width: int, image_height: int, padding: float):
    """Return one padded box around every detected hand, clipped to the image."""
    xs, ys = [], []
    for hand in landmarks:
        for point in hand.landmark:
            xs.append(float(point.x) * image_width)
            ys.append(float(point.y) * image_height)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_width = max(1.0, x_max - x_min)
    box_height = max(1.0, y_max - y_min)
    pad_x = box_width * padding
    pad_y = box_height * padding

    left = max(0, math.floor(x_min - pad_x))
    top = max(0, math.floor(y_min - pad_y))
    right = min(image_width, math.ceil(x_max + pad_x))
    bottom = min(image_height, math.ceil(y_max + pad_y))
    return left, top, right, bottom


def fit_image(image, width: int, height: int):
    """Fit an image in a white tile without changing its aspect ratio."""
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(
        image,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    tile = np.full((height, width, 3), 255, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    tile[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return tile


def make_preview(records, preview_path: Path):
    """Create a side-by-side sheet of source boxes and saved crops."""
    if not records:
        return
    panel_w, panel_h, header_h, row_h = 480, 260, 62, 330
    sheet = np.full((len(records) * row_h, panel_w * 2, 3), 255, dtype=np.uint8)
    for index, record in enumerate(records):
        y = index * row_h
        source = record["original"].copy()
        crop = record["crop"].copy()
        left, top, right, bottom = record["box"]
        status = record["status"]
        if status == "cropped":
            cv2.rectangle(source, (left, top), (right - 1, bottom - 1), (0, 180, 0), 5)
        title = record["relative"].replace("\\", "/")
        cv2.putText(sheet, title[:68], (8, y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (0, 0, 0), 2, cv2.LINE_AA)
        left_label = "Original with crop box" if status == "cropped" else "No hand detected; original kept"
        label_color = (0, 100, 0) if status == "cropped" else (0, 0, 200)
        cv2.putText(sheet, left_label, (8, y + 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, label_color, 2, cv2.LINE_AA)
        cv2.putText(sheet, "Saved output", (panel_w + 8, y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        y0 = y + header_h + 4
        sheet[y0:y0 + panel_h, 0:panel_w] = fit_image(source, panel_w, panel_h)
        sheet[y0:y0 + panel_h, panel_w:panel_w * 2] = fit_image(crop, panel_w, panel_h)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview_path), sheet):
        raise OSError(f"Could not save preview image: {preview_path}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repo_root / "data", help="Input dataset root")
    parser.add_argument("--output", type=Path, default=repo_root / "data_hand_crops", help="New output folder")
    parser.add_argument("--padding", type=float, default=0.30,
                        help="Padding on each side as a fraction of the detected box (default: 0.30)")
    parser.add_argument("--min-confidence", type=float, default=0.4,
                        help="MediaPipe hand detection confidence (default: 0.4)")
    parser.add_argument("--max-hands", type=int, default=2,
                        help="Maximum detected hands per image (default: 2)")
    parser.add_argument("--preview-per-signer", type=int, default=3,
                        help="Random preview images per signer (default: 3)")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        parser.error(f"Input dataset folder does not exist: {root}")
    if output == root or root in output.parents:
        parser.error("Output folder must be separate from and outside the input dataset folder.")
    if args.padding < 0:
        parser.error("--padding must be non-negative")
    if not 0 < args.min_confidence <= 1:
        parser.error("--min-confidence must be in (0, 1]")
    if args.max_hands < 1:
        parser.error("--max-hands must be at least 1")
    if args.preview_per_signer < 0:
        parser.error("--preview-per-signer must be non-negative")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output folder is not empty; refusing to overwrite files: {output}")
    output.mkdir(parents=True, exist_ok=True)

    signer_dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold())
    if not signer_dirs:
        parser.error(f"No signer folders found under {root}")
    image_paths = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.as_posix().casefold(),
    )
    if not image_paths:
        parser.error(f"No supported images found under {root}")

    rng = random.Random(17)
    preview_paths = set()
    for signer_dir in signer_dirs:
        signer_images = [p for p in image_paths if signer_dir in p.parents]
        preview_paths.update(rng.sample(signer_images, min(args.preview_per_signer, len(signer_images))))

    detector = create_hands_detector(args.max_hands, args.min_confidence)
    counts = {"cropped": 0, "no_hand_detected": 0, "read_failed": 0, "multi_hand": 0}
    previews = []
    try:
        for index, source_path in enumerate(image_paths, start=1):
            relative = source_path.relative_to(root)
            target_path = output / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            original = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if original is None:
                shutil.copy2(source_path, target_path)
                counts["read_failed"] += 1
                continue

            height, width = original.shape[:2]
            result = detector.process(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            detected_hands = result.multi_hand_landmarks or []
            if not detected_hands:
                shutil.copy2(source_path, target_path)
                counts["no_hand_detected"] += 1
                saved = original
                box = (0, 0, width, height)
                status = "no_hand_detected"
            else:
                box = padded_hand_box(detected_hands, width, height, args.padding)
                left, top, right, bottom = box
                saved = original[top:bottom, left:right]
                if saved.size == 0:
                    shutil.copy2(source_path, target_path)
                    saved = original
                    status = "no_hand_detected"
                    counts["no_hand_detected"] += 1
                else:
                    if not cv2.imwrite(str(target_path), saved):
                        raise OSError(f"Could not save crop: {target_path}")
                    status = "cropped"
                    counts["cropped"] += 1
                    if len(detected_hands) > 1:
                        counts["multi_hand"] += 1

            if source_path in preview_paths:
                previews.append({"relative": str(relative), "original": original,
                                 "crop": saved, "box": box, "status": status})
            if index % 100 == 0 or index == len(image_paths):
                print(f"Processed {index}/{len(image_paths)} images", flush=True)
    finally:
        detector.close()

    preview_path = output / "previews" / "hand_crop_preview.png"
    make_preview(previews, preview_path)
    print("\nCrop summary:")
    print(f"  Total source images: {len(image_paths)}")
    print(f"  Cropped: {counts['cropped']}")
    print(f"  Images with multiple hands: {counts['multi_hand']}")
    print(f"  No hand detected; original copied: {counts['no_hand_detected']}")
    print(f"  Could not read; original copied: {counts['read_failed']}")
    print(f"  Output dataset: {output}")
    print(f"  Preview: {preview_path}")


if __name__ == "__main__":
    main()
