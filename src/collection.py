"""Collect webcam images in data/<signer>/<label>/<label>_001.png.

Restored from the original collection script, with resumable numbered filenames.
Default settings: A-Z and 0-9, 20 images per sign, 15 seconds to prepare,
and 2 seconds between captures. Keep both hands inside the green rectangle.
Press S to capture immediately, or Q / Escape to quit.
"""

from __future__ import annotations

import argparse
import math
import string
import time
from pathlib import Path

import cv2


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LABELS = tuple(string.ascii_uppercase + string.digits)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def next_image_path(folder: Path, label: str) -> Path:
    """Continue after the largest existing index, including JPEG filenames."""
    indices = []
    for path in image_files(folder):
        prefix, separator, number = path.stem.rpartition("_")
        if prefix == label and separator and number.isdecimal():
            indices.append(int(number))
    return folder / f"{label}_{max(indices, default=0) + 1:03d}.png"


def collect(
    root: Path,
    signer: str,
    labels: list[str],
    goal: int,
    camera: int,
    interval: float,
    ready_seconds: float,
) -> None:
    pending = [label for label in labels if len(image_files(root / signer / label)) < goal]
    if not pending:
        print(f"{signer}: all selected signs already have at least {goal} images.")
        return

    cam = cv2.VideoCapture(camera)
    try:
        if not cam.isOpened():
            raise RuntimeError(f"Cannot open camera {camera}. Try --camera 1.")

        print("Keep both hands inside the green rectangle. S: capture now. Q / Escape: quit.")
        for index, label in enumerate(pending, start=1):
            folder = root / signer / label
            folder.mkdir(parents=True, exist_ok=True)
            count = len(image_files(folder))
            next_shot = time.monotonic() + ready_seconds
            print(f"Sign {label}: {count}/{goal} images; get ready.")

            while count < goal:
                ok, frame = cam.read()
                if not ok or frame is None:
                    raise RuntimeError("Failed to read a frame from the camera.")

                height, width = frame.shape[:2]
                x1, y1 = width // 8, height // 6
                x2, y2 = 7 * width // 8, 5 * height // 6
                preview = frame.copy()
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                seconds_left = max(0.0, next_shot - time.monotonic())
                cv2.putText(
                    preview, f"{signer} | Sign: {label}  {count}/{goal}  ({index}/{len(pending)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                )
                cv2.putText(
                    preview, f"Next photo in {seconds_left:.1f}s | S: capture | Q: quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                )
                cv2.imshow("Collect", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print("Collection stopped; saved images will be kept.")
                    return
                if key in (ord("s"), ord("S")):
                    next_shot = time.monotonic()

                if time.monotonic() >= next_shot:
                    crop = frame[y1:y2, x1:x2]
                    path = next_image_path(folder, label)
                    ok, encoded = cv2.imencode(".png", crop)
                    if not ok:
                        raise RuntimeError(f"Failed to encode image for {path}")
                    # Exclusive creation also protects against a concurrent collector.
                    with path.open("xb") as image_file:
                        image_file.write(encoded.tobytes())
                    count += 1
                    print(f"Saved {path} ({count}/{goal})")
                    next_shot = time.monotonic() + interval

        print("Collection complete.")
    finally:
        cam.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signer", help="Signer folder name; prompted if omitted")
    parser.add_argument("--root", type=Path, default=DATA_DIR, help="Dataset root (default: repository data/)")
    parser.add_argument("--labels", nargs="+", type=str.upper, choices=LABELS, default=list(LABELS),
                        help="Signs to collect (default: A-Z, then 0-9)")
    parser.add_argument("--goal", type=int, default=20, help="Total images per sign, including existing images")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between photos (default: 2)")
    parser.add_argument("--ready-seconds", type=float, default=15.0,
                        help="Preparation time before each sign (default: 15)")
    args = parser.parse_args()

    if args.goal < 1:
        parser.error("--goal must be at least 1")
    if args.camera < 0:
        parser.error("--camera must be non-negative")
    for name, value in (("--interval", args.interval), ("--ready-seconds", args.ready_seconds)):
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be a finite, non-negative number")

    signer = (args.signer if args.signer is not None else input("Enter signer name: ")).strip()
    if (not signer or signer in {".", ".."} or signer.endswith(".")
            or any(char in '<>:"/\\|?*' or ord(char) < 32 for char in signer)
            or Path(signer).is_reserved()):
        parser.error("Signer must be a valid single folder name, such as Chirayu.")

    try:
        collect(args.root.resolve(), signer, args.labels, args.goal, args.camera,
                args.interval, args.ready_seconds)
    except (OSError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Collection failed: {error}\n")
    except KeyboardInterrupt:
        print("\nCollection interrupted; saved images will be kept.")


if __name__ == "__main__":
    main()
