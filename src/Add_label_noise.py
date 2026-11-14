
from pathlib import Path
import csv

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "Preprocessed_Data_Final"

IMAGE_SIZE = 256
HAND_PADDING = 0.60

# None = process all 2160 images
MAX_IMAGES = None

# False is recommended for SURF feature extraction.
# True draws the 21 landmarks and connections on each hand.
DRAW_SKELETON = True

# Preserve original hand details instead of aggressively
# removing fingers using skin-colour segmentation.
USE_SKIN_MASK = False

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

mp_hands = mp.solutions.hands


# ============================================================
# RESIZE IMAGE WITHOUT DISTORTION
# ============================================================

def resize_with_padding(image, size=IMAGE_SIZE):

    height, width = image.shape[:2]

    scale = min(size / width, size / height)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    interpolation = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )

    if image.ndim == 2:
        canvas = np.zeros(
            (size, size),
            dtype=np.uint8
        )
    else:
        canvas = np.zeros(
            (size, size, image.shape[2]),
            dtype=np.uint8
        )

    offset_x = (size - new_width) // 2
    offset_y = (size - new_height) // 2

    canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width
    ] = resized

    return canvas, scale, offset_x, offset_y


# ============================================================
# DETECT BOTH HANDS USING MEDIAPIPE
# ============================================================

def detect_hands(image, hands):

    height, width = image.shape[:2]

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    rgb.flags.writeable = False

    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    all_hands_points = []

    for hand_landmarks in result.multi_hand_landmarks:

        points = []

        for landmark in hand_landmarks.landmark:

            x = landmark.x * width
            y = landmark.y * height

            points.append((x, y))

        all_hands_points.append(points)

    return all_hands_points


# ============================================================
# GENERATE SAFE CROP FOR BOTH HANDS
# ============================================================

def calculate_crop(image, all_hands_points):

    height, width = image.shape[:2]

    all_points = [
        point
        for hand_points in all_hands_points
        for point in hand_points
    ]

    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]

    min_x = min(xs)
    max_x = max(xs)

    min_y = min(ys)
    max_y = max(ys)

    hand_width = max_x - min_x
    hand_height = max_y - min_y

    margin = HAND_PADDING * max(
        hand_width,
        hand_height
    )

    x1 = max(0, int(np.floor(min_x - margin)))
    y1 = max(0, int(np.floor(min_y - margin)))

    x2 = min(width, int(np.ceil(max_x + margin)))
    y2 = min(height, int(np.ceil(max_y + margin)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


# ============================================================
# RESEARCH-BASED SKIN SEGMENTATION
# OPTIONAL: NOT USED BY DEFAULT
# ============================================================

def create_skin_mask(image):

    # Gaussian blurring
    blurred = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    # BGR to HSV
    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    # BGR to YCrCb
    ycrcb = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2YCrCb
    )

    # HSV thresholding and binarization
    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, 20, 35], dtype=np.uint8),
        np.array([25, 255, 255], dtype=np.uint8)
    )

    # YCrCb thresholding
    ycrcb_mask = cv2.inRange(
        ycrcb,
        np.array([0, 125, 75], dtype=np.uint8),
        np.array([255, 180, 140], dtype=np.uint8)
    )

    # Combine both masks
    mask = cv2.bitwise_and(
        hsv_mask,
        ycrcb_mask
    )

    # Morphological operations
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Preserve multiple contours, not just the largest
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final_mask = np.zeros_like(mask)

    minimum_area = 0.001 * mask.size

    for contour in contours:

        if cv2.contourArea(contour) >= minimum_area:

            cv2.drawContours(
                final_mask,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED
            )

    return final_mask


# ============================================================
# DRAW LANDMARKS FOR BOTH HANDS
# ============================================================

def draw_hand_skeleton(
    image,
    all_hands_points,
    x1,
    y1,
    scale,
    offset_x,
    offset_y
):

    for hand_points in all_hands_points:

        transformed_points = []

        for px, py in hand_points:

            new_x = int(round(
                (px - x1) * scale + offset_x
            ))

            new_y = int(round(
                (py - y1) * scale + offset_y
            ))

            new_x = int(np.clip(
                new_x, 0, IMAGE_SIZE - 1
            ))

            new_y = int(np.clip(
                new_y, 0, IMAGE_SIZE - 1
            ))

            transformed_points.append(
                (new_x, new_y)
            )

        # Draw connections
        for start, end in mp_hands.HAND_CONNECTIONS:

            cv2.line(
                image,
                transformed_points[start],
                transformed_points[end],
                255,
                2,
                cv2.LINE_AA
            )

        # Draw 21 landmark points
        for point in transformed_points:

            cv2.circle(
                image,
                point,
                3,
                255,
                -1,
                cv2.LINE_AA
            )

    return image


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_image(image, hands):

    # 1. Detect up to two hands
    all_hands_points = detect_hands(
        image,
        hands
    )

    if all_hands_points is None:
        return None, "no_hand_detected", 0

    detected_count = len(all_hands_points)

    # For a two-handed gesture, detecting only one hand
    # is insufficient. Use full image to avoid cropping
    # out the undetected hand.
    if detected_count == 1:

        height, width = image.shape[:2]

        x1, y1 = 0, 0
        x2, y2 = width, height

    else:

        bbox = calculate_crop(
            image,
            all_hands_points
        )

        if bbox is None:
            return None, "invalid_crop", detected_count

        x1, y1, x2, y2 = bbox

    # 2. Crop image
    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        return None, "empty_crop", detected_count

    # 3. Gaussian filtering
    blurred = cv2.GaussianBlur(
        cropped,
        (3, 3),
        0
    )

    # 4. Grayscale conversion
    grayscale = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2GRAY
    )

    # 5. Optional research skin segmentation
    if USE_SKIN_MASK:

        mask = create_skin_mask(cropped)

        # Avoid applying an empty mask
        if cv2.countNonZero(mask) > 0:

            grayscale = cv2.bitwise_and(
                grayscale,
                grayscale,
                mask=mask
            )

    # 6. Resize to 256 x 256
    final, scale, offset_x, offset_y = (
        resize_with_padding(
            grayscale,
            IMAGE_SIZE
        )
    )

    # 7. Optional skeleton drawing
    if DRAW_SKELETON:

        final = draw_hand_skeleton(
            final,
            all_hands_points,
            x1,
            y1,
            scale,
            offset_x,
            offset_y
        )

    # ONE FINAL IMAGE
    return final, None, detected_count


# ============================================================
# PROCESS COMPLETE DATASET
# ============================================================

def main():

    if not DATA_DIR.exists():

        print("ERROR: data folder not found.")
        print("Expected:", DATA_DIR)

        return

    image_paths = sorted(
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in EXTENSIONS
    )

    total_original = len(image_paths)

    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========== DATASET PREPROCESSING ==========")
    print("Total original images:", total_original)
    print("Images selected:", len(image_paths))
    print("Maximum hands:", 2)
    print("Crop padding:", HAND_PADDING)
    print("Skin segmentation:", USE_SKIN_MASK)
    print("Draw skeleton:", DRAW_SKELETON)

    success = 0
    failed = 0

    one_hand_count = 0
    two_hand_count = 0

    failures = []
    detection_log = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        for index, path in enumerate(image_paths, 1):

            relative = path.relative_to(DATA_DIR)

            image = cv2.imread(str(path))

            if image is None:

                failed += 1

                failures.append(
                    (str(relative), "image_read_failed")
                )

                continue

            processed, error, hand_count = (
                preprocess_image(
                    image,
                    hands
                )
            )

            if error is not None:

                failed += 1

                failures.append(
                    (str(relative), error)
                )

                continue

            if hand_count == 1:
                one_hand_count += 1

            elif hand_count == 2:
                two_hand_count += 1

            destination = (
                OUTPUT_DIR
                / relative.with_suffix(".png")
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            saved = cv2.imwrite(
                str(destination),
                processed
            )

            if saved:

                success += 1

                detection_log.append(
                    (
                        str(relative),
                        hand_count,
                        "review" if hand_count == 1 else "ok"
                    )
                )

            else:

                failed += 1

                failures.append(
                    (str(relative), "save_failed")
                )

            if index % 100 == 0:

                print(
                    f"Processed {index}/{len(image_paths)}"
                )

    # Save failure log
    with open(
        OUTPUT_DIR / "failures.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image",
            "reason"
        ])

        writer.writerows(failures)

    # Save detection log
    with open(
        OUTPUT_DIR / "hand_detection_log.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image",
            "detected_hands",
            "status"
        ])

        writer.writerows(detection_log)

    print("\n========== FINAL RESULTS ==========")

    print("Original images:", total_original)
    print("Attempted:", len(image_paths))
    print("Successfully saved:", success)
    print("Failed:", failed)
    print("One hand detected:", one_hand_count)
    print("Two hands detected:", two_hand_count)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nPreprocessing completed.")


if __name__ == "__main__":
    main()
