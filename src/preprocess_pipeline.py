
from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np
import mediapipe as mp



ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "Preprocessed_Data_Final"

IMAGE_SIZE = 256

MAX_IMAGES = None

HAND_PADDING = 0.80

REQUIRE_TWO_HANDS_FOR_CROP = True

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

GAUSSIAN_KERNEL = (3, 3)
MEDIAN_KERNEL = 3

MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.35
MODEL_COMPLEXITY = 1

DRAW_LANDMARKS = True
DRAW_CONNECTIONS = True

# HSV, binarization and Canny are analysis-only.
# They do not erase pixels from the final image.
RUN_RESEARCH_ANALYSIS = True

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

mp_hands = mp.solutions.hands



def resize_with_padding(image, size=IMAGE_SIZE):

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError("Image has invalid dimensions")

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



def detect_hands(image, hands):

    height, width = image.shape[:2]

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    rgb.flags.writeable = False

    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return []

    detected_hands = []

    for hand_landmarks in result.multi_hand_landmarks:

        points = []

        for landmark in hand_landmarks.landmark:

            points.append({
                "x": float(landmark.x * width),
                "y": float(landmark.y * height),
                "z": float(landmark.z)
            })

        detected_hands.append(points)

    return detected_hands



def calculate_safe_crop(image, detected_hands):

    height, width = image.shape[:2]

    # If one or zero hands are detected, do not crop.
    # This prevents cutting off an undetected hand.
    if (
        REQUIRE_TWO_HANDS_FOR_CROP
        and len(detected_hands) < 2
    ):
        return image.copy(), 0, 0, "full_frame"

    if not detected_hands:
        return image.copy(), 0, 0, "full_frame"

    all_points = [
        point
        for hand_points in detected_hands
        for point in hand_points
    ]

    xs = [point["x"] for point in all_points]
    ys = [point["y"] for point in all_points]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    region_width = max_x - min_x
    region_height = max_y - min_y

    margin = HAND_PADDING * max(
        region_width,
        region_height
    )

    x1 = max(
        0,
        int(math.floor(min_x - margin))
    )

    y1 = max(
        0,
        int(math.floor(min_y - margin))
    )

    x2 = min(
        width,
        int(math.ceil(max_x + margin))
    )

    y2 = min(
        height,
        int(math.ceil(max_y + margin))
    )

    if x2 <= x1 or y2 <= y1:
        return image.copy(), 0, 0, "full_frame"

    cropped = image[y1:y2, x1:x2].copy()

    if cropped.size == 0:
        return image.copy(), 0, 0, "full_frame"

    return cropped, x1, y1, "cropped"



def preprocess_main_image(image):


    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )

    enhanced = clahe.apply(grayscale)


    gaussian = cv2.GaussianBlur(
        enhanced,
        GAUSSIAN_KERNEL,
        0
    )


    median = cv2.medianBlur(
        gaussian,
        MEDIAN_KERNEL
    )


    final, scale, offset_x, offset_y = (
        resize_with_padding(
            median,
            IMAGE_SIZE
        )
    )

    return final, scale, offset_x, offset_y



def calculate_research_analysis(image):


    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    threshold_value, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )


    edges = cv2.Canny(
        gray,
        50,
        150
    )

    return {
        "mean_hue": float(
            np.mean(hsv[:, :, 0])
        ),
        "otsu_threshold": float(
            threshold_value
        ),
        "binary_foreground_ratio": float(
            np.count_nonzero(binary) / binary.size
        ),
        "edge_pixel_ratio": float(
            np.count_nonzero(edges) / edges.size
        )
    }



def draw_landmarks_on_image(
    image,
    detected_hands,
    crop_x,
    crop_y,
    scale,
    offset_x,
    offset_y
):

    output = image.copy()

    for hand_points in detected_hands:

        transformed_points = []

        for point in hand_points:

            local_x = point["x"] - crop_x
            local_y = point["y"] - crop_y

            final_x = int(round(
                local_x * scale + offset_x
            ))

            final_y = int(round(
                local_y * scale + offset_y
            ))

            final_x = int(np.clip(
                final_x,
                0,
                IMAGE_SIZE - 1
            ))

            final_y = int(np.clip(
                final_y,
                0,
                IMAGE_SIZE - 1
            ))

            transformed_points.append(
                (final_x, final_y)
            )

        if DRAW_CONNECTIONS:

            for start, end in mp_hands.HAND_CONNECTIONS:

                cv2.line(
                    output,
                    transformed_points[start],
                    transformed_points[end],
                    255,
                    2,
                    cv2.LINE_AA
                )

        if DRAW_LANDMARKS:

            for point in transformed_points:

                cv2.circle(
                    output,
                    point,
                    3,
                    255,
                    -1,
                    cv2.LINE_AA
                )

    return output



def extract_skeleton_connections(detected_hands):

    skeletons = []

    for hand_index, hand_points in enumerate(detected_hands):

        connections = []

        for start, end in mp_hands.HAND_CONNECTIONS:

            connections.append({
                "start_index": int(start),
                "end_index": int(end),
                "start": hand_points[start],
                "end": hand_points[end]
            })

        skeletons.append({
            "hand_index": hand_index,
            "connections": connections
        })

    return skeletons



def save_csv(filename, header, rows):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_DIR / filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)



def main():

    if not DATA_DIR.exists():
        print("ERROR: Dataset folder not found.")
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

    print("\n========== PREPROCESSING STARTED ==========")
    print("Original images:", total_original)
    print("Selected images:", len(image_paths))
    print("Final image size:", IMAGE_SIZE)
    print("Crop padding:", HAND_PADDING)
    print("Maximum detected hands:", MAX_HANDS)
    print("Landmark drawing:", DRAW_LANDMARKS)
    print("Skeleton drawing:", DRAW_CONNECTIONS)
    print("Output:", OUTPUT_DIR)

    success = 0
    failed = 0

    zero_hand_count = 0
    one_hand_count = 0
    two_hand_count = 0

    cropped_count = 0
    full_frame_count = 0

    failures = []
    detection_log = []
    analysis_log = []
    landmark_log = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE
    ) as hands:

        for index, path in enumerate(image_paths, 1):

            relative = path.relative_to(DATA_DIR)
            relative_str = str(relative)

            image = cv2.imread(str(path))

            if image is None:
                failed += 1
                failures.append([
                    relative_str,
                    "read_failed"
                ])
                continue

            try:


                detected_hands = detect_hands(
                    image,
                    hands
                )

                hand_count = len(detected_hands)


                cropped, crop_x, crop_y, crop_status = (
                    calculate_safe_crop(
                        image,
                        detected_hands
                    )
                )


                final, scale, ox, oy = (
                    preprocess_main_image(cropped)
                )


                if RUN_RESEARCH_ANALYSIS:
                    analysis = calculate_research_analysis(
                        cropped
                    )
                else:
                    analysis = {
                        "mean_hue": None,
                        "otsu_threshold": None,
                        "binary_foreground_ratio": None,
                        "edge_pixel_ratio": None
                    }


                final = draw_landmarks_on_image(
                    final,
                    detected_hands,
                    crop_x,
                    crop_y,
                    scale,
                    ox,
                    oy
                )


                skeleton = extract_skeleton_connections(
                    detected_hands
                )


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
                    final
                )

                if not saved:
                    raise RuntimeError("Image save failed")

                success += 1

                if hand_count == 0:
                    zero_hand_count += 1
                elif hand_count == 1:
                    one_hand_count += 1
                elif hand_count == 2:
                    two_hand_count += 1

                if crop_status == "cropped":
                    cropped_count += 1
                else:
                    full_frame_count += 1

                status = (
                    "two_detected"
                    if hand_count == 2
                    else "review_detection"
                )

                detection_log.append([
                    relative_str,
                    hand_count,
                    crop_status,
                    status
                ])

                analysis_log.append([
                    relative_str,
                    analysis["mean_hue"],
                    analysis["otsu_threshold"],
                    analysis["binary_foreground_ratio"],
                    analysis["edge_pixel_ratio"]
                ])

                landmark_log.append([
                    relative_str,
                    hand_count,
                    json.dumps(detected_hands),
                    json.dumps(skeleton)
                ])

            except Exception as error:

                failed += 1

                failures.append([
                    relative_str,
                    str(error)
                ])

            if index % 100 == 0:
                print(
                    f"Processed {index}/{len(image_paths)}"
                )


    save_csv(
        "failures.csv",
        ["image", "reason"],
        failures
    )

    save_csv(
        "hand_detection_log.csv",
        [
            "image",
            "detected_hands",
            "crop_status",
            "status"
        ],
        detection_log
    )

    save_csv(
        "research_analysis.csv",
        [
            "image",
            "mean_hue",
            "otsu_threshold",
            "binary_foreground_ratio",
            "edge_pixel_ratio"
        ],
        analysis_log
    )

    save_csv(
        "mediapipe_landmarks.csv",
        [
            "image",
            "detected_hands",
            "landmarks_json",
            "skeleton_connections_json"
        ],
        landmark_log
    )


    print("\n========== FINAL RESULTS ==========")
    print("Original images:", total_original)
    print("Attempted:", len(image_paths))
    print("Successfully processed:", success)
    print("Failed:", failed)

    print("\nHAND DETECTION")
    print("Zero hands:", zero_hand_count)
    print("One hand:", one_hand_count)
    print("Two hands:", two_hand_count)

    print("\nCROPPING")
    print("Images cropped:", cropped_count)
    print("Full-frame images:", full_frame_count)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nPreprocessing completed.")


if __name__ == "__main__":
    main()
