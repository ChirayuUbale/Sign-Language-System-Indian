
import random
import csv
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Change this to your actual original dataset path.
ORIGINAL_DATASET = Path(
    r"C:\Users\YourName\Desktop\Indian_Sign_Language\Original_Dataset"
)

# New output directory.
OUTPUT_DIR = Path(
    r"C:\Users\YourName\Desktop\Indian_Sign_Language\Label_Noise_Experiments"
)

# Dataset split ratios.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Label noise experiments.
NOISE_LEVELS = [0, 10, 20, 30]

# Fixed seed for reproducibility.
RANDOM_SEED = 42

# Supported image formats.
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


# ============================================================
# 2. LOAD ORIGINAL DATASET
# ============================================================

def load_dataset(dataset_path):

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    classes = sorted(
        folder.name
        for folder in dataset_path.iterdir()
        if folder.is_dir()
    )

    if len(classes) < 2:
        raise ValueError(
            "At least two class folders are required."
        )

    dataset = []

    for class_name in classes:

        class_folder = dataset_path / class_name

        images = sorted(
            file for file in class_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in images:

            dataset.append({
                "image_path": image_path,
                "true_label": class_name
            })

        print(
            f"Class {class_name}: {len(images)} images"
        )

    if not dataset:
        raise ValueError("No images found in the dataset.")

    return dataset, classes


# ============================================================
# 3. SPLIT DATASET BEFORE INTRODUCING LABEL NOISE
# ============================================================

def split_dataset(dataset, classes):

    rng = random.Random(RANDOM_SEED)

    grouped_data = defaultdict(list)

    for item in dataset:
        grouped_data[item["true_label"]].append(item)

    train_data = []
    validation_data = []
    test_data = []

    for class_name in classes:

        images = grouped_data[class_name].copy()

        rng.shuffle(images)

        total = len(images)

        train_count = int(total * TRAIN_RATIO)
        validation_count = int(total * VALIDATION_RATIO)

        train_data.extend(images[:train_count])

        validation_data.extend(
            images[
                train_count:
                train_count + validation_count
            ]
        )

        test_data.extend(
            images[train_count + validation_count:]
        )

    rng.shuffle(train_data)
    rng.shuffle(validation_data)
    rng.shuffle(test_data)

    return train_data, validation_data, test_data


# ============================================================
# 4. GENERATE NESTED LABEL NOISE
# ============================================================

def generate_noise_assignments(train_data, classes):

    rng = random.Random(RANDOM_SEED)

    total_training_images = len(train_data)

    # Random order determines which samples are corrupted first.
    shuffled_indices = list(range(total_training_images))
    rng.shuffle(shuffled_indices)

    # Preselect incorrect labels for up to 30% of training data.
    corruption_labels = {}

    for index in shuffled_indices:

        original_label = train_data[index]["true_label"]

        possible_labels = [
            label for label in classes
            if label != original_label
        ]

        corruption_labels[index] = rng.choice(
            possible_labels
        )

    assignments = {}

    for noise_percentage in NOISE_LEVELS:

        number_to_corrupt = round(
            total_training_images * noise_percentage / 100
        )

        selected_indices = set(
            shuffled_indices[:number_to_corrupt]
        )

        assignments[noise_percentage] = {}

        for index, item in enumerate(train_data):

            if index in selected_indices:

                assigned_label = corruption_labels[index]
                is_noisy = True

            else:

                assigned_label = item["true_label"]
                is_noisy = False

            assignments[noise_percentage][index] = {
                "assigned_label": assigned_label,
                "is_noisy": is_noisy
            }

    return assignments


# ============================================================
# 5. COPY IMAGES INTO THE EXPERIMENT FOLDERS
# ============================================================

def copy_split(
    experiment_folder,
    split_name,
    data,
    classes,
    noise_assignments=None
):

    csv_records = []

    for index, item in enumerate(data):

        original_path = item["image_path"]
        true_label = item["true_label"]

        if split_name == "train":

            noise_info = noise_assignments[index]

            assigned_label = noise_info["assigned_label"]
            is_noisy = noise_info["is_noisy"]

        else:

            # Validation and test labels remain unchanged.
            assigned_label = true_label
            is_noisy = False

        destination_folder = (
            experiment_folder /
            split_name /
            assigned_label
        )

        destination_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # Prefix original class to prevent filename collisions.
        new_filename = (
            f"{true_label}__{original_path.name}"
        )

        destination_path = (
            destination_folder / new_filename
        )

        shutil.copy2(
            original_path,
            destination_path
        )

        csv_records.append({
            "image_filename": original_path.name,
            "original_path": str(original_path),
            "copied_path": str(destination_path),
            "true_label": true_label,
            "assigned_label": assigned_label,
            "is_label_noisy": is_noisy,
            "split": split_name
        })

    return csv_records


# ============================================================
# 6. CREATE CSV METADATA
# ============================================================

def save_metadata(records, output_path, noise_percentage):

    fieldnames = [
        "image_filename",
        "original_path",
        "copied_path",
        "true_label",
        "assigned_label",
        "is_label_noisy",
        "split",
        "noise_percentage"
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for record in records:

            record["noise_percentage"] = noise_percentage

            writer.writerow(record)


# ============================================================
# 7. CREATE ALL FOUR EXPERIMENTS
# ============================================================

def main():

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {OUTPUT_DIR}\n"
            "Choose a new output folder to avoid overwriting results."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading original dataset...")

    dataset, classes = load_dataset(
        ORIGINAL_DATASET
    )

    print("\nSplitting original dataset...")

    train_data, validation_data, test_data = split_dataset(
        dataset,
        classes
    )

    print("\nDataset split summary:")
    print("Training:", len(train_data))
    print("Validation:", len(validation_data))
    print("Testing:", len(test_data))
    print("Total:", len(dataset))

    # Generate nested noise configurations.
    print("\nGenerating label-noise assignments...")

    noise_assignments = generate_noise_assignments(
        train_data,
        classes
    )

    # Create all four experiments.
    for noise_percentage in NOISE_LEVELS:

        print(
            f"\nCreating {noise_percentage}% label-noise dataset..."
        )

        experiment_folder = (
            OUTPUT_DIR /
            f"Noise_{noise_percentage}_Percent"
        )

        experiment_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # Training images: noisy labels according to experiment.
        train_records = copy_split(
            experiment_folder,
            "train",
            train_data,
            classes,
            noise_assignments[noise_percentage]
        )

        # Validation: always clean.
        validation_records = copy_split(
            experiment_folder,
            "validation",
            validation_data,
            classes
        )

        # Testing: always clean.
        test_records = copy_split(
            experiment_folder,
            "test",
            test_data,
            classes
        )

        # Save metadata.
        all_records = (
            train_records +
            validation_records +
            test_records
        )

        save_metadata(
            all_records,
            experiment_folder / "metadata.csv",
            noise_percentage
        )

        corrupted_count = sum(
            record["is_label_noisy"]
            for record in train_records
        )

        print(
            f"Training images: {len(train_records)}"
        )

        print(
            f"Corrupted training labels: {corrupted_count}"
        )

        print(
            f"Clean validation images: {len(validation_records)}"
        )

        print(
            f"Clean testing images: {len(test_records)}"
        )

    print("\n====================================")
    print("ALL EXPERIMENTS CREATED SUCCESSFULLY")
    print("====================================")

    print(f"\nOutput location: {OUTPUT_DIR}")


# ============================================================
# 8. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
