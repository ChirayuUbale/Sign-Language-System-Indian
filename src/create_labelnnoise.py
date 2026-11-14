
import random
import csv
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\chirayu ubale\OneDrive\Desktop\Computer Vision\Indian-Sign-Language-System-repo"
)

# Original dataset: data/Signer_1/A, data/Signer_2/B, etc.
ORIGINAL_DATASET = PROJECT_DIR / "data"

# New experimental datasets will be created here.
OUTPUT_DIR = PROJECT_DIR / "Label_Noise_Experiments"

# Dataset split ratios.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Label noise experiments.
NOISE_LEVELS = [0, 10, 20, 30]

# Reproducibility.
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


# ============================================================
# 2. LOAD ALL SIGNERS AND THEIR LABELS
# ============================================================

def load_dataset():

    if not ORIGINAL_DATASET.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {ORIGINAL_DATASET}"
        )

    dataset = []
    signers = []
    all_labels = set()

    signer_folders = sorted(
        folder for folder in ORIGINAL_DATASET.iterdir()
        if folder.is_dir()
    )

    if not signer_folders:
        raise ValueError("No signer folders found inside data/")

    print("\n========== DATASET SCAN ==========")

    for signer_folder in signer_folders:

        signer_name = signer_folder.name
        signers.append(signer_name)

        print(f"\nSigner: {signer_name}")

        label_folders = sorted(
            folder for folder in signer_folder.iterdir()
            if folder.is_dir()
        )

        if not label_folders:
            raise ValueError(
                f"No label folders found for {signer_name}"
            )

        signer_count = 0

        for label_folder in label_folders:

            label = label_folder.name
            all_labels.add(label)

            images = sorted(
                file for file in label_folder.iterdir()
                if file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            )

            print(f"  Label {label}: {len(images)} images")

            for image_path in images:

                dataset.append({
                    "signer": signer_name,
                    "true_label": label,
                    "image_path": image_path
                })

            signer_count += len(images)

        print(f"Total images for {signer_name}: {signer_count}")

    if not dataset:
        raise ValueError("No images found in the dataset.")

    if len(all_labels) < 2:
        raise ValueError("At least two labels are required.")

    print("\nTotal signers:", len(signers))
    print("Total classes:", len(all_labels))
    print("Total images:", len(dataset))

    return dataset, signers, sorted(all_labels)


# ============================================================
# 3. SPLIT DATASET BY SIGNER AND LABEL
# ============================================================

def split_dataset(dataset):

    rng = random.Random(RANDOM_SEED)

    grouped_data = defaultdict(list)

    # Group images by signer and original class.
    for item in dataset:

        key = (
            item["signer"],
            item["true_label"]
        )

        grouped_data[key].append(item)

    train_data = []
    validation_data = []
    test_data = []

    for key in sorted(grouped_data):

        images = grouped_data[key].copy()

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
# 4. GENERATE NESTED SYMMETRIC LABEL NOISE
# ============================================================

def generate_noise_assignments(train_data, all_labels):

    rng = random.Random(RANDOM_SEED)

    # Group training images by signer and original label.
    grouped_data = defaultdict(list)

    for index, item in enumerate(train_data):

        key = (
            item["signer"],
            item["true_label"]
        )

        grouped_data[key].append(index)

    assignments = {
        percentage: {}
        for percentage in NOISE_LEVELS
    }

    for key in sorted(grouped_data):

        indices = grouped_data[key]

        original_label = key[1]

        # Randomize image selection within each signer/class.
        shuffled_indices = indices.copy()
        rng.shuffle(shuffled_indices)

        # Select an incorrect replacement label for each image.
        possible_labels = [
            label for label in all_labels
            if label != original_label
        ]

        replacement_labels = {
            index: rng.choice(possible_labels)
            for index in shuffled_indices
        }

        # Create nested noise configurations.
        for percentage in NOISE_LEVELS:

            number_to_corrupt = round(
                len(indices) * percentage / 100
            )

            selected_indices = set(
                shuffled_indices[:number_to_corrupt]
            )

            for index in indices:

                if index in selected_indices:

                    assigned_label = replacement_labels[index]
                    is_noisy = True

                else:

                    assigned_label = original_label
                    is_noisy = False

                assignments[percentage][index] = {
                    "assigned_label": assigned_label,
                    "is_noisy": is_noisy
                }

    return assignments


# ============================================================
# 5. COPY IMAGES INTO THEIR ASSIGNED CLASS FOLDERS
# ============================================================

def copy_split(
    experiment_folder,
    split_name,
    data,
    noise_assignments=None
):

    records = []

    for index, item in enumerate(data):

        signer = item["signer"]
        original_label = item["true_label"]
        original_path = item["image_path"]

        if split_name == "train":

            noise_info = noise_assignments[index]

            assigned_label = noise_info["assigned_label"]
            is_noisy = noise_info["is_noisy"]

        else:

            # Validation and test labels always remain correct.
            assigned_label = original_label
            is_noisy = False

        # Preserve signer structure and use assigned class folder.
        destination_folder = (
            experiment_folder /
            split_name /
            signer /
            assigned_label
        )

        destination_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # Include original class in filename to prevent collisions.
        new_filename = (
            f"{original_label}__{original_path.name}"
        )

        destination_path = destination_folder / new_filename

        shutil.copy2(
            original_path,
            destination_path
        )

        records.append({

            "signer": signer,

            "image_filename": original_path.name,

            "original_path": str(original_path),

            "copied_path": str(destination_path),

            "original_label": original_label,

            "assigned_label": assigned_label,

            "is_label_noisy": is_noisy,

            "split": split_name

        })

    return records


# ============================================================
# 6. SAVE CSV METADATA
# ============================================================

def save_metadata(records, output_path, noise_percentage):

    fieldnames = [
        "signer",
        "image_filename",
        "original_path",
        "copied_path",
        "original_label",
        "assigned_label",
        "is_label_noisy",
        "split",
        "noise_percentage",
        "random_seed"
    ]

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
            record["random_seed"] = RANDOM_SEED

            writer.writerow(record)


# ============================================================
# 7. PRINT EXPERIMENT SUMMARY
# ============================================================

def print_summary(
    percentage,
    train_records,
    validation_records,
    test_records
):

    noisy_count = sum(
        record["is_label_noisy"]
        for record in train_records
    )

    clean_count = len(train_records) - noisy_count

    print("\n------------------------------------")
    print(f"LABEL NOISE: {percentage}%")
    print("------------------------------------")

    print("Training images:", len(train_records))
    print("Correct training labels:", clean_count)
    print("Incorrect training labels:", noisy_count)

    print("Validation images:", len(validation_records))
    print("Testing images:", len(test_records))


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main():

    # Protect existing experimental results.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):

        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIR}\n"
            "Please choose another output folder or "
            "move the existing experiments before running."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load all signers and classes.
    dataset, signers, all_labels = load_dataset()

    # Split before introducing label noise.
    print("\n========== DATASET SPLITTING ==========")

    train_data, validation_data, test_data = split_dataset(
        dataset
    )

    print("Training:", len(train_data))
    print("Validation:", len(validation_data))
    print("Testing:", len(test_data))

    # Generate nested label-noise configurations.
    print("\n========== GENERATING LABEL NOISE ==========")

    assignments = generate_noise_assignments(
        train_data,
        all_labels
    )

    # Create all four experimental datasets.
    for percentage in NOISE_LEVELS:

        print(f"\nCreating {percentage}% experiment...")

        experiment_folder = (
            OUTPUT_DIR /
            f"Noise_{percentage}_Percent"
        )

        experiment_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # Training: apply selected label noise.
        train_records = copy_split(
            experiment_folder,
            "train",
            train_data,
            assignments[percentage]
        )

        # Validation: clean labels.
        validation_records = copy_split(
            experiment_folder,
            "validation",
            validation_data
        )

        # Testing: clean labels.
        test_records = copy_split(
            experiment_folder,
            "test",
            test_data
        )

        all_records = (
            train_records +
            validation_records +
            test_records
        )

        # Save complete metadata.
        save_metadata(
            all_records,
            experiment_folder / "metadata.csv",
            percentage
        )

        print_summary(
            percentage,
            train_records,
            validation_records,
            test_records
        )

    print("\n========================================")
    print("ALL FOUR EXPERIMENTS CREATED SUCCESSFULLY")
    print("========================================")

    print("\nOutput directory:")
    print(OUTPUT_DIR)


# ============================================================
# 9. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
