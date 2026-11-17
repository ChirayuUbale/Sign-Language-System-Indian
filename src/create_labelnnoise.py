
from pathlib import Path
import random
import shutil
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split



ROOT = Path(__file__).resolve().parent.parent

DATA_FOLDER = ROOT / "Preprocessed_Data_Final"
OUTPUT_FOLDER = ROOT / "Label_Noise_Experiments"

NOISE_LEVELS = [0, 10, 20]
TEST_SIZE = 0.30
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp"
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)



records = []

if not DATA_FOLDER.is_dir():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FOLDER}"
    )

for signer_folder in sorted(DATA_FOLDER.iterdir()):

    if not signer_folder.is_dir():
        continue

    signer = signer_folder.name

    for label_folder in sorted(signer_folder.iterdir()):

        if not label_folder.is_dir():
            continue

        label = label_folder.name

        for image_path in sorted(label_folder.iterdir()):

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                records.append({
                    "image_path": str(image_path.resolve()),
                    "signer": signer,
                    "original_label": label
                })

df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No images found.")

print("\n========== DATASET LOADED ==========")
print("Total images:", len(df))

print("\nImages per signer:")
print(df.groupby("signer").size())

print("\nImages per label:")
print(df.groupby("original_label").size())

signers = sorted(df["signer"].unique())
labels = sorted(df["original_label"].unique())

if len(signers) != 3:
    raise ValueError(
        f"Expected 3 signers, found {len(signers)}"
    )

if len(labels) != 36:
    raise ValueError(
        f"Expected 36 classes, found {len(labels)}"
    )



df["stratify_key"] = (
    df["signer"] + "_" + df["original_label"]
)

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=df["stratify_key"]
)

train_df = train_df.drop(
    columns=["stratify_key"]
).reset_index(drop=True)

test_df = test_df.drop(
    columns=["stratify_key"]
).reset_index(drop=True)

print("\n========== TRAIN / TEST SPLIT ==========")
print("Training:", len(train_df))
print("Testing:", len(test_df))



TEST_FOLDER = OUTPUT_FOLDER / "Test_Clean"

# Rebuild generated test folder to avoid stale files
if TEST_FOLDER.exists():
    shutil.rmtree(TEST_FOLDER)

test_records = []

for _, row in test_df.iterrows():

    source = Path(row["image_path"])

    destination_dir = (
        TEST_FOLDER / row["original_label"]
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Include signer to prevent duplicate filenames
    destination_name = (
        f"{row['signer']}__{source.name}"
    )

    destination = (
        destination_dir / destination_name
    )

    shutil.copy2(source, destination)

    test_records.append({
        "image_path": str(destination),
        "signer": row["signer"],
        "original_label": row["original_label"],
        "assigned_label": row["original_label"],
        "is_noisy": False
    })

pd.DataFrame(test_records).to_csv(
    OUTPUT_FOLDER / "test_clean.csv",
    index=False
)

print("Clean test images copied:", len(test_records))



rng = random.Random(RANDOM_SEED)

shuffled_indices = list(range(len(train_df)))
rng.shuffle(shuffled_indices)

# Keep incorrect labels consistent between experiments
wrong_label_mapping = {}

for index in shuffled_indices:

    correct_label = train_df.loc[
        index, "original_label"
    ]

    possible_labels = [
        label
        for label in labels
        if label != correct_label
    ]

    wrong_label_mapping[index] = rng.choice(
        possible_labels
    )



summary = []
all_experiment_records = {}

for noise_level in NOISE_LEVELS:

    print(
        f"\n========== GENERATING {noise_level}% NOISE =========="
    )

    experiment_folder = (
        OUTPUT_FOLDER / f"Noise_{noise_level}" / "train"
    )

    if experiment_folder.exists():
        shutil.rmtree(experiment_folder)

    number_to_change = round(
        len(train_df) * noise_level / 100
    )

    selected_indices = set(
        shuffled_indices[:number_to_change]
    )

    experiment_records = []

    for index, row in train_df.iterrows():

        source = Path(row["image_path"])

        original_label = row["original_label"]

        is_noisy = index in selected_indices

        if is_noisy:
            assigned_label = wrong_label_mapping[index]
        else:
            assigned_label = original_label

        destination_dir = (
            experiment_folder / assigned_label
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination_name = (
            f"{row['signer']}__"
            f"{original_label}__"
            f"{source.name}"
        )

        destination = (
            destination_dir / destination_name
        )

        shutil.copy2(source, destination)

        experiment_records.append({
            "image_path": str(destination),
            "signer": row["signer"],
            "original_label": original_label,
            "assigned_label": assigned_label,
            "is_noisy": is_noisy
        })

    experiment_df = pd.DataFrame(
        experiment_records
    )

    experiment_df.to_csv(
        OUTPUT_FOLDER /
        f"train_noise_{noise_level}.csv",
        index=False
    )

    all_experiment_records[noise_level] = (
        experiment_df
    )

    incorrect_count = int(
        experiment_df["is_noisy"].sum()
    )

    correct_count = (
        len(experiment_df) - incorrect_count
    )

    summary.append({
        "noise_level": noise_level,
        "training_images": len(experiment_df),
        "correct_labels": correct_count,
        "incorrect_labels": incorrect_count,
        "test_images": len(test_df)
    })

    print("Images generated:", len(experiment_df))
    print("Correct labels:", correct_count)
    print("Incorrect labels:", incorrect_count)



summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT_FOLDER / "noise_summary.csv",
    index=False
)

print("\n========== EXPERIMENT SUMMARY ==========")
print(summary_df.to_string(index=False))



fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    ["Training", "Testing"],
    [len(train_df), len(test_df)]
)

ax.bar_label(bars, padding=3)

ax.set_title("Dataset Training and Testing Split")
ax.set_ylabel("Number of Images")

ax.set_ylim(0, len(train_df) * 1.15)

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "train_test_split.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)



x = np.arange(len(NOISE_LEVELS))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(
    x - width / 2,
    summary_df["correct_labels"],
    width,
    label="Correct Labels"
)

bars2 = ax.bar(
    x + width / 2,
    summary_df["incorrect_labels"],
    width,
    label="Incorrect Labels"
)

ax.set_xticks(x)
ax.set_xticklabels(
    [f"{level}%" for level in NOISE_LEVELS]
)

ax.set_title(
    "Correct vs Incorrect Training Labels"
)

ax.set_xlabel("Label Noise Level")
ax.set_ylabel("Number of Images")

ax.legend()

ax.bar_label(bars1, padding=3)
ax.bar_label(bars2, padding=3)

ax.set_ylim(
    0,
    summary_df["training_images"].max() * 1.15
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "label_noise_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)



demo_df = all_experiment_records[20]

noisy_examples = demo_df[
    demo_df["is_noisy"]
]

sample_count = min(
    12,
    len(noisy_examples)
)

if sample_count > 0:

    samples = noisy_examples.sample(
        n=sample_count,
        random_state=RANDOM_SEED
    )

    fig, axes = plt.subplots(
        3, 4,
        figsize=(15, 11)
    )

    axes = axes.flatten()

    for ax, (_, row) in zip(
        axes,
        samples.iterrows()
    ):

        with Image.open(row["image_path"]) as img:
            image = img.convert("RGB").copy()

        ax.imshow(image)

        ax.set_title(
            f"Signer: {row['signer']}\n"
            f"Original: {row['original_label']}\n"
            f"Assigned: {row['assigned_label']}",
            fontsize=10
        )

        ax.axis("off")

    for ax in axes[sample_count:]:
        ax.axis("off")

    plt.suptitle(
        "Examples of Incorrect Labels (20% Noise)",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER / "noisy_image_examples.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)



print("\n========== FINAL VERIFICATION ==========")

original_train_paths = set(
    train_df["image_path"]
)

original_test_paths = set(
    test_df["image_path"]
)

assert not (
    original_train_paths & original_test_paths
), "Training and testing images overlap."

for noise_level in NOISE_LEVELS:

    experiment_df = (
        all_experiment_records[noise_level]
    )

    expected = round(
        len(train_df) * noise_level / 100
    )

    actual = int(
        experiment_df["is_noisy"].sum()
    )

    assert actual == expected

    assert (
        experiment_df["original_label"]
        != experiment_df["assigned_label"]
    ).sum() == expected

    assert all(
        Path(path).is_file()
        for path in experiment_df["image_path"]
    )

    print(
        f"{noise_level}% verified: "
        f"{len(experiment_df)} images, "
        f"{actual} incorrect labels"
    )

print("\nClean testing images:", len(test_df))
print("All datasets verified.")
print("Output:", OUTPUT_FOLDER)
print("\nLABEL NOISE EXPERIMENT COMPLETED")
