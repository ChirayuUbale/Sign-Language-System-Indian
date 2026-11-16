
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image


DATA_FOLDER = Path("data")
OUTPUT_FOLDER = Path("Label_Noise_Experiments")

OUTPUT_FOLDER.mkdir(exist_ok=True)

NOISE_LEVELS = [0, 10, 20, 30, 40]

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


image_data = []

for signer_folder in sorted(DATA_FOLDER.iterdir()):

    if not signer_folder.is_dir():
        continue

    signer = signer_folder.name

    for label_folder in sorted(signer_folder.iterdir()):

        if not label_folder.is_dir():
            continue

        label = label_folder.name

        for image_path in label_folder.iterdir():

            if image_path.suffix.lower() in IMAGE_EXTENSIONS:

                image_data.append({
                    "image_path": str(image_path),
                    "signer": signer,
                    "original_label": label,
                    "noisy_label": label,
                    "is_noisy": False
                })

df = pd.DataFrame(image_data)

if df.empty:
    raise ValueError(
        "No images found. Check your data folder location."
    )

print("Dataset loaded successfully!")
print("\nTotal images:", len(df))

print("\nImages per signer:")
print(df.groupby("signer").size())

print("\nImages per label:")
print(df.groupby("original_label").size())

signers = sorted(df["signer"].unique())

if len(signers) != 3:
    raise ValueError(
        f"Expected 3 signers, but found {len(signers)}: {signers}"
    )


all_labels = sorted(df["original_label"].unique())

if len(all_labels) < 2:
    raise ValueError("At least two labels are required.")

for noise_level in NOISE_LEVELS:

    noisy_df = df.copy()

    number_to_change = round(
        len(noisy_df) * noise_level / 100
    )

    selected_indices = random.sample(
        list(noisy_df.index),
        number_to_change
    )

    for index in selected_indices:

        correct_label = noisy_df.loc[
            index, "original_label"
        ]

        possible_labels = [
            label for label in all_labels
            if label != correct_label
        ]

        wrong_label = random.choice(possible_labels)

        noisy_df.loc[index, "noisy_label"] = wrong_label
        noisy_df.loc[index, "is_noisy"] = True

    output_csv = (
        OUTPUT_FOLDER / f"noisy_labels_{noise_level}.csv"
    )

    noisy_df.to_csv(output_csv, index=False)

    print("\n------------------------------")
    print(f"NOISE LEVEL: {noise_level}%")
    print("------------------------------")
    print("Total images:", len(noisy_df))
    print("Incorrect labels:", noisy_df["is_noisy"].sum())
    print("Saved:", output_csv)


demo_df = pd.read_csv(
    OUTPUT_FOLDER / "noisy_labels_20.csv"
)

noisy_examples = demo_df[
    demo_df["is_noisy"] == True
].sample(
    n=min(12, len(demo_df[demo_df["is_noisy"] == True])),
    random_state=RANDOM_SEED
)

fig, axes = plt.subplots(3, 4, figsize=(15, 11))

axes = axes.flatten()

for ax, (_, row) in zip(axes, noisy_examples.iterrows()):

    image_path = Path(row["image_path"])

    image = Image.open(image_path).convert("RGB")

    ax.imshow(image)

    ax.set_title(
        f"Signer: {row['signer']}\n"
        f"Original: {row['original_label']}\n"
        f"Noisy: {row['noisy_label']}",
        fontsize=10
    )

    ax.axis("off")

for ax in axes[len(noisy_examples):]:
    ax.axis("off")

plt.suptitle(
    "Random Label Noise - 20% Experiment",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "noisy_image_examples.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


noise_counts = []

for noise_level in NOISE_LEVELS:

    result = pd.read_csv(
        OUTPUT_FOLDER / f"noisy_labels_{noise_level}.csv"
    )

    noise_counts.append(result["is_noisy"].sum())

plt.figure(figsize=(9, 5))

plt.bar(
    [f"{level}%" for level in NOISE_LEVELS],
    noise_counts
)

plt.title("Number of Incorrect Labels at Each Noise Level")
plt.xlabel("Noise Percentage")
plt.ylabel("Number of Incorrect Labels")

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "noise_distribution.png",
    dpi=300
)

plt.show()

print("\n================================")
print("LABEL NOISE EXPERIMENT COMPLETED")
print("================================")
print("Check the Label_Noise_Experiments folder.")
