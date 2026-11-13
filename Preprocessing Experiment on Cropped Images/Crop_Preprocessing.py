import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm



# Input folder containing cropped palm images
INPUT_FOLDER = Path("data_hand_crops")

# Output folder for this experiment
OUTPUT_FOLDER = Path("Experiment/cropped_preprocessing_outputs")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Supported image formats
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# Collect all images, including images in subfolders
image_paths = [
    p for p in INPUT_FOLDER.rglob("*")
    if p.suffix.lower() in IMAGE_EXTENSIONS
]

print("Total cropped images:", len(image_paths))
print("Input folder:", INPUT_FOLDER)
print("Output folder:", OUTPUT_FOLDER)

if len(image_paths) == 0:
    raise ValueError("No images found. Check your data_hand_crops folder.")