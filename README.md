# Indian-Sign-Language-System

Signer-independent Indian Sign Language (A–Z) recognition: SURF+BoVW+SVM vs. CNN, tested on a multi-signer dataset with controlled label noise.

## Dataset

The original images are stored in `data/`, organized by signer and sign class. The dataset contains 2,160 images across Areeb, Ayomide, and Chirayu (720 per signer).

## Collecting images

`src/collection.py` captures webcam images for A-Z and 0-9, saving clean color crops as `data/<signer>/<label>/<label>_001.png`. It resumes existing folders without overwriting images and skips signs that already meet the target of 20 images. Each sign starts with 15 seconds to prepare, followed by one photo every 2 seconds. Keep both hands inside the green rectangle; press **S** to capture immediately or **Q / Escape** to quit.

```bash
python src/collection.py                         # prompts for the signer name
python src/collection.py --signer NewSigner       # collects all 36 signs
python src/collection.py --signer Chirayu --labels A B --goal 25
```

The last command adds five new photos each for A and B when those folders already contain 20 images. Use `--camera 1` to select another webcam, or `--help` for timing options. The crop uses the original collector's proportions and preserves the camera's resolution without resizing or preprocessing.

## Preprocessing Experiments

The notebooks in `Experiment/` show image preprocessing steps used to explore the Indian Sign Language dataset. Each notebook takes a deterministic sample of 30 images: 10 from each signer (Areeb, Ayomide, and Chirayu), using random seed 42. The notebooks display image results; they do not calculate coverage percentages or write measurement CSV files.

### Techniques

| Notebook | Method and settings | Visual output |
| --- | --- | --- |
| [HSV Colour Space Conversion](<Experiment/HSV Colour Space Conversion.ipynb>) | Converts OpenCV BGR images to HSV and splits the H, S, and V channels. | Original image, grayscale channel views, and hue histogram. |
| [Skin Colour Thresholding](<Experiment/Skin Colour Thresholding.ipynb>) | Applies `cv2.inRange` with HSV lower bound `[0, 30, 40]` and upper bound `[25, 255, 255]`. | Original image beside the binary skin-color mask. |
| [Image Binarization](<Experiment/Image Binarization.ipynb>) | Applies a threshold of 127 to the `cv2.inRange` mask. | Threshold mask beside binary result. The range mask is already binary, so this threshold does not change which pixels are selected. |
| [Gaussian Blurring](<Experiment/Gaussian Blurring.ipynb>) | Applies a 5x5 Gaussian blur with sigma 0 to the threshold mask. | Mask before and after blur. |
| [Median Filtering](<Experiment/Median Filtering.ipynb>) | Applies a 5x5 median filter to the threshold mask. | Mask before and after filtering. |
| [Morphological Operations](<Experiment/Morphological Operations.ipynb>) | Uses a 5x5 elliptical kernel for opening followed by closing. | Original mask, opened mask, and closed mask. |
| [Maximum Contour Extraction](<Experiment/Maximum Contour Extraction.ipynb>) | Applies opening and closing, finds external contours, and outlines the largest contour. | Original image with the largest detected contour drawn in green. |
| [Canny Edge Detection](<Experiment/Canny Edge Detection.ipynb>) | Applies opening and closing, fills the largest external contour, then runs Canny with thresholds 50 and 150. | Extracted contour region and its Canny edges. |

### Saved images

Each notebook contains its sample visualizations. Image-only contact sheets are also saved in `preprocessing_outputs/`:

- [HSV conversion images](<preprocessing_outputs/hsv_colour_space_conversion_images.png>)
- [Skin thresholding images](<preprocessing_outputs/skin_colour_thresholding_images.png>)
- [Binarization images](<preprocessing_outputs/image_binarization_images.png>)
- [Gaussian blur images](<preprocessing_outputs/gaussian_blurring_images.png>)
- [Median filter images](<preprocessing_outputs/median_filtering_images.png>)
- [Morphological operation images](<preprocessing_outputs/morphological_operations_images.png>)
- [Maximum contour images](<preprocessing_outputs/maximum_contour_extraction_images.png>)
- [Canny edge images](<preprocessing_outputs/canny_edge_detection_images.png>)

### Notes

The HSV threshold is an exploratory color mask, not a hand annotation. It may include clothing or background pixels and miss parts of a hand under different lighting or camera conditions. The filters and contour operations change the mask but do not establish that the hand was segmented correctly. Use annotated hand masks to evaluate segmentation quality, and a held-out signer split to evaluate recognition performance.

## CNN Feature Inputs

`src/build_feature_inputs.py` runs SURF on the images in `Preprocessed_Data_Final/` and writes to `Feature_Inputs/`:

- `surf_map/`: SURF keypoint heat map for each image
- `skeleton/`: MediaPipe skeleton on a black background, drawn from `mediapipe_landmarks.csv`
- `surf_keypoints/`: SURF keypoints and 64-value descriptors (`.npz`)
- `feature_log.csv` and `preview.png`

`src/preprocess_pipeline.py` is a copy of `src/Preprocesspipeline` from the PreprocessingPipeline branch, renamed so its functions can be imported.

`src/features/cnn_inputs.py` stacks the preprocessed image, SURF map and skeleton into 100x100x3 arrays for the `Label_Noise_Experiments` splits.

```bash
python src/build_feature_inputs.py --limit 10   # test on 10 images
python src/build_feature_inputs.py              # all images
python src/features/cnn_inputs.py --export      # writes Feature_Inputs/stacked_100/<split>.npz
```

SURF is patented and is not included in the OpenCV pip wheels. Build OpenCV with it enabled (Python 3.12 or lower for MediaPipe 0.10.14):

```bash
pip uninstall -y opencv-python opencv-contrib-python
CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON" ENABLE_CONTRIB=1 \
  pip install --no-binary opencv-contrib-python opencv-contrib-python==4.11.0.86
```

Note: `data/` contains exact duplicate images (1,309 unique out of 2,160), so 375 of the 648 images in `Label_Noise_Experiments/Test_Clean` also appear in training.
