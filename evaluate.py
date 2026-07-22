import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd
import voxelmorph as vxm
from PIL import Image
from tensorflow.keras.models import load_model

from utils import *

Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser(description="CLI Tool Description")
parser.add_argument("-d1", "--dataset-1", type=str, help="Path to the reference/target dataset", required=True)
parser.add_argument("-d2", "--dataset-2", type=str, help="Path to the dataset to be warped", required=True)
parser.add_argument("-c", "--csv", type=str, help="Path to the additional .csv file provided as part of the dataset", required=True)
parser.add_argument("-m", "--model", type=str, help="Path to the model", required=True)
args = parser.parse_args()

dataset_path_1 = args.dataset_1
dataset_path_2 = args.dataset_2
core_info_csv_path = args.csv
model_path = args.model

img_paths_1 = glob.glob(os.path.join(dataset_path_1, "Slides", "*"))
img_paths_2 = glob.glob(os.path.join(dataset_path_2, "Slides", "*"))

cutline_paths_1 = glob.glob(os.path.join(dataset_path_1, "Cutlines", "*"))
cutline_paths_2 = glob.glob(os.path.join(dataset_path_2, "Cutlines", "*"))

landmark_paths_1 = glob.glob(os.path.join(dataset_path_1, "Masks", "*"))
landmark_paths_2 = glob.glob(os.path.join(dataset_path_2, "Masks", "*"))

roi_paths_1 = []
roi_paths_2 = []
for path in cutline_paths_1:
    roi_path = os.path.join(dataset_path_1, "ROIs", os.path.basename(path))
    roi_paths_1.append(roi_path if os.path.exists(roi_path) else None)
for path in cutline_paths_2:
    roi_path = os.path.join(dataset_path_2, "ROIs", os.path.basename(path))
    roi_paths_2.append(roi_path if os.path.exists(roi_path) else None)

N = len(img_paths_1)

assert len(img_paths_2) == N
assert len(landmark_paths_1) == N
assert len(landmark_paths_2) == N
assert len(cutline_paths_1) == N
assert len(cutline_paths_2) == N
assert len(roi_paths_1) == N
assert len(roi_paths_2) == N

mask_binary_1 = []
mask_binary_2 = []

landmark_binary_1 = []
landmark_binary_2 = []

N_components = []
mask_components = []

core_components_1 = []
core_components_2 = []

core_info = pd.read_csv(core_info_csv_path)

for i in range(N):
    print("Pre-processing: " + img_paths_1[i])

    # Perform Otsu tissue segmentation
    mask_binary_1.append(segmentation(img_paths_1[i], roi_paths_1[i], level=2)[0])
    mask_binary_2.append(segmentation(img_paths_2[i], roi_paths_2[i], level=2)[0])

    # Read and save landmarks
    H, W = mask_binary_1[i].shape
    landmark_binary_1.append(read_image(landmark_paths_1[i], H, W))
    H, W = mask_binary_2[i].shape
    landmark_binary_2.append(read_image(landmark_paths_2[i], H, W))
    
    # Generate individual cores
    kernel_size = core_info[core_info["n"] == 0]["CoreDilation"].iloc[0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask_binary_1 = cv2.dilate(mask_binary_1[i], kernel)
    dilated_mask_binary_2 = cv2.dilate(mask_binary_2[i], kernel)
    _, dilated_mask_binary_2 = cv2.threshold(dilated_mask_binary_2, 0, 255, cv2.THRESH_BINARY)
    _, dilated_mask_binary_2 = cv2.threshold(dilated_mask_binary_2, 0, 255, cv2.THRESH_BINARY)

    contours_1, _ = cv2.findContours(dilated_mask_binary_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_2, _ = cv2.findContours(dilated_mask_binary_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_1 = sorted(contours_1, key=lambda c: len(c), reverse=True)
    contours_2 = sorted(contours_2, key=lambda c: len(c), reverse=True)

    # Match pairs of cores based on predefined values
    for index, row in core_info.iterrows():
        if row["n"] != i:
            continue
        
        core_contour_1 = contours_1[row["Component1"]]
        core_contour_2 = contours_2[row["Component2"]]

        M_1 = np.zeros_like(mask_binary_1[i])
        M_2 = np.zeros_like(mask_binary_2[i])
        cv2.drawContours(M_1, [core_contour_1], -1, color=1, thickness=-1)
        cv2.drawContours(M_2, [core_contour_2], -1, color=1, thickness=-1)

        # Apply cut lines
        M_1, N_1 = apply_splits(M_1, cutline_paths_1[i], row["Horizontal"] == 0)
        M_2, N_2 = apply_splits(M_2, cutline_paths_2[i], row["Horizontal"] == 0)

        assert N_1 == N_2

        N_components.append(N_1)
        mask_components.append(i)
        core_components_1.append(M_1)
        core_components_2.append(M_2)

# Load trained model
trained_model = load_model(
    model_path,
    compile=False,
    custom_objects={"VxmDense": vxm.networks.VxmDense}
)

print("Beginning warp...")
print(f"ID:\tPre Reg. TRE\tPost Aff. TRE\tPost Def. TRE\tPre Reg. Dice\tPost Aff. Dice\tPost Def. Dice")

for i in range(len(core_components_1)):
    pre_reg_tre = []
    post_affine_tre = []
    post_deform_tre = []
    pre_reg_dice = []
    post_affine_dice = []
    post_deform_dice = []

    padded_size = core_info.iloc[i]["PadSize"]
    target_size = 1024

    H, W = core_components_1[i].shape
    core_resized_2 = np.zeros_like(core_components_1[i])
    core_affined = np.zeros_like(core_components_1[i])
    core_warped = np.zeros_like(core_components_1[i])
    core_landmark_resized_2 = np.zeros_like(core_components_1[i])
    core_landmark_affined = np.zeros_like(core_components_1[i])
    core_landmark_warped = np.zeros_like(core_components_1[i])

    for j in range(1, N_components[i] + 1):
        mask_resized_1, landmark_resized_1, (y, x) = resize_mask(
            core_components_1[i] == j, (core_components_1[i] == j) * landmark_binary_1[mask_components[i]], padded_size=padded_size)
        mask_resized_2, landmark_resized_2, _ = resize_mask(
            core_components_2[i] == j, (core_components_2[i] == j) * landmark_binary_2[mask_components[i]], padded_size=padded_size)

        mask_affined, angle, _ = find_rotation(mask_resized_1, mask_resized_2, angle_range=(-30, 30), step=0.1)
        assert angle != -30
        assert angle != 30

        mask_resized_1 = prep(mask_resized_1)
        mask_affined = prep(mask_affined)
        
        mask_warped, flow = trained_model.predict([mask_affined, mask_resized_1], verbose=0)

        mask_resized_1 = mask_resized_1.squeeze()
        mask_affined = mask_affined.squeeze()
        mask_warped = mask_warped.squeeze()
        flow = flow.squeeze()

        mask_warped = cv2.resize(mask_warped, (padded_size, padded_size), interpolation=cv2.INTER_AREA)
        mask_warped = mask_warped[max(0, -y):min(padded_size, H-y), max(0, -x):min(padded_size, W-x)] > 0
        core_warped[max(0, y):min(padded_size+y, H), max(0, x):min(padded_size+x, W)][mask_warped] = 1

        mask_affined = cv2.resize(mask_affined, (padded_size, padded_size), interpolation=cv2.INTER_AREA)
        mask_affined = mask_affined[max(0, -y):min(padded_size, H-y), max(0, -x):min(padded_size, W-x)] > 0
        core_affined[max(0, y):min(padded_size+y, H), max(0, x):min(padded_size+x, W)][mask_affined] = 1

        mask_resized_2 = cv2.resize(mask_resized_2.astype(np.uint8), (padded_size, padded_size), interpolation=cv2.INTER_AREA)
        mask_resized_2 = mask_resized_2[max(0, -y):min(padded_size, H-y), max(0, -x):min(padded_size, W-x)] > 0
        core_resized_2[max(0, y):min(padded_size+y, H), max(0, x):min(padded_size+x, W)][mask_resized_2] = 1

        temp = (core_components_1[i] == j) * landmark_binary_1[mask_components[i]]
        if np.sum(temp) != 0:
            landmark_affined = rotate(landmark_resized_2, angle, reshape=False, order=0)
            landmark_warped = warp(landmark_affined, flow)

            landmark_warped = cv2.resize(landmark_warped, (padded_size, padded_size), interpolation=cv2.INTER_AREA)
            landmark_warped = landmark_warped[max(0, -y):min(padded_size, H-y), max(0, -x):min(padded_size, W-x)] > 0
            core_landmark_warped[max(0, y):min(padded_size+y, H), max(0, x):min(padded_size+x, W)][landmark_warped] = 1

            landmark_affined = cv2.resize(landmark_affined.astype(np.uint8), (padded_size, padded_size), interpolation=cv2.INTER_AREA)
            landmark_affined = landmark_affined[max(0, -y):min(padded_size, H-y), max(0, -x):min(padded_size, W-x)] > 0
            core_landmark_affined[max(0, y):min(padded_size+y, H), max(0, x):min(padded_size+x, W)][landmark_affined] = 1

            landmark_resized_2 = cv2.resize(landmark_resized_2.astype(np.uint8), (padded_size, padded_size), interpolation=cv2.INTER_AREA)
            landmark_resized_2 = landmark_resized_2[max(0, -y):min(padded_size, H-y), max(0, -x):min(padded_size, W-x)] > 0
            core_landmark_resized_2[max(0, y):min(padded_size+y, H), max(0, x):min(padded_size+x, W)][landmark_resized_2] = 1

    core_landmark_resized_1 = (core_components_1[i] > 0) * landmark_binary_1[mask_components[i]]
    pre_reg_tre = compute_tre(core_landmark_resized_1, core_landmark_resized_2)
    post_affine_tre = compute_tre(core_landmark_resized_1, core_landmark_affined)
    post_deform_tre = compute_tre(core_landmark_resized_1, core_landmark_warped)

    core_resized_1 = core_components_1[i] > 0
    pre_reg_dice = dice_coefficient(core_resized_1, core_resized_2)
    post_affine_dice = dice_coefficient(core_resized_1, core_affined)
    post_deform_dice = dice_coefficient(core_resized_1, core_warped)

    print(f"{i}:\t{np.mean(pre_reg_tre)}\t{np.mean(post_affine_tre)}\t{np.mean(post_deform_tre)}\t{np.mean(pre_reg_dice)}\t{np.mean(post_affine_dice)}\t{np.mean(post_deform_dice)}")
