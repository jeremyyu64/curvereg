import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import voxelmorph as vxm
from PIL import Image

from utils import *

Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser(description="CLI Tool Description")
parser.add_argument("-d1", "--dataset-1", type=str, help="Path to the reference/target dataset", required=True)
parser.add_argument("-d2", "--dataset-2", type=str, help="Path to the dataset to be warped", required=True)
parser.add_argument("-c", "--csv", type=str, help="Path to the additional .csv file provided as part of the dataset", required=True)
parser.add_argument("-o", "--output", type=str, help="Path to the output model", required=True)
args = parser.parse_args()

dataset_path_1 = args.dataset_1
dataset_path_2 = args.dataset_2
core_info_csv_path = args.csv
model_path = args.output

img_paths_1 = glob.glob(os.path.join(dataset_path_1, "Slides", "*"))
img_paths_2 = glob.glob(os.path.join(dataset_path_2, "Slides", "*"))

cutline_paths_1 = glob.glob(os.path.join(dataset_path_1, "Cutlines", "*"))
cutline_paths_2 = glob.glob(os.path.join(dataset_path_2, "Cutlines", "*"))

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
assert len(cutline_paths_1) == N
assert len(cutline_paths_2) == N
assert len(roi_paths_1) == N
assert len(roi_paths_2) == N

mask_binary_1 = []
mask_binary_2 = []

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

print("Generating Tiles...")

masks_1 = []
masks_2 = []

for i in range(len(core_components_1)):
    padded_size = core_info.iloc[i]["PadSize"]
    target_size = 1024

    for j in range(1, N_components[i] + 1):
        mask_resized_1, _ = resize_mask(core_components_1[i] == j, padded_size=padded_size)
        mask_resized_2, _ = resize_mask(core_components_2[i] == j, padded_size=padded_size)
        mask_resized_2, _, _ = find_rotation(mask_resized_1, mask_resized_2, step=0.1)

        masks_1.append(mask_resized_1)
        masks_2.append(mask_resized_2)
        
        # Generate additional tiles by performing data augmentation
        mask_augmented_pairs = augment_pair(mask_resized_1.astype(np.uint8), mask_resized_2.astype(np.uint8), num_aug=5)
        for (mask_augmented_1, mask_augmented_2) in mask_augmented_pairs:
            masks_1.append(mask_augmented_1)
            masks_2.append(mask_augmented_2)

masks_1 = np.stack(masks_1, axis=0)
masks_2 = np.stack(masks_2, axis=0)

random_seed = 42
batch_size = 4

# Create training & validation split (optional)
N, H, W = masks_1.shape
val_fraction = 0.5
val_n = int(N * val_fraction)

if val_n > 0:
    np.random.seed(random_seed)
    idx = np.random.permutation(N)
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    train_masks_1, train_masks_2 = masks_1[train_idx], masks_2[train_idx]
    val_masks_1, val_masks_2 = masks_1[val_idx], masks_2[val_idx]
else:
    train_masks_1, train_masks_2 = masks_1, masks_2
    val_masks_1 = val_masks_2 = None

# Create TF datasets
def make_dataset(mov_arr, fix_arr, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(((mov_arr, fix_arr), fix_arr))
    ds = ds.shuffle(buffer_size=len(mov_arr), seed=random_seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

tf.keras.backend.clear_session()
train_ds = make_dataset(train_masks_2, train_masks_1, batch_size)
val_ds = make_dataset(val_masks_2, val_masks_1, batch_size) if val_masks_2 is not None else None

epochs = 100
verbose = 1

# Build VoxelMorph model
vxm_model = vxm.networks.VxmDense(
    # Input spatial shape (H, W)
    inshape=(H, W),
    # Choose a UNet architecture: tuple (encoder_list, decoder_list)
    # Keep it small for speed if GPU memory is limited
    nb_unet_features=([32, 64, 64, 64], [64, 64, 64, 32]),
    # 0 produces a direct displacement field; you can experiment with int_steps > 0 for diffeomorphic integration
    int_steps=0
)

# Loss, optimizer, compile
sim_loss = vxm.losses.NCC(win=9).loss
smooth_loss = vxm.losses.Grad('l2').loss

vxm_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=[sim_loss, smooth_loss],
    loss_weights=[1.0, 0.02]
)

# Print model summary to confirm shapes
vxm_model.summary()

callbacks = [
    # Save best model
    tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor='val_loss' if val_ds is not None else 'loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=verbose),
    # Reduce LR on plateau
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss' if val_ds is not None else 'loss',
        factor=0.5, 
        patience=10,
        min_lr=1e-6,
        verbose=verbose),
    # Early stopping
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss' if val_ds is not None else 'loss',
        patience=25, 
        restore_best_weights=True, 
        verbose=verbose)
]

if val_ds is not None:
    history = vxm_model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=verbose)
else:
    history = vxm_model.fit(train_ds, epochs=epochs, callbacks=callbacks, verbose=verbose)