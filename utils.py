from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import openslide
from PIL import Image
from scipy.ndimage import center_of_mass, label, map_coordinates, rotate
from scipy.spatial.distance import cdist
from skimage.filters import threshold_otsu

Image.MAX_IMAGE_PIXELS = None

def read_image(path, height, width):
    """
    Reads a binary image mask.

    Args:
        path (str): File path
        height (int): Height to be resized to
        width (int): Width to be resized to

    Returns:
        NDArray: The binary image mask
    """
    image = np.array(Image.open(path), dtype=np.float32)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    image = image > 0
    return image

def process_tile(x, y, w, h, slide, downsample_factor, tile_size):
    """
    Reads one tile, downsamples it, and returns resized tile + position.

    Args:
        x (int): x location
        y (int): y location
        w (int): Width of the tile
        h (int): Height of the tile
        slide (OpenSlide): OpenSlide slide object
        downsample_factor (_type_): Downsample factor
        tile_size (_type_): Tile size

    Returns:
        tuple: A tuple containing: the tile (x, y) position and the tile image.
    """
    read_w = min(tile_size, w - x)
    read_h = min(tile_size, h - y)

    # Read tile from slide
    region = slide.read_region((x, y), 0, (read_w, read_h)).convert("RGB")

    # Downsample tile immediately
    region_resized = region.resize((read_w // downsample_factor, read_h // downsample_factor), Image.BICUBIC)
    return (x // downsample_factor, y // downsample_factor), region_resized

def find_background_seed(img_gray, thresh=200):
    """
    Returns a seed point (x, y) in the background.

    Args:
        img_gray (NDArray): Grayscaled image with range 0 - 255
        thresh (int, optional): Intensity threshold to consider pixel as background. Defaults to 200.

    Returns:
        tuple: The background seed (x, y) position
    """
    h, w = img_gray.shape

    # Check corners first, skip border pixels to avoid resizing artifacts
    corners = [(1, 1), (1, w - 2), (h - 2, 1), (h - 2, w - 2)]
    for y, x in corners:
        if img_gray[y, x] > thresh:
            return (x, y)

    # If no corners are bright, scan top row or left column
    for x in range(1, w - 1):
        if img_gray[0, x] > thresh:
            return (x, 0)
    for y in range(1, h - 1):
        if img_gray[y, 1] > thresh:
            return (1, y)

    # Fallback: middle pixel
    return (w//2, h//2)

def segmentation(slide_path, roi_path=None, level=2, read_at_level_0=False):
    """_summary_

    Args:
        slide_path (str): The file path of the image slide.
        roi_path (str, optional): The file path to the ROI mask. Defaults to None.
        level (int, optional): The desired output level. Defaults to 2.
        read_at_level_0 (bool, optional): Whether to read and level 0 and perform downsample. Defaults to False.

    Returns:
        tuple: The binary tissue segmentation mask, and the corresponding RGB image
    """
    if slide_path.endswith(".svs") or slide_path.endswith(".tif") or slide_path.endswith(".ndpi"):
        slide = openslide.OpenSlide(slide_path)
        if read_at_level_0:
            w0, h0 = slide.level_dimensions[0]
            downsampling_factor = round(4 ** level)
            w = w0 // downsampling_factor
            h = h0 // downsampling_factor
            result = Image.new("RGB", (w, h))

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for y in range(0, h0, 8192):
                    for x in range(0, w0, 8192):
                        futures.append(executor.submit(process_tile, x, y, w0, h0, slide, downsampling_factor, 8192))
            
                for f in futures:
                    (x_pos, y_pos), tile = f.result()
                    result.paste(tile, (x_pos, y_pos))
        else:
            w, h = slide.level_dimensions[level]
            result = slide.read_region((0, 0), level, (w, h)).convert("RGB")
            result = np.array(result)
    else:
        # If not a slide, read as a regular image
        result = np.array(Image.open(slide_path))
        h0, w0 = result.shape[:2]
        downsampling_factor = round(4 ** level)
        w = w0 // downsampling_factor
        h = h0 // downsampling_factor
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_AREA)
    
    # Perform Otsu segmentation
    downsampled = np.array(result)
    img_gray = cv2.cvtColor(downsampled, cv2.COLOR_RGB2GRAY)

    # Apply ROI if exists
    if roi_path is not None:
        image_roi = read_image(roi_path, h, w)

        # Compute bounding box
        rows = np.where(image_roi.any(axis=1))[0]
        cols = np.where(image_roi.any(axis=0))[0]
        rmin, rmax = rows[0], rows[-1]
        cmin, cmax = cols[0], cols[-1]
        thresh_val = threshold_otsu(img_gray[rmin:rmax+1, cmin:cmax+1])
    else:
        thresh_val = threshold_otsu(img_gray)
        
    mask_binary = (img_gray > thresh_val).astype(np.uint8)

    # Morphological cleanup and filling holes
    kernel = np.ones((5, 5), np.uint8)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

    # Apply ROI if exists
    if roi_path is not None:
        mask_binary = (((mask_binary == 0) * image_roi) == 0).astype(np.uint8)

    mask_flood_fill = mask_binary.copy()
    target_h, target_w = mask_binary.shape
    mask = np.zeros((target_h + 2, target_w + 2), np.uint8)
    seed_point = find_background_seed(img_gray, thresh_val)
    cv2.floodFill(mask_flood_fill, mask, seed_point, 255)
    mask_binary = cv2.bitwise_or(mask_binary, cv2.bitwise_not(mask_flood_fill))

    # Floodfill the top-left corner in case there are artifacts there as well...
    if mask_binary[0, 0] == 255:
        cv2.floodFill(mask_binary, mask, [0, 0], 0)
    
    # Invert + normalize to 0/1
    mask_binary = cv2.bitwise_not(mask_binary)
    mask_binary = (mask_binary == 0).astype(np.uint8)

    return mask_binary, downsampled

def find_splits(line_binary, gap_size):
    """
    Finds the centroid of each line

    Args:
        line_binary (NDArray): The binary mask of the lines
        gap_size (int): The margin size

    Returns:
        list: The list of centroids
    """
    centroids = []

    start = -1
    end = -1
    in_shape = False
    for i in range(len(line_binary)):
        if line_binary[i]:
            if start == -1 and not in_shape:
                start = i
            in_shape = True
            end = i
        else:
            if not in_shape and start != -1 and i - end > gap_size:
                    centroids.append((end + start) // 2)
                    start = -1
                    end = -1
            in_shape = False

    if start != -1:
        centroids.append((end + start) // 2)
    
    return centroids

def apply_splits(mask_binary, split_path, is_vertical=True, gap_size=50):
    """
    Applies the cutline to a mask

    Args:
        mask_binary (NDArray): The binary mask
        split_path (str): The file path to the cutline mask
        is_vertical (bool, optional): Whether the mask should be split vertically. Defaults to True.
        gap_size (int, optional): The margin size. Defaults to 50.

    Returns:
        tuple: The mask labelled with each component from 1 to N, with 0 being the background, as well as the number N
    """
    H, W = mask_binary.shape
    image_splits = read_image(split_path, H, W)
    splits_binary = mask_binary * image_splits
    mask_components = mask_binary.copy()
    prev = 0
    if is_vertical:
        h_splits = np.any(splits_binary, axis=1)
        h_centroids = find_splits(h_splits, gap_size)
        for i, c in enumerate(h_centroids):
            c = round(c)
            mask_components[prev:c, :] = mask_binary[prev:c, :] * (i + 1)
            prev = c
        mask_components[prev:, :] = mask_binary[prev:, :] * (len(h_centroids) + 1)
        return mask_components, len(h_centroids) + 1
    else:
        v_splits = np.any(splits_binary, axis=0)
        v_centroids = find_splits(v_splits, gap_size)
        for i, c in enumerate(v_centroids):
            c = round(c)
            mask_components[:, prev:c] = mask_binary[:, prev:c] * (i + 1)
            prev = c
        mask_components[:, prev:] = mask_binary[:, prev:] * (len(v_centroids) + 1)
        return mask_components, len(v_centroids) + 1

def resize_mask(mask_binary, landmark=None, margin=10, padded_size=1024, target_size=1024):
    """
    Creates a tile based on the binary mask

    Args:
        mask_binary (NDArray): The binary mask
        landmark (tuple | NDArray, optional): The landmark (either in tuple of polygons or mask). Defaults to None.
        margin (int, optional): The margin. Defaults to 10.
        padded_size (int, optional): The padded size before resizing. Defaults to 1024.
        target_size (int, optional): The target tile size after resizing. Defaults to 1024.

    Returns:
        tuple: The tile, the landmark (as a tuple of shifted polygons or tiled mask, if applicable), and the (y, x) shift
    """
    H, W = mask_binary.shape
    ys, xs = np.where(mask_binary)
    y_cen = np.average(ys)
    x_cen = np.average(xs)
    y_mar = max(y_cen - ys.min(), ys.max() - y_cen)
    x_mar = max(x_cen - xs.min(), xs.max() - x_cen)
    y_min = round(y_cen - y_mar - margin)
    y_max = round(y_cen + y_mar + margin + 1)
    x_min = round(x_cen - x_mar - margin)
    x_max = round(x_cen + x_mar + margin + 1)
    
    mask_cropped = mask_binary[y_min:y_max, x_min:x_max]

    H, W = mask_cropped.shape
    pad_h = padded_size - H
    pad_w = padded_size - W
    
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Cropped image is larger than target size")
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    mask_padded = np.pad(mask_cropped.astype(np.float32), ((pad_top, pad_bottom), (pad_left, pad_right)))
    mask_padded = cv2.resize(mask_padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
    mask_padded = mask_padded > 0
    if landmark is not None:
        if isinstance(landmark, tuple):
            landmark_shift = np.array([[[x_min - pad_left + padded_size // 2, y_min - pad_top + padded_size // 2]]])
            landmark_padded = [(lm - landmark_shift) * target_size / padded_size for lm in landmark]
            return mask_padded, landmark_padded, (y_min - pad_top, x_min - pad_left)
        elif isinstance(landmark, np.ndarray):
            landmark_cropped = landmark[y_min:y_max, x_min:x_max]
            landmark_padded = np.pad(landmark_cropped.astype(np.float32), ((pad_top, pad_bottom), (pad_left, pad_right)))
            landmark_padded = cv2.resize(landmark_padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
            landmark_padded = landmark_padded > 0
            return mask_padded, landmark_padded, (y_min - pad_top, x_min - pad_left)
    return mask_padded, (y_min - pad_top, x_min - pad_left)

def dice_coefficient(mask1, mask2, eps=1e-6):
    """
    Compute Dice coefficient between two binary masks.

    Args:
        mask1 (NDArray): Mask 1
        mask2 (NDArray): Mask 2
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        float: The Dice coefficient
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    dice = (2. * intersection) / (mask1.sum() + mask2.sum() + eps)
    return dice

def find_rotation(fix, move, angle_range=(-30, 30), step=0.5):
    """
    Finds the optimal rotation angle between two masks

    Args:
        fix (NDArray): Mask 1
        move (NDArray): Mask 2
        angle_range (tuple, optional): The range of angle in degrees. Defaults to (-30, 30).
        step (float, optional): The step in degrees. Defaults to 0.5.

    Returns:
        tuple: The aligned mask 2, the best angle, and the best Dice coefficient
    """
    best_dice = -1
    best_angle = 0
    best_aligned = None

    for angle in np.arange(angle_range[0], angle_range[1] + step, step):
        rotated = rotate(move, angle, reshape=False, order=0)

        dice = dice_coefficient(fix, rotated)

        if dice > best_dice:
            best_dice = dice
            best_angle = angle
            best_aligned = rotated

    return best_aligned, best_angle, best_dice

def rotate_coordinates(coords, degrees):
    """
    Rotates a set of coordinates based on the given angle

    Args:
        coords (NDArray): The coordinates to rotate in N x 1 x 2
        degrees (float): The angle in degrees

    Returns:
        NDArray: The rotated coordinates in N x 1 x 2
    """
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    N = coords.shape[0]
    points = coords.reshape(N, 2)
    rotated_points = points.dot(R.T)
    return rotated_points.reshape(N, 1, 2)

def augment_pair(fix, move, num_aug=5,
                 max_rotate=10,
                 max_shift=20,
                 max_scale=0.1,
                 allow_flip=False):
    """
    Generate augmented image pairs based on a ground-truth image pair.
    
    Args:
        fix (NDArray): Mask 1
        move (NDArray): Mask 2
        num_aug (int, optional): Number of augmented samples to generate. Defaults to 5.
        max_rotate (int, optional): The maximum rotation in degrees. Defaults to 10.
        max_scale (float, optional): The maximum scale. Defaults to 0.1.
        allow_flip (bool, optional): Whether to allow flip. Defaults to False.

    Returns:
        list: A list of augmented image pairs
    """
    H, W = fix.shape
    out = []

    for _ in range(num_aug):
        # Random Transform Parameters
        angle = np.random.uniform(-max_rotate, max_rotate)
        tx    = np.random.uniform(-max_shift, max_shift)
        ty    = np.random.uniform(-max_shift, max_shift)
        scale = 1.0 + np.random.uniform(-max_scale, max_scale)

        # Optional flip
        do_flip = allow_flip and np.random.rand() < 0.5

        # Build affine transform
        M = cv2.getRotationMatrix2D((W/2, H/2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        # Apply transform to both masks
        fix_aug  = cv2.warpAffine(fix,  M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        move_aug = cv2.warpAffine(move, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Flip identically
        if do_flip:
            fix_aug  = np.fliplr(fix_aug)
            move_aug = np.fliplr(move_aug)

        # Keep masks binary (important!)
        fix_aug  = (fix_aug  > 0.5).astype(np.float32)
        move_aug = (move_aug > 0.5).astype(np.float32)

        out.append((fix_aug, move_aug))

    return out

def prep(img):
    """
    Ensure shape -> (1, H, W, 1) and dtype float32, scaled to [0,1].

    Args:
        img (NDArray): Input

    Returns:
        NDArray: Output
    """
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = img[..., np.newaxis]  # (256,256,1)
    img = np.clip(img, 0.0, 1.0)
    img = img[np.newaxis, ...]      # (1,256,256,1)
    return img

def warp(binary_mask, flow):
    """
    Warp a 2D mask using a 2D flow field

    Args:
        binary_mask (NDArray): The mask
        flow (NDArray): The flow field in H x W x 2

    Returns:
        NDArray: The warped mask
    """
    H, W = binary_mask.shape
    S = flow.shape[0]

    rows, cols = np.nonzero(binary_mask)
    rows = rows.astype(np.float32)
    cols = cols.astype(np.float32)

    rows = rows / H * S
    cols = cols / H * S

    # Sample flow at landmark positions
    rows += map_coordinates(flow[..., 0], [rows, cols], order=1, mode='nearest')
    cols += map_coordinates(flow[..., 1], [rows, cols], order=1, mode='nearest')

    rows = np.round(rows / S * H).astype(int)
    cols = np.round(cols / S * H).astype(int)

    # Output
    lm_warped = np.zeros_like(binary_mask, dtype=np.uint8)

    for r, c in zip(rows, cols):
        if 0 <= r < H and 0 <= c < W:
            lm_warped[r, c] = 1

    return lm_warped

def warp_coordinates(coordinates, flow):
    """
    Warp a set of coordinates using a 2D flow field

    Args:
        coordinates (NDArray): The coordinates to be warped in N x 1 x 2
        flow (NDArray): The flow field in H x W x 2

    Returns:
        NDArray: The warped coordinates
    """
    rows = np.array(coordinates[:, 0, 1])
    cols = np.array(coordinates[:, 0, 0])
    rows += map_coordinates(flow[..., 0], [rows, cols], order=1, mode='nearest')
    cols += map_coordinates(flow[..., 1], [rows, cols], order=1, mode='nearest')
    return np.stack([cols,  rows], axis=-1)[:, None]

def compute_tre(f_mask, m_mask):
    """
    Computes Target Registration Error (TRE) between the two inputs.

    Args:
        f_mask (tuple | NDArray): Landmark 1, can be mask or polygons
        m_mask (tuple | NDArray): Landmark 2, can be mask or polygons

    Returns:
        NDArray: The TRE per landmark
    """
    if isinstance(f_mask, tuple):
        lm_fixed = contour_to_landmark_centroids(f_mask)
        lm_moving = contour_to_landmark_centroids(m_mask)
    else:
        lm_fixed = mask_to_landmark_centroids(f_mask)
        lm_moving = mask_to_landmark_centroids(m_mask)

    n_fixed = lm_fixed.shape[0]
    n_moving = lm_moving.shape[0]

    # If no landmarks at all → return empty
    if n_fixed == 0 or n_moving == 0:
        return np.array([], dtype=np.float32)

    lm_fixed = lm_fixed.astype(np.float32)
    lm_moving = lm_moving.astype(np.float32)

    # Distance matrix: (N_fixed, N_moving)
    dist_matrix = cdist(lm_fixed, lm_moving)

    # For each fixed LM, find closest moving LM
    nearest_moving_idx = np.argmin(dist_matrix, axis=1)
    tre_per_point = dist_matrix[np.arange(n_fixed), nearest_moving_idx].astype(np.float32)
    return tre_per_point

def mask_to_landmark_centroids(lm_mask):
    """
    Calculate centroids based on a binary landmark mask

    Args:
        lm_mask (NDArray): Binary landmark mask

    Returns:
        NDArray: List of centroids
    """
    labeled_mask, num = label(lm_mask)

    if num == 0:
        return np.empty((0, 2))

    centers = center_of_mass(
        lm_mask,
        labeled_mask,
        range(1, num + 1)
    )

    return np.array(centers)

def contour_to_landmark_centroids(lm_coords):
    """
    Calculate centroids based on a list of landmark polygons

    Args:
        lm_mask (tuple): Landmark polygons

    Returns:
        NDArray: List of centroids
    """
    centroids = []
    for lm_coord in lm_coords:
        M = cv2.moments(lm_coord.astype(np.float32))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
    return np.array(centroids)