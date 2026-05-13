"""
Functions to preprocess raw data
"""

from pathlib import Path
import random
from typing import Callable, Any

import numpy as np
import cv2
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def split_filenames(
        raw_root: Path,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        stratify: bool = True,
        seed: int = 42
        ) -> dict:
    """
    Split file IDs into train/val/test sets with random shuffling.

    Parameters
    ----------
    raw_root : Path
        Directory with raw TIFF images and coordinate .mat files.
    train_ratio : float, optional
        Proportion for training set (default: 0.7).
    val_ratio : float, optional
        Proportion for validation set (default: 0.15).
    stratify : bool, optional
        Splits with balanced data distributions (default: True).
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    dict
        Dictionary with keys 'train', 'val', 'test' containing lists of file IDs.
    """ 
    # Setup random number generator
    rng = random.Random(seed)

    # Get all data ids
    data_ids = np.array(sorted(p.stem for p in raw_root.glob("*.tiff")))
    
    # Balance splits by trichome counts
    if stratify:
        # Init array for all trichome counts
        coord_lens = np.zeros(len(data_ids))
        
        # Read out number of trichomes per data instance
        for i,data_id in enumerate(data_ids):
            coord_path = raw_root / f"{data_id}_coords.mat"
            coord_data = sio.loadmat(coord_path)
            n_coords = len(coord_data["coords"])
            coord_lens[i] = n_coords

        # Calculate bin edges
        n_bins = 5
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(coord_lens, quantiles)
        bin_edges[-1] += 1e-8 # open last bin

        # Get bin index of each data instance
        bin_indices = np.digitize(coord_lens, bin_edges[1:-1], right=True)

        # Init id lists for all splits
        train_ids = []
        val_ids = []
        test_ids = []

        # Fill split id lists with binned (balanced) data ids
        for b in range(n_bins):
            # Get data indices for current bin + shuffle them
            indices = np.where(bin_indices == b)[0]
            indices = list(indices)
            rng.shuffle(indices)

            # Get number of (train, val) data instances for current bin
            n = len(indices)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            # Extend split id lists
            train_ids.extend(data_ids[indices[:n_train]])
            val_ids.extend(data_ids[indices[n_train:n_train+n_val]])
            test_ids.extend(data_ids[indices[n_train+n_val:]])

        # Sort balannced ids into split-categories
        data_id_splits = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids
        }

    # Do not balance data splits by trichome counts
    else:
        # Random shuffle ids
        data_ids = list(data_ids)
        rng.shuffle(data_ids)

        # Get lengths of train and val data
        n = len(data_ids)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        # Sort ids into split-categories
        data_id_splits = {
            "train": data_ids[:n_train],
            "val": data_ids[n_train:n_train+n_val],
            "test": data_ids[n_train+n_val:]
        }

    return data_id_splits


def process_single_image(
        data_path: Path,
        data_id: str
        ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Extract ROI from image and adjust trichome coordinates to ROI bounds.

    Parameters
    ----------
    data_path : Path
        Directory containing image and coordinate files.
    data_id : str
        Base filename (without extension).

    Returns
    -------
    new_img : numpy.ndarray
        Cropped image containing only the ROI.
    coords : numpy.ndarray
        Nx2 array of adjusted trichome coordinates.
    roi_size : tuple of int
        ROI dimensions as (width, height).

    Raises
    ------
    RuntimeError
        If ROI exceeds outer image boundaries.
    RuntimeError
        If coordinates fall outside ROI boundaries.
    """
    # Open coordinate data
    coord_path = data_path / f"{data_id}_coords.mat"
    coord_data = sio.loadmat(coord_path)

    # Get ROI
    rectPos = coord_data["rectPos"]
    rect_x, rect_y, rect_w, rect_h = rectPos[0]
    # Account for matlab index shift
    rect_x -= 1
    rect_y -= 1
    # Discretize ROI such that it does not shrink and Ws and Hs stay equal across ROIs
    rect_x = int(max(np.floor(rect_x),0))
    rect_y = int(max(np.floor(rect_y),0))
    rect_w = int(np.ceil(rect_w) + 1) # +1 to make sure that flooring lower bounds is no greater shift than ceiling dimensions
    rect_h = int(np.ceil(rect_h) + 1)

    # Open img
    img_path = data_path / f"{data_id}.tiff"
    img = cv2.imread(str(img_path))

    # Handle case that ROI is on the edge of the image H,W
    H, W = img.shape[:2]
    if rect_y+rect_h > H or rect_x+rect_w > W:
        diff_x, diff_y = 0,0
        if rect_y+rect_h > H:
            diff_y = (rect_y+rect_h) - H
        if rect_x+rect_w > W:
            diff_x = (rect_x+rect_w) - W
        diff_x = (rect_y+rect_h) - H  
        # Case that ROI exceeds image edges unexpectedly much: Error (Should never be the case, but just to be sure)
        if diff_x > 1 or diff_y > 1:
            raise RuntimeError(f"ROI does not behave as expected. Check ROI on image {data_id} and correct manually.")
        # Case that ROI exceeds image edges slightly: Correct lower image corner slightly
        else:
            rect_y -= 1
            rect_x -= 1
    
    # Cutout ROI
    new_img = img[rect_y:rect_y+rect_h,
                    rect_x:rect_x+rect_w,
                    :]
 
    # Get trichome coordinate vector 
    coords = coord_data["coords"].astype(np.float32)
    # If coordinates are not empty (leaf with trichomes), process them
    if len(coords > 0):
        # Shift coordinate data by matlab indexing missmatch
        coords -= 1
        # Shift coordinate data by ROI lower bounds
        coords[:,0] -= rect_x
        coords[:,1] -= rect_y

        # Check if all coordinates are within image bounds 
        if (np.any(coords[:, 0] < 0) or 
            np.any(coords[:, 0] >= rect_w) or 
            np.any(coords[:, 1] < 0) or 
            np.any(coords[:, 1] >= rect_h)):
            raise RuntimeError(
                f"File {data_id} has coordinates outside its ROI."
            )
    
    return new_img, coords, (rect_w, rect_h)


def preprocess_dataset(raw_root: Path, 
                       out_root: Path
                       ) -> None:
    """
    Process dataset by splitting files and extracting ROIs with coordinates.

    Creates train/val/test splits and saves processed images as JPEGs and 
    coordinates as .npy files. Validates ROI size consistency across dataset.

    Parameters
    ----------
    raw_root : Path
        Directory with raw TIFF images and coordinate .mat files.
    out_root : Path
        Output directory for processed data.

    Raises
    ------
    FileExistsError
        If output directories already exist.
    RuntimeError
        If ROI sizes are inconsistent.
    """  
    # Get all file ids and define splits
    splits = split_filenames(raw_root)

    last_roi_size = None

    for split, ids in splits.items():
        print(f"Processing {split} data...")

        # Create image and point split directories
        img_out = out_root / split / "images"
        coords_out = out_root / split / "coords"

        if img_out.exists() or coords_out.exists():
            raise FileExistsError("Preprocessing paths already exist. Delete manually if preprocessing should be redone.")

        img_out.mkdir(parents=True, exist_ok=True)
        coords_out.mkdir(parents=True, exist_ok=True)

        for fid in tqdm(ids):
            img, coords, roi_size = process_single_image(raw_root,fid)

            # Check for unequal ROI sizes across images
            if last_roi_size is not None and roi_size != last_roi_size:
                raise RuntimeError("Inconsistent ROI sizes detected.")
            last_roi_size = roi_size

            # Save data
            cv2.imwrite(str(img_out / f"{fid}.jpg"), img)
            np.save(coords_out / f"{fid}.npy", coords)
