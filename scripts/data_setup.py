"""
Functions to preprocess raw data, and class for creating custom PyTorch dataset.
"""

from pathlib import Path
import scipy.io as sio
import random
import numpy as np
import cv2
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset


def split_filenames(
        file_ids,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
        ):
    """
    Split file IDs into train/val/test sets with random shuffling.

    Parameters
    ----------
    file_ids : list of str
        List of file identifier strings.
    train_ratio : float, optional
        Proportion for training set (default: 0.7).
    val_ratio : float, optional
        Proportion for validation set (default: 0.15).
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    dict
        Dictionary with keys 'train', 'val', 'test' containing lists of file IDs.
    """ 
    # Random shuffle ids
    rng = random.Random(seed)
    file_ids = list(file_ids)
    rng.shuffle(file_ids)

    # Get lengths of train and val data
    n = len(file_ids)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    # Sort shuffled ids into split-categories
    file_id_splits = {
        "train": file_ids[:n_train],
        "val": file_ids[n_train:n_train+n_val],
        "test": file_ids[n_train+n_val:]
    }

    return file_id_splits


def process_single_image(
        data_path: Path,
        data_id: str
        ):
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


def preprocess_dataset(raw_root: Path, out_root: Path):
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
    file_ids = sorted(p.stem for p in raw_root.glob("*.tiff"))
    splits = split_filenames(file_ids)

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


class TrichomeDataset(Dataset):
    """
    PyTorch Dataset for trichome detection with customizable target map generation.

    Loads images and point coordinates, applies transformations, and generates
    target maps (e.g., density maps, heatmaps) using a provided target map function.

    Parameters
    ----------
    root : str or Path
        Root directory containing 'images/' and 'coords/' subdirectories.
    target_map_fun : callable
        Function to generate target maps from coordinates. Must have signature
        target_map_fun(coords, H, W, *args, **kwargs) and return a tensor.
    transform : callable, optional
        Transform to apply to images and coordinates (default: None).
    *target_map_args
        Positional arguments to pass to target_map_fun (e.g., sigma for Gaussian
        density maps).
    **target_map_kwargs
        Keyword arguments to pass to target_map_fun.

    Attributes
    ----------
    images : list of Path
        Sorted list of image file paths.

    Methods
    -------
    __len__()
        Return number of samples in dataset.
    __getitem__(idx)
        Get image, target map, and coordinates for given index.
    """
    def __init__(self, root, target_map_fun, transform=None, *target_map_args, **target_map_kwargs):
        self.root = Path(root)
        self.transform = transform
        self.target_map_fun = target_map_fun
        self.target_map_args = target_map_args
        self.target_map_kwargs = target_map_kwargs
        self.images = sorted((self.root / "images").glob("*.jpg"))

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        img : torch.Tensor
            RGB image array, normalized to [0, 1].
        target_map : torch.Tensor
            Generated target map from the provided target_map_fun.
        coords : torch.Tensor
            Nx2 tensor of trichome (x,y)-coordinates.
        """
        # Construct file paths
        img_path = self.images[idx]
        coord_path = self.root / "coords" / f"{img_path.stem}.npy"

        # Load image and convert to RGB + load coordinates + transform to tensors
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        coords = np.load(coord_path)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        coords = torch.from_numpy(coords).float()

        # Apply transforms if provided
        if self.transform:
            img, coords = self.transform(img, coords)

        # Generate target map from coordinates
        _, H, W = img.shape
        target_map = self.target_map_fun(coords, H, W, *self.target_map_args, **self.target_map_kwargs)

        return img, target_map, coords
