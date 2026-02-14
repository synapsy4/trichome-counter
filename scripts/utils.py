"""
Utility functions.
"""

from pathlib import Path
import random
import numpy as np
import cv2
import scipy.io as sio
import torch
import matplotlib.pyplot as plt

def get_random_data_paths(seed=None):
    """
    Get random file paths for raw and preprocessed image and coordinate data.

    Randomly selects a data split (train/val/test) and a file from that split,
    then returns paths to both the raw and preprocessed versions of the data.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    image_path_raw : Path
        Path to the raw TIFF image file.
    coord_path_raw : Path
        Path to the raw coordinate .mat file.
    image_path_pre : Path
        Path to the preprocessed JPEG image file.
    coord_path_pre : Path
        Path to the preprocessed coordinate .npy file.
    """
    # Use seed if given
    if seed is not None:
        random.seed(seed)

    # Choose processed image and coord path randomly
    data_paths = [Path("data/preprocessed/train"),
                  Path("data/preprocessed/val"),
                  Path("data/preprocessed/test")]
    data_path = random.choice(data_paths) 
    img_path = data_path / "images"
    coord_path = data_path / "coords"

    # List all image ids in path and choose a id randomly
    file_ids = [p.stem for p in img_path.glob("*.jpg")]
    file_id = random.choice(file_ids)

    # Select paths to raw and processed data based on file id
    image_path_raw = Path("data/raw") / (file_id + ".tiff")
    coord_path_raw = Path("data/raw") / (file_id + "_coords.mat")
    image_path_pre = img_path / (file_id + ".jpg")
    coord_path_pre = coord_path / (file_id + ".npy")

    return image_path_raw, coord_path_raw, image_path_pre, coord_path_pre



def load_raw_image_data(image_path_raw,coord_path_raw):
    """
    Load raw image and coordinate data from file paths.
    
    Parameters
    ----------
    image_path_raw : str or Path
        Path to the raw image file.
    coord_path_raw : str or Path
        Path to the .mat file containing coordinate and ROI data.
    
    Returns
    -------
    img_raw : numpy.ndarray
        Raw image in RGB format (H, W, 3).
    rect_x : float
        X-coordinate of the top-left corner of the rectangular ROI.
    rect_y : float
        Y-coordinate of the top-left corner of the rectangular ROI.
    rect_w : float
        Width of the rectangular ROI.
    rect_h : float
        Height of the rectangular ROI.
    coords_raw : numpy.ndarray
        Array of image (x,y)-coordinates with shape (N, 2).
    """
    # Load image and convert to RGB
    img_raw = cv2.imread(str(image_path_raw))
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    # Load coordinate and ROI data
    coord_data_raw = sio.loadmat(coord_path_raw)

    # Read out coordinate data and add index shift
    coords_raw = coord_data_raw["coords"]  
    if len(coords_raw) > 0:
        coords_raw -= 1 # Indexing from matlab format to python format

    # Read out ROI data and add index shift
    rectPos = coord_data_raw["rectPos"]
    rect_x, rect_y, rect_w, rect_h = rectPos[0]
    rect_x -= 1 # Indexing from matlab format to python format
    rect_y -= 1

    return img_raw, rect_x, rect_y, rect_w, rect_h, coords_raw



def load_preprocessed_image_data(image_path_pre,coord_path_pre):
    """
    Load preprocessed image and coordinate data from file paths.
    
    Parameters
    ----------
    image_path_pre : str or Path
        Path to the preprocessed image file.
    coord_path_pre : str or Path
        Path to the .npy file containing preprocessed coordinate data.
    
    Returns
    -------
    img_pre : numpy.ndarray
        Preprocessed image in RGB format (H, W, 3).
    coords_pre : numpy.ndarray
        Array of image (x,y)-coordinates with shape (N, 2).
    """
    # Load image and convert to RGB
    img_pre = cv2.imread(str(image_path_pre))
    img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)

    # Load coordinate data
    coords_pre = np.load(coord_path_pre)
    
    return img_pre, coords_pre


def plot_data_instance(img, coords, title="", ax=None):
    """
    Plot an image with overlaid coordinate points.
    
    Parameters
    ----------
    img : numpy.ndarray or torch.Tensor
        Image to display with shape (H, W, 3) in RGB format.
    coords : numpy.ndarray or torch.Tensor
        Array of (x, y)-coordinates with shape (N, 2). Can be empty.
    title : str, optional
        Title to display above the plot. Default is empty string.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, creates a new figure with
        size (10, 5). Default is None.
    
    Returns
    -------
    None
        Displays the plot on the provided or created axes.
    """
    # If no axes provided, create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,5))
    # Plot image
    ax.imshow(img)
    # Plot coordinates if given
    if len(coords) > 0:
        ax.scatter(coords[:,0], coords[:,1], s=10, c="red", marker="o")
    # Set title and turn off axis
    ax.set_title(title)
    ax.axis("off")


def collate_fn(batch):
    """
    Custom collate function for batching dataset samples.
    
    Stacks images and target maps into batched tensors while keeping coordinates
    as a list of tensors (since each sample has a different number of coordinates).
    
    Parameters
    ----------
    batch : list of tuple
        List of dataset samples, where each sample is a tuple of
        (image, target_map, coords).
    
    Returns
    -------
    images : torch.Tensor
        Batched images with shape (B, 3, H, W).
    target_maps : torch.Tensor
        Batched target maps with shape (B, H, W).
    coords : list of torch.Tensor
        List of coordinate tensors, one per sample. Each tensor has shape
        (N_i, 2) where N_i is the number of coordinates for sample i.
    """
    # Unzip batch into separate lists of images, target maps, and coordinates
    images, target_maps, coords = zip(*batch)
    
    # Stack images and target maps into batched tensors along batch dimension
    images = torch.stack(images, dim=0)
    target_maps = torch.stack(target_maps, dim=0)

    return images, target_maps, coords  # coords = list of tensors