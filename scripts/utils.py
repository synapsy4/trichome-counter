"""
Utility functions.
"""

import os
from pathlib import Path
import json
import scipy.io as sio
import argparse

import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from . import models

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


def save_model(model,
               model_name,
               metadata,
               best_cp,
               target_dir="models"):
    """
    Saves a PyTorch model and its metadata.

    Parameters
    ----------
        model : torch.nn.Module
            A target PyTorch model to save.
        model_name : str
            Model name. If dir for model name already exists, model is saved 
            in existing dir. If not, a new dir is created.
        metadata : dict
            Dictionary with hyperparameter and results dict.
        best_cp : OrderedDict
            State dict of model at epoch with lowest validation mae.
        target_dir : str or pathlib.Path, optional
            Root directory where models are stored.
            Default is "models".
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                            exist_ok=True)

    # Create model save path
    if model_name.endswith(".pth") or model_name.endswith(".pt"):
        model_name = model_name.split(".")[0]
    model_dir_path = target_dir_path / Path(model_name)
    model_dir_path.mkdir(parents=True,
                            exist_ok=True)
    
    # Create model instance save path
    model_list = os.listdir(model_dir_path)
    model_idx = 1
    model_instance_id = f"training_run_{model_idx:02d}"
    while model_instance_id in model_list:
        model_idx += 1
        model_instance_id = f"training_run_{model_idx:02d}"
    model_instance_path = model_dir_path / Path(model_instance_id)
    model_instance_path.mkdir(parents=True,
                            exist_ok=True)

    # Save the model state_dict()
    model_save_path = model_instance_path / "model.pth"
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
                f=model_save_path)
    
    # Save metadata (hyperparams and results)
    metadata_save_path = model_instance_path / "metadata.json"
    print(f"[INFO] Saving metadata to: {metadata_save_path}")
    with open(metadata_save_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Save best cp
    cp_save_path = model_instance_path / "best_cp.pth"
    print(f"[INFO] Saving best cp to: {cp_save_path}")
    torch.save(obj=best_cp,
                f=cp_save_path)
    

def load_model(model_name,
               run_id=None,
               cp="last",
               target_dir="models"):
    """
    Load a trained model instance and its corresponding checkpoint from disk.

    Parameters
    ----------
    model_name : str
        Name of the model directory inside `target_dir`.
    run_id : int or None, optional
        Identifier of the training run to load (e.g., 1 -> "training_run_01").
        If None, the function loads the most recent training run.
        Default is None.
    cp : {"last", "best"}, optional
        Specifies which checkpoint to load:
        - "last": Loads the final saved model weights ("model.pth").
        - "best": Loads the best validation checkpoint ("best_cp.pth").
        Default is "last".
    target_dir : str or pathlib.Path, optional
        Root directory where models are stored.
        Default is "models".

    Returns
    -------
    model : torch.nn.Module
        Instantiated model with loaded state dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified model directory or training run does not exist.
    KeyError
        If `cp` is not one of {"last", "best"}.
    TypeError
        If the model type specified in metadata is not implemented.
    
    Notes
    -----
    - Expects the following directory structure:
        target_dir/
            model_name/
                training_run_XX/
                    model.pth
                    best_cp.pth
                    metadata.json
    """
    
    model_dir_path = Path(target_dir) / model_name

    if not model_dir_path.exists():
        raise FileNotFoundError(f"No model path found matching the given path '{model_dir_path}'")
    
    # If not otherwise specifiec load model from last training run
    if run_id is None:
        model_list = os.listdir(model_dir_path)
        model_idx = 1
        model_instance_id = f"training_run_{model_idx:02d}"
        while model_instance_id in model_list:
            model_idx += 1
            model_instance_id = f"training_run_{model_idx:02d}"
        model_instance_path = model_dir_path / Path(model_instance_id)
    # Else load model with given run id
    else:
        model_instance_id = f"training_run_{run_id:02d}"
        model_instance_path = model_dir_path / Path(model_instance_id)
        if not model_instance_path.exists():
            raise FileNotFoundError(f"No model path found for the given run id, i.e. under path '{model_instance_path}'")
        
    # Decide if last or best checkpoint should be used
    if cp == "last":
        model_path = model_instance_path / "model.pth"
    elif cp == "best":
        model_path = model_instance_path / "best_cp.pth"
    else:
        raise KeyError(f"Model cp must be one of 'last' or 'best'")

    # Load metadata
    json_path = model_instance_path / "metadata.json"
    with open(json_path, "r") as f:
        metadata = json.load(f)

    # Return model if model type matches an implemented model type
    model_type = metadata["model_type"]
    if model_type == "density_model":
        model = models.DensityModel()
        model.load_state_dict(torch.load(model_path))
        return model
    else:
        raise TypeError(f"Model type {model_type} unknown. Update of load_model function required.")
    
def init_model(model_name, model_type):
    
    # Check if model under name already exists
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    models_dir = os.path.join(parent_dir, "models")
    models = os.listdir(models_dir)

    if model_name in models:
        user_in = input("Model with model name alrady exists.\nContinue training (y) existing model or exit (n)?\n [y/n]:")

        while user_in not in ["y","n"]:
            user_in = input("Valid input: {y,n} ({Continue training, exit})\n")
        
        if user_in == "n":
            raise InterruptedError("Exit chosen by user.")
        else:
            # Load model
            model = load_model(model_name)
    else:
        if model_type == "density_model":
            model = models.DensityModel()

def parse_train_args():
    
    # Creating a parser
    parser = argparse.ArgumentParser(description="Train model")
    
    # Add parser arguments
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs")

    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for train/val/test")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")

    parser.add_argument("--model-name", type=str, default="model0",
                        help="Saved model filename")
    
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                    help="Weight decay")
    
    parser.add_argument("--short-side", type=int, default=512,
                        help="Short side len of transformed image")
    
    parser.add_argument("--sigma", type=float, default=1,
                        help="Target density map standard deviation")
    
    parser.add_argument("--lbda-count", type=float, default=0.5,
                        help="Count loss weight")
    
    

    

    return parser.parse_args()