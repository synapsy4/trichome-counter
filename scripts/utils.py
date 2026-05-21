"""
Utility functions.
"""

import os
import random
from pathlib import Path
import json
import yaml
import argparse
from typing import Any
from collections import OrderedDict
from collections.abc import Iterator

import numpy as np
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio

from scripts import models
from scripts import loss


def get_random_data_paths(seed: int = None
                          ) -> tuple[Path, Path, Path, Path]:
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



def load_raw_image_data(image_path_raw: str | Path,
                        coord_path_raw: str | Path
                        ) -> tuple[np.ndarray, float, float, float, float, np.ndarray]:
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



def load_preprocessed_image_data(image_path_pre: str | Path,
                                 coord_path_pre: str | Path
                                 ) -> tuple[np.ndarray, np.ndarray]:
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


def plot_data_instance(img: np.ndarray | torch.Tensor, 
                       coords: np.ndarray | torch.Tensor, 
                       title: str = "", 
                       ax: matplotlib.axes.Axes = None
                       ) -> None:
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
        _, ax = plt.subplots(figsize=(10,5))
    # Plot image
    ax.imshow(img)
    # Plot coordinates if given
    if len(coords) > 0:
        ax.scatter(coords[:,0], coords[:,1], s=10, c="red", marker="o")
    # Set title and turn off axis
    ax.set_title(title)
    ax.axis("off")


def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
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


def save_model(last_cp: OrderedDict[str, torch.Tensor],
               optim_cp: OrderedDict[str, torch.Tensor],
               best_cp: OrderedDict[str, torch.Tensor],
               cfg: dict[str, Any], 
               metrics: dict[str, Any],
               ) -> None:
    """
    Saves a PyTorch model and its metadata.

    Parameters
    ----------
        last_cp : OrderedDict
            State dict of model from last epoch.
        optim_cp : OrderedDict
            State dict of optimizer from last epoch.
        best_cp : OrderedDict
            State dict of model at epoch with lowest validation mae.
        cfg : dict
            Config dict.
        metircs : dict
            Dictionary with results.
    """
    # Create target directory
    target_dir_path = Path(cfg["paths"]["models"])
    target_dir_path.mkdir(parents=True,
                            exist_ok=True)

    # Create model save path
    model_name = cfg["model"]["model_name"]
    model_dir_path = target_dir_path / Path(model_name)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create model instance save path
    model_list = os.listdir(model_dir_path)
    idx_list = [int(run[-3:]) for run in model_list if run != "overview.json"] if len(model_list)>0 else [0]
    run_idx = max(idx_list) + 1
    model_instance_id = f"run_{run_idx:03d}"
    model_instance_path = model_dir_path / Path(model_instance_id)
    model_instance_path.mkdir(parents=True,
                            exist_ok=True)
    
    # Get model overview path
    overview_path = model_dir_path / "overview.json"

    # First training run -> create overview
    if run_idx == 1: 
        # Create overview dict
        overview = {}
        # Save main information
        overview["model_name"] = cfg["model"]["model_name"]
        overview["epochs"] = cfg["training"]["epochs"]
        overview["best_epoch"] = metrics["best_epoch"]
        overview["training_runs"] = 1
        overview["best_run"] = 1
        overview["best_val_mae"] = metrics["best_epoch_val_mae"]
        # Init + fill model history with epoch-wise metrics
        overview["history"] = []
        for epoch_idx in range(cfg["training"]["epochs"]):
            epoch_metrics = {"epoch": epoch_idx+1, 
                             "train_loss": round(metrics["train_loss_list"][epoch_idx], 3),
                             "val_loss": round(metrics["val_loss_list"][epoch_idx], 3),
                             "train_mae": round(metrics["train_mae_list"][epoch_idx], 3),
                             "val_mae": round(metrics["val_mae_list"][epoch_idx], 3),
                             "lr": cfg["training"]["lr"],
                             "wd": cfg["training"]["weight_decay"],
                             "loss_args": cfg["loss"]["loss_args"],
                             "target_map_args": cfg["target_map"]["target_map_args"]}
            overview["history"].append(epoch_metrics)
    # Not first training run -> write to overview
    else:
        # Load overview
        with open(overview_path, "r") as f:
            overview = json.load(f)
        # Increment epoch + run tracks
        n_prev_epochs = overview["epochs"]
        overview["epochs"] += cfg["training"]["epochs"]
        overview["training_runs"] += 1
        # Update best epoch information
        if metrics["best_epoch_val_mae"] < overview["best_val_mae"]:
            overview["best_epoch"] = n_prev_epochs + metrics["best_epoch"] 
            overview["best_val_mae"] = metrics["best_epoch_val_mae"]
            overview["best_run"] = run_idx
        # Extend epoch history
        for epoch_idx in range(cfg["training"]["epochs"]):
            epoch_metrics = {"epoch": n_prev_epochs + epoch_idx+1, 
                             "train_loss": round(metrics["train_loss_list"][epoch_idx], 3),
                             "val_loss": round(metrics["val_loss_list"][epoch_idx], 3),
                             "train_mae": round(metrics["train_mae_list"][epoch_idx], 3),
                             "val_mae": round(metrics["val_mae_list"][epoch_idx], 3),
                             "lr": cfg["training"]["lr"],
                             "wd": cfg["training"]["weight_decay"],
                             "loss_args": cfg["loss"]["loss_args"],
                             "target_map_args": cfg["target_map"]["target_map_args"]}
            overview["history"].append(epoch_metrics)

    # Save model overview
    with open(overview_path, "w") as f:
        json.dump(overview, f, indent=4)

    # Save config
    cfg_save_path = model_instance_path / "config.yaml"
    print(f"[INFO] Saving config to '{cfg_save_path}'.")
    with open(cfg_save_path, "w") as f:
        yaml.dump(cfg, f)

    # Save last model state dict
    cp_save_path = model_instance_path / "last_cp.pth"
    print(f"[INFO] Saving last model cp to '{cp_save_path}'.")
    torch.save(obj=last_cp,
                f=cp_save_path)
    
    # Save optimizer state dict
    cp_save_path = model_instance_path / "optim_cp.pth"
    print(f"[INFO] Saving optimizer cp to '{cp_save_path}'.")
    torch.save(obj=optim_cp,
                f=cp_save_path)

    # Save best model state dict
    cp_save_path = model_instance_path / "best_cp.pth"
    print(f"[INFO] Saving best model cp to '{cp_save_path}'.")
    torch.save(obj=best_cp,
                f=cp_save_path)
    

def load_model(model_name: str,
               run_id: int = None,
               target_dir: str | Path = "models"
               ) -> torch.nn.Module:
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
    target_dir : str or Path, optional
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
    TypeError
        If the model type specified in metadata is not implemented.
    
    Notes
    -----
    - Expects the following directory structure:
        target_dir/
            model_name/
                training_run_XX/
                    best_cp.pth
                    metadata.json
    """
    
    model_dir_path = Path(target_dir) / model_name

    if not model_dir_path.exists():
        raise FileNotFoundError(f"No model path found matching the given path '{model_dir_path}'")
    
    # If not otherwise specified load model from last training run
    if run_id is None:
        model_list = os.listdir(model_dir_path)
        idx_list = [int(run[-3:]) for run in model_list if run != "overview.json"]
        model_idx = max(idx_list)
        model_instance_id = f"training_run_{model_idx:03d}"
        model_instance_path = model_dir_path / Path(model_instance_id)
    # Else load model with given run id
    else:
        model_instance_id = f"training_run_{run_id:03d}"
        model_instance_path = model_dir_path / Path(model_instance_id)
        if not model_instance_path.exists():
            raise FileNotFoundError(f"No model path found for the given run id, i.e. under path '{model_instance_path}'")

        
    # Load metadata
    json_path = model_instance_path / "metadata.json"
    with open(json_path, "r") as f:
        metadata = json.load(f)

    # Return model if model type matches an implemented model type
    model_type = metadata["hyperparameters"]["model_type"]
    if model_type == "density_model": 
        model = models.DensityModel(activation=metadata["hyperparameters"]["activation"])
        model_path = model_instance_path / "best_cp.pth"
        model.load_state_dict(torch.load(model_path))
        print(f"[INFO] Loaded model from '{model_path}'.")
        return model
    else:
        raise TypeError(f"Model type {model_type} unknown. Update of load_model function required.")
    
def init_model(model_name: str, 
               model_type: str, 
               activation: str = "ReLU", 
               run_id: int = None, 
               cp: str = "last", 
               target_dir: str = "models"
               ) -> torch.nn.Module:
    """
    Initialize a new model or load an existing model
    
    Checks if a model with the given name exists in the target directory. 
    If it exists, prompts the user to continue training the existing model or exit.
    If it does not exist, initializes a new model of the specified type.
    
    Parameters
    ----------
    model_name : str
        Name of the model to initialize or load.
    model_type : str
        Type of model to create if it does not exist.
    activation : {"ReLU", "ReLUTanh", "Sigmoid"}, optional
        Last layer activation function of model. Default is "ReLU".
    run_id : int or None, optional
        Identifier of the training run to load. 
        If None, loads the most recent training run. Default is None.
    cp : {"last", "best"}, optional
        Specifies which checkpoint to load when initializing an existing model.
        Default is "last".
    target_dir : str, optional
        Root directory where models are stored. Default is "models".
    
    Returns
    -------
    model : torch.nn.Module
        The loaded or newly initialized model.
    continue_training : bool
        Flag indicating if new model is created or existing one loaded.
    
    Raises
    ------
    InterruptedError
        If the user chooses to exit instead of continuing training an existing model.
    TypeError
        If the model type is not implemented.
    """
    continue_training = False

    # Get list of models from the target_dir (default: "models")
    current_dir = os.getcwd()
    if target_dir in os.listdir(current_dir): # Case 1: wd in root dir
        models_dir = os.path.join(current_dir, target_dir)
    else: # Case 2: wd in scripts dir
        parent_dir = os.path.dirname(current_dir)
        models_dir = os.path.join(parent_dir, target_dir)
    model_names = os.listdir(models_dir)

    # Check if model under name already exists
    if model_name in model_names:
        # If model exists: Ask user to continue training or exit
        user_in = input("Model with model name alrady exists.\nContinue training (y) existing model or exit (n)?\n [y/n]:")
        
        # Wait for valid user answer
        while user_in not in ["y","n"]:
            user_in = input("Valid input: {y,n} ({Continue training, exit})\n")
        
        # Case 1: Exit chosen
        if user_in == "n":
            raise InterruptedError("Exit chosen by user.")
        # Case 2: Continue training chosen => load model
        else:
            continue_training = True
            return load_model(model_name, run_id, cp, target_dir), continue_training
    
    # If model not exists yet, init a new one
    else:
        if model_type == "density_model":
            return models.DensityModel(activation=activation), continue_training
        else:
            raise TypeError(f"Model type {model_type} unknown.")

def init_loss(cfg: dict[str, Any]):
    """
    Initialize the loss function specified in the config.

    Parameters
    ----------
    cfg : dict[str, Any]
        Config dict.

    Returns
    -------
    torch.nn.Module
        Loss function.

    Raises
    ------
    KeyError
        If the loss function specified in cfg["loss"]["loss_fun"] is not supported.
    """

    if cfg["loss"]["loss_fun"] == "DensityCountLoss":
        return loss.DensityCountLoss(lambda_count=cfg["loss"]["loss_args"]["lbda_count"])
    else:
        raise KeyError("Loss function unknown. Specify existing loss function in the config.")
    
def init_optimizer(model_params: Iterator[torch.nn.Parameter],
                   cfg: dict[str, Any],
                   continue_training: bool,
                   run_id: int = None
                   ) -> torch.optim.Optimizer:
    """
    Initialize an optimizer and optionally restore its state from a previous training run.

    Initializes the optimizer specified in the config. If continue_training is True,
    attempts to load the optimizer state from the most recent (or specified) training
    run to preserve accumulated momentum. The state is discarded and the optimizer
    is freshly initialized if the optimizer type changed, the learning rate changed
    by more than a factor of 10, or no saved state is found.

    Parameters
    ----------
    model_params : Iterator[torch.nn.Parameter]
        Parameters of the model to be optimized (i.e. model.parameters()).
    cfg : dict[str, Any]
        Config dict.
    continue_training : bool
        If True, attempts to resume from a previous run's optimizer state.
    run_id : int or None, optional
        Identifier of the training run to load. 
        If None, loads the most recent training run. Default is None.

    Returns
    -------
    torch.optim.Optimizer
        Initialized optimizer.

    Raises
    ------
    KeyError
        If the optimizer specified in cfg["training"]["optimizer"] is not supported.
    FileNotFoundError
        If continue_training is True, run_id is specified, but no corresponding
        model directory exists.
    """
    # Init optimizer
    if cfg["training"]["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW( 
                model_params,
                lr=cfg["training"]["lr"],
                weight_decay=cfg["training"]["weight_decay"]
            )
    else:
        raise KeyError("Optimizer unknown. Set in config one of \{AdamW\}.")
    
    # Continue training: Load optimizer state dict
    if continue_training:
        # Define model path (no need to check if existing as continue_training can only be True for existing model path)
        model_dir_path = Path(cfg["paths"]["models"]) / cfg["model"]["model_name"]
        
        # If not otherwise specified load optimizer from last training run
        if run_id is None:
            model_list = os.listdir(model_dir_path)
            idx_list = [int(run[-3:]) for run in model_list if run != "overview.json"]
            model_idx = max(idx_list)
            model_instance_id = f"training_run_{model_idx:03d}"
            model_instance_path = model_dir_path / Path(model_instance_id)
        # Else load optimizer with given run id
        else:
            model_instance_id = f"training_run_{run_id:03d}"
            model_instance_path = model_dir_path / Path(model_instance_id)
            if not model_instance_path.exists():
                raise FileNotFoundError(f"No model path found for the given run id, i.e. under path '{model_instance_path}'")
        
        # Load old config
        config_path = model_instance_path / "config.yaml"
        old_cfg = load_config(path=config_path)

        ## Check if we can use existing optimizer
        optim_path = model_instance_path / "optim_cp.pth"
        use_existing = True

        # Strong lr change?
        lr_ratio = old_cfg["training"]["lr"] / cfg["training"]["lr"]
        if lr_ratio > 10 or lr_ratio < 1/10:
            use_existing = False
        
        # Optimizer changed?
        if old_cfg["training"]["optimizer"] != cfg["training"]["optimizer"]:
            use_existing = False

        # Optimizer not existend (case of old runs, where optim was ot saved)
        if not optim_path.exists():
            use_existing = False

        if use_existing:
            # Load optimizer
            optimizer.load_state_dict(torch.load(optim_path, map_location="cpu"))

            # Make sure that 'new' lr and weight_decay is used (not from loaded state dict)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg["training"]["lr"]
                param_group["weight_decay"] = cfg["training"]["weight_decay"]

    return optimizer


def load_config(path: str | Path ="config/config.yaml"
                ) -> dict[str, Any]:
    """
    Load config file with hyperparameters.
    
    Parameters
    ----------
    path : str or Path, optional
        Path to config yaml.
    
    Returns
    -------
    cfg : dict
        The config file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_train_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training a model.
    
    Defines and parses hyperparameters for training.
    
    Returns
    -------
    args : argparse.Namespace
        Parsed command-line arguments with attributes:
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for train/val/test.
        - lr (float): Learning rate.
        - model_name (str): Name of the saved model file.
        - model_type (str): Type of model to use.
        - activation (str): Last layer activation of model.
        - target_map_fun (str): Function id to create target map.
        - weight_decay (float): Weight decay for optimizer.
        - short_side (int): Short side length for image transformation.
        - sigma (float): Standard deviation for target density maps.
        - lbda_count (float): Weight for the count loss.
    """
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
        
    parser.add_argument("--model-type", type=str, default="density-model",
                        help="Type of model")
    
    parser.add_argument("--activation", type=str, default="ReLU",
                        help="Last layer activation")

    parser.add_argument("--target-map-fun", type=str, default="generate_density_map",
                        help="Function id to create target map")
    
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    
    parser.add_argument("--short-side", type=int, default=512,
                        help="Short side len of transformed image")
    
    parser.add_argument("--sigma", type=float, default=1,
                        help="Target density map standard deviation")
    
    parser.add_argument("--lbda-count", type=float, default=0.5,
                        help="Count loss weight")
    
    return parser.parse_args()


def parse_test_args() -> argparse.Namespace:
    """
    Parse command-line arguments for testing a model.
    
    Defines and parses hyperparameters for testing.
    
    Returns
    -------
    args : argparse.Namespace
        Parsed command-line arguments with attributes:
        - model_name (str): Name of the saved model file.
        - batch_size (int): Batch size for train/val/test.
        - short_side (int): Short side length for image transformation.
        - target_map_fun (str): Function id to create target map.
        - sigma (float): Standard deviation for target density maps.
        - lbda_count (float): Weight for the count loss.
        - run_id (int): ID of the training run to load model from.
        - cp (str): "last" or "best" model checkpoint to load.
    """
    # Creating a parser
    parser = argparse.ArgumentParser(description="Test model")
    
    # Add parser arguments

    parser.add_argument("--model-name", type=str, default="model0",
                    help="Saved model filename")
    
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for train/val/test")
    
    parser.add_argument("--short-side", type=int, default=512,
                        help="Short side len of transformed image")
    
    parser.add_argument("--sigma", type=float, default=1,
                        help="Target density map standard deviation")
    
    parser.add_argument("--target-map-fun", type=str, default="generate_density_map",
                    help="Function id to create target map")
    
    parser.add_argument("--lbda-count", type=float, default=0.5,
                        help="Count loss weight")
    
    parser.add_argument("--run-id", type=int, default=None,
                    help="The training run from which to take the model")
    
    parser.add_argument("--cp", type=str, default="last",
                help="Test model after 'last' epoch or in 'best' epoch")
    
    return parser.parse_args()