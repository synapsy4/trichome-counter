"""
Utility functions.
"""

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
    root_dir = Path(cfg["paths"]["models"])
    root_dir.mkdir(parents=True,
                            exist_ok=True)

    # Create model save dir
    model_name = cfg["model"]["model_name"]
    model_dir = root_dir / Path(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model instance save path
    model_list = [p.name for p in model_dir.iterdir() if p.is_dir()]
    idx_list = [int(run[-3:]) for run in model_list] if len(model_list)>0 else [0]
    run_idx = max(idx_list) + 1
    model_instance_id = f"run_{run_idx:03d}"
    model_instance_dir = Path(model_dir) / model_instance_id
    model_instance_dir.mkdir(parents=True,
                            exist_ok=True)
    
    # Flag if new global best model cp is found
    new_best_model = True
    
    # Define overview and epoch log paths
    overview_path = model_dir / "overview.json"
    history_path = model_dir / "history.jsonl"

    # First run: Create new overview + history files
    if run_idx == 1:
        # Create overview dict
        overview = {
            "model_name": cfg["model"]["model_name"],
            "epochs": cfg["training"]["epochs"],
            "best_epoch": metrics["best_epoch"],
            "training_runs": 1,
            "best_run": 1,
            "best_val_mae": round(metrics["best_epoch_val_mae"], 3),
        }

        # Create history file
        with open(history_path, "w") as f:
            # Write epoch logs
            for epoch_idx in range(cfg["training"]["epochs"]):
                epoch_metrics = {
                    "epoch": epoch_idx + 1,
                    "train_loss": round(metrics["train_loss_list"][epoch_idx], 3),
                    "val_loss": round(metrics["val_loss_list"][epoch_idx], 3),
                    "train_mae": round(metrics["train_mae_list"][epoch_idx], 3),
                    "val_mae": round(metrics["val_mae_list"][epoch_idx], 3),
                    "lr": cfg["training"]["lr"],
                    "wd": cfg["training"]["weight_decay"],
                    "loss_args": cfg["loss"]["loss_args"],
                    "target_map_args": cfg["target_map"]["target_map_args"],
                }
                f.write(json.dumps(epoch_metrics) + "\n")

    # Not first run: Extend overview + history files
    else:
        # Load overview
        with open(overview_path, "r") as f:
            overview = json.load(f)
        
        # Get previous epoch count
        n_prev_epochs = overview["epochs"]

        # Update overview stats
        overview["epochs"] += cfg["training"]["epochs"]
        overview["training_runs"] += 1

        # Update best model info
        if metrics["best_epoch_val_mae"] < overview["best_val_mae"]:
            overview["best_epoch"] = n_prev_epochs + metrics["best_epoch"]
            overview["best_val_mae"] = round(metrics["best_epoch_val_mae"], 3)
            overview["best_run"] = run_idx
        else:
            new_best_model = False

        # Append new epoch logs
        with open(history_path, "a") as f:
            for epoch_idx in range(cfg["training"]["epochs"]):
                epoch_metrics = {
                    "epoch": n_prev_epochs + epoch_idx + 1,
                    "train_loss": round(metrics["train_loss_list"][epoch_idx], 3),
                    "val_loss": round(metrics["val_loss_list"][epoch_idx], 3),
                    "train_mae": round(metrics["train_mae_list"][epoch_idx], 3),
                    "val_mae": round(metrics["val_mae_list"][epoch_idx], 3),
                    "lr": cfg["training"]["lr"],
                    "wd": cfg["training"]["weight_decay"],
                    "loss_args": cfg["loss"]["loss_args"],
                    "target_map_args": cfg["target_map"]["target_map_args"],
                }
                f.write(json.dumps(epoch_metrics) + "\n")

    # Save overview
    with open(overview_path, "w") as f:
        json.dump(overview, f, indent=4)

    # Save config
    cfg_save_path = model_instance_dir / "config.yaml"
    print(f"[INFO] Saving config to '{cfg_save_path}'.")
    with open(cfg_save_path, "w") as f:
        yaml.dump(cfg, f)

    # Save last model state dict
    cp_save_path = model_dir / "last_cp.pth"
    print(f"[INFO] Saving last model cp to '{cp_save_path}'.")
    torch.save(obj=last_cp,
                f=cp_save_path)
    
    # Save optimizer state dict
    cp_save_path = model_dir / "optim_cp.pth"
    print(f"[INFO] Saving optimizer cp to '{cp_save_path}'.")
    torch.save(obj=optim_cp,
                f=cp_save_path)

    # Save best model state dict if new best model found
    if new_best_model:
        cp_save_path = model_dir / "best_cp.pth"
        print(f"[INFO] Saving best model cp to '{cp_save_path}'.")
        torch.save(obj=best_cp,
                    f=cp_save_path)
    
def get_model_instance_path(model_dir: str | Path,
                            cp: str = "last"
                            ) -> Path:
    """
    TODO: Add docstring
    """
    # Load last training run path
    if cp == "last":
        model_list = [p.name for p in model_dir.iterdir() if p.is_dir()]
        idx_list = [int(run[-3:]) for run in model_list]
        model_idx = max(idx_list)
        model_instance_id = f"run_{model_idx:03d}"
        model_instance_path = Path(model_dir) / model_instance_id
    # Else load path with best cp
    elif cp == "best":
        overview_path = Path(model_dir) / "overview.json"
        with open(overview_path, "r") as f:
            overview = json.load(f) 
        best_run_id = overview["best_run"]
        model_instance_id = f"run_{best_run_id:03d}"
        model_instance_path = Path(model_dir) / model_instance_id
    else:
        raise KeyError("Model cp must be one of \{'last','best'\}")
    return model_instance_path

def load_overview(model_name: str,
                  root: Path | str = "models"
                  ) -> dict:
    """
    TODO: Add docstring
    """
    model_dir = Path(root) / model_name
    overview_path = model_dir / "overview.json"

    # Load overview
    if overview_path.exists():
        with open(overview_path, "r") as f:
            return json.load(f)
    else:
        return None
    
def flatten_dict(d: dict[str, Any], 
                 parent_key: str = ""
                 ) -> dict[str, Any]:
    """
    TODO: Add docstirng
    """
    items = {}
    for k, v in d.items():
        key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, key))
        elif v is None:
            items[key] = "null"  # convert to string
        else:
            items[key] = v
    return items


def load_model(model_name: str,
               cp: str = "last",
               root_dir: str | Path = "models"
               ) -> torch.nn.Module:
    """
    Load a trained model instance and its corresponding checkpoint from disk.

    Parameters
    ----------
    model_name : str
        Name of the model directory inside `root_dir`.
    cp : str, optional
        Checkpoint to load: "last" for cp from last epoch, "best" for cp from
        best epoch.
        Default is "last".
    root_dir : str or Path, optional
        Root directory where models are stored.
        Default is "models".

    Returns
    -------
    model : torch.nn.Module
        Instantiated model with loaded state dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified model directory does not exist.
    KeyError
        If chosen cp does not exist.
    TypeError
        If the model type specified in metadata is not implemented.
    
    Notes
    -----
    - Expects the following directory structure:
        root_dir/
            model_name/
                run_XX/
                    config.yaml
                best_cp.pth
                last_cp.pth
                overview.json
    """
    
    model_dir = Path(root_dir) / model_name

    if not model_dir.exists():
        raise FileNotFoundError(f"No model path found matching the given path '{model_dir}'")
        
    # Get cp file name
    if cp == "last":
        cp_file = "last_cp.pth"
    elif cp == "best":
        print("[WARNING] 'best' cp chosen. Should only be used for evaluation or transfer learning. Otherwise number of training epochs is logged incorrectly + optimizer has later state.")
        cp_file = "best_cp.pth"
    else:
        raise KeyError("CP must be one of \{'last','best'\}")

    # Load config of model
    model_instance_path = get_model_instance_path(model_dir, cp)
    config_path = model_instance_path / "config.yaml"
    old_cfg = load_config(path=config_path)

    # Return model if model type matches an implemented model type
    model_type = old_cfg["model"]["model_type"]
    if model_type == "density-model": 
        model = models.DensityModel(activation=old_cfg["model"]["activation"])
        model_path = model_dir / cp_file
        model.load_state_dict(torch.load(model_path))
        print(f"[INFO] Loaded model from '{model_path}'.")
        return model
    else:
        raise TypeError(f"Model type {model_type} unknown. Update of load_model function required.")
    
def init_model(cfg: dict[str, Any],
               cp: str = "last", 
               root_dir: str = "models"
               ) -> tuple[torch.nn.Module, bool]:
    """
    Initialize a new model or load an existing model
    
    Checks if a model with the given name exists in the target directory. 
    If it exists, prompts the user to continue training the existing model or exit.
    If it does not exist, initializes a new model of the specified type or loads a
    pretrained model if specified in cfg.
    
    Parameters
    ----------
    cfg : dict[str, Any]
        Config dict.
    cp : {"last", "best"}, optional
        Specifies which checkpoint to load when initializing an existing model.
        Default is "last".
    root_dir : str, optional
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
        If the user chooses to exit.
    TypeError
        If the model type is not implemented or the pretrained type differs.
    KeyError
        If pretrained model does not exist.
    """
    continue_training = False

    # Get list of models from the root_dir (default: "models")
    current_dir = Path.cwd()
    if (current_dir / root_dir).is_dir():   # Case 1: wd in root dir
        models_dir = current_dir / root_dir
    else:                                   # Case 2: wd in scripts dir
        models_dir = current_dir.parent / root_dir
    model_names = [p.name for p in models_dir.iterdir()]

    # Case 1: Continue training
    if cfg["model"]["model_name"] in model_names:
        # If model exists: Ask user to continue training or exit
        user_in = input("Model with model name already exists.\nContinue training (y) existing model or exit (n)?\n [y/n]:")
        while user_in not in ["y","n"]:
            user_in = input("Valid input: {y,n} ({Continue training, exit})\n")
        
        # Case 1: Exit chosen
        if user_in == "n":
            raise InterruptedError("Exit chosen by user.")
        # Case 2: Continue training chosen => load model
        else:
            continue_training = True
            return load_model(cfg["model"]["model_name"], cp, root_dir), continue_training
    
    # Case 2: Init model from another model path (transfer learning)
    elif cfg["model"]["pre_model_name"]:

        if cfg["model"]["pre_model_name"] not in model_names:
            raise KeyError(f"Pretrained model {cfg['model']['pre_model_name']} not found in existing models.")
        
        # Get pretrained model path (last or specified run)
        pre_model_dir = Path(models_dir) / cfg["model"]["pre_model_name"]
        pre_model_instance_path = get_model_instance_path(pre_model_dir, cfg["model"]["pre_cp"])
        
        # Get pretrained model config to look for conflicts
        pre_model_cfg_path = pre_model_instance_path / "config.yaml"
        pre_model_cfg = load_config(pre_model_cfg_path)

        # Conflict case 1: Model type mismatch
        if pre_model_cfg["model"]["model_type"] != cfg["model"]["model_type"]:
            raise TypeError("Model type of pretrained model must be equal to model type of new model.")
        # Conflict case 2: Activation function mismatch
        if pre_model_cfg["model"]["activation"] != cfg["model"]["activation"]:
            user_in = input("Pretrained model last layer activation differs from chosen activation.\nStill proceed (y) or exit (n)?\n [y/n]")
            while user_in not in ["y","n"]:
                user_in = input("Valid input: {y,n} ({Proceed, exit})\n")
            if user_in == "n":
                raise InterruptedError("Exit chosen by user.")
   
        return load_model(cfg["model"]["pre_model_name"], cfg["model"]["pre_cp"], root_dir), continue_training

        
    # Case 3: Init new model
    else:
        if cfg["model"]["model_type"] == "density-model":
            return models.DensityModel(activation=cfg["model"]["activation"]), continue_training
        else:
            raise TypeError(f"Model type {cfg['model']['model_type']} unknown.")

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
                   ) -> torch.optim.Optimizer:
    """
    Initialize an optimizer and optionally restore its state from the last training run.

    Initializes the optimizer specified in the config. If continue_training is True,
    attempts to load the optimizer state from the last training run to preserve 
    accumulated momentum. The state is discarded and the optimizer is freshly 
    initialized if the optimizer type changed, the learning rate changed by more than 
    a factor of 10, or no saved state is found.

    Parameters
    ----------
    model_params : Iterator[torch.nn.Parameter]
        Parameters of the model to be optimized (i.e. model.parameters()).
    cfg : dict[str, Any]
        Config dict.
    continue_training : bool
        If True, attempts to resume from a previous run's optimizer state.

    Returns
    -------
    torch.optim.Optimizer
        Initialized optimizer.

    Raises
    ------
    KeyError
        If the optimizer specified in cfg["training"]["optimizer"] is not supported.
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
        model_dir = Path(cfg["paths"]["models"]) / cfg["model"]["model_name"]
        
        # Load old config
        model_instance_path = get_model_instance_path(model_dir, "last") # Optimizer is always from last cp
        config_path = model_instance_path / "config.yaml"
        old_cfg = load_config(path=config_path)

        ## Check if we can use existing optimizer
        optim_path = model_dir / "optim_cp.pth"
        use_existing = True

        # Strong lr change?
        lr_ratio = old_cfg["training"]["lr"] / cfg["training"]["lr"]
        if lr_ratio > 10 or lr_ratio < 1/10:
            use_existing = False
        
        # Optimizer changed?
        if old_cfg["training"]["optimizer"] != cfg["training"]["optimizer"]:
            use_existing = False

        # Optimizer not existend (case of old runs, where optim was not saved)
        if not optim_path.exists():
            use_existing = False

        if use_existing:
            # Load optimizer
            optimizer.load_state_dict(torch.load(optim_path, map_location="cpu"))
            print(f"[INFO] Loaded optimizer from '{optim_path}'.")

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


def parse_test_args() -> argparse.Namespace:
    """
    Parse command-line arguments for testing a model.
    
    Defines and parses hyperparameters for testing.
    
    Returns
    -------
    args : argparse.Namespace
        Parsed command-line arguments with attributes:
        - model_name (str): Name of the saved model file.
        - cp (str): "last" or "best" model checkpoint to load.
        - model-root-dir (str): Root directory where model is located.
    """
    # Creating a parser
    parser = argparse.ArgumentParser(description="Test model")
    
    # Add parser arguments

    parser.add_argument("--model-name", type=str, default="model0",
                    help="Saved model filename")
    
    parser.add_argument("--cp", type=str, default="last",
                help="Test model after 'last' epoch or in 'best' epoch")
    
    parser.add_argument("--model-root-dir", type=str, default="models",
            help="Root directory for model directory.")
    
    return parser.parse_args()