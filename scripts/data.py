"""
Class for creating custom PyTorch dataset and dataloader setup.
"""

from pathlib import Path
from typing import Callable, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import scripts.data_transformations as transforms
from scripts.utils import collate_fn
from scripts.target_maps import generate_density_map, generate_density_map_adaptive



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
    use_blend_maps: bool, optional
        If true, pregenerated target maps are used.
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
    def __init__(self, 
                 root: str | Path, 
                 target_map_fun: Callable, 
                 transform: Callable = None,
                 use_blend_maps: bool = False, 
                 *target_map_args: Any, 
                 **target_map_kwargs: Any
                 ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.target_map_fun = target_map_fun
        self.target_map_args = target_map_args
        self.target_map_kwargs = target_map_kwargs
        self.use_blend_maps = use_blend_maps
        self.images = sorted((self.root / "images").glob("*.jpg"))

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, 
                    idx: int
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Blend map logic
        if self.use_blend_maps:
            blend_path = self.root / "blend_maps" / f"{img_path.stem}.npz"
            if blend_path.exists():
               target_map = torch.from_numpy(np.load(blend_path)["map"]).unsqueeze(0)
            else: 
                raise FileExistsError(f"Blend map path does not exist: '{blend_path}'.")
            
            # Apply transforms if provided
            if self.transform:
                img, coords, target_map = self.transform(img, coords, target_map)
        else:
            # Apply transforms if provided
            if self.transform:
                img, coords = self.transform(img, coords)

            # Generate target map from coordinates
            _, H, W = img.shape
            target_map = self.target_map_fun(coords, H, W, *self.target_map_args, **self.target_map_kwargs) 
            # NOTE: TypeError? -> Check config for target_map_fun - target_map_args mismatch
            target_map = target_map.unsqueeze(0) # Add dim s.t. target maps are later of dim (B,1,H,W) matching model output

        return img, target_map, coords
    


def get_dataloader(split: str,
                    cfg: dict[str, Any]
                    ) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader for a given data split.

    Parameters
    ----------
    split : {'train', 'val', 'test'}
        Dataset split to load. Training split is shuffled; validation
        and test splits are not.
    cfg : dict
        Config dict.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        DataLoader for the specified split.

    Raises
    ------
    ValueError
        If split is not one of {'train', 'val', 'test'}.
    ValueError
        If the target map function specified in cfg is not recognised.
    """

    # Set target map function 
    if cfg["target_map"]["target_map_fun"] == "generate_density_map":
        tmf = generate_density_map
    elif cfg["target_map"]["target_map_fun"] == "generate_density_map_adaptive":
        tmf = generate_density_map_adaptive
    else:
        raise ValueError("Unknown target map function. Specify existing target map function in config file.")
    
    # Set dataset args according to split
    if split == "train":
        data_root  = cfg["paths"]["train_data"]
        shuffle = True
    elif split == "val":
        data_root = cfg["paths"]["val_data"]
        shuffle = False
    elif split == "test":
        data_root = cfg["paths"]["test_data"]
        shuffle = False
    else:
        raise ValueError("split must be one of {'train', 'val', 'test'}.")

    # Create dataset
    ds = TrichomeDataset(root=data_root,
                         transform=get_transforms(split, cfg),
                         use_blend_maps=cfg["target_map"].get("use_blend_maps", False),
                         target_map_fun=tmf,
                        **cfg["target_map"]["target_map_args"])
    
    # Create dataloader
    dataloader = DataLoader(dataset=ds,
                    batch_size=cfg["training"]["batch_size"],
                    shuffle=shuffle,
                    collate_fn=collate_fn)

    return dataloader

def get_transforms(split: str, 
                   cfg: dict[str, Any]
                   ) -> transforms.Compose:
    """
    Build the transform pipeline for a given data split.

    Training splits receive augmentation transforms (random flips,
    brightness jitter) in addition to resizing and padding. Validation
    and test splits receive only resizing and padding. ImageNet
    normalisation is appended to all splits if enabled in cfg.

    Parameters
    ----------
    split : {'train', 'val', 'test'}
        Dataset split to build transforms for.
    cfg : dict
        Config dict.

    Returns
    -------
    transform : transforms.Compose
        Composed transform pipeline for the specified split.

    Raises
    ------
    ValueError
        If split is not one of {'train', 'val', 'test'}.
    """
    if split == "train":
        transform_list = [
            transforms.ResizeShortSide(cfg["transforms"]["short_side"]),
            transforms.PadToMultipleOf32(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomBrightness(cfg["transforms"]["brightness"])
        ]
    elif split == "val" or split == "test":
        transform_list = [
            transforms.ResizeShortSide(cfg["transforms"]["short_side"]),
            transforms.PadToMultipleOf32()
        ]
    else:
        raise ValueError("split must be one of {'train', 'val', 'test'}.")
    
    # Add imagenet normalization if specified
    if cfg["transforms"]["imagenet_normalization"]:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    return transforms.Compose(transform_list)

def generate_blend_maps(split: str,
                        model: torch.nn.Module,
                        cfg: dict[str, Any],
                        target_map_fun: Callable,
                        alpha_blend: float,
                        device: torch.device
                        ) -> None:
    """
    Generate and save blend maps for all images in a data split.

    Blend maps are a weighted combination of the model's predicted
    density map and the ground-truth target map, scaled to preserve
    total count. They are saved as compressed .npz files under
    <data_root>/blend_maps/ and can be used in subsequent training
    runs via the use_blend_maps option.

    Note: only use models trained without cropping augmentation, as
    blend maps are upsampled back to the original image resolution and
    spatial consistency is required.

    Parameters
    ----------
    split : str
        Data split for which blend maps are generated.
    model : torch.nn.Module
        Trained density estimation model used to produce predictions.
    cfg : dict
        Config dict.
    target_map_fun : callable
        Function to generate ground-truth target maps from coordinates.
    alpha_blend : float
        Blending weight in [0, 1]. A value of 1.0 uses the model
        prediction only; 0.0 uses the target map only.
    device : torch.device
        Device on which to run model inference.

    Raises
    ------
    ValueError
        If split is not one of {'train', 'val', 'test'}.
    """
    # Get data path
    if split == "train":
        root = Path(cfg["paths"]["train_data"])
    elif split == "val":
        root = Path(cfg["paths"]["val_data"])
    elif split == "test":
        root = Path(cfg["paths"]["test_data"])
    else:
        raise ValueError("Split must be one of /{'train', 'val', 'test'}.")
    
    # Make output dir
    out_dir = root / "blend_maps"
    out_dir.mkdir(exist_ok=True)

    # Setup transformtion (use val split transformations to get consistent 
    # target maps not influenced by training transformations)
    transform = get_transforms(split="val", cfg=cfg)
    print("[WARNING] Only generate blend maps based on models where no crop is applied, s.t. blend maps can be upsampled to the orignal image scale.")

    model.eval().to(device)
    image_paths = sorted((root / "images").glob("*.jpg"))

    with torch.no_grad():
        for img_path in image_paths:

            # Load img + coords, transform to tensors
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
            coord_path = root / "coords" / f"{img_path.stem}.npy"
            coords = torch.from_numpy(np.load(coord_path)).float()

            _, H_orig, W_orig = img.shape

            # Apply trasformations
            img, coords = transform(img, coords)

            _, H_t, W_t = img.shape

            # Make predictions
            pred  = model(img.unsqueeze(0).to(device)).squeeze().cpu()

            # Upsample prediction back to original image space
            pred = F.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=(H_orig, W_orig),
                mode="bilinear",
                align_corners=False
            ).squeeze()  # (H_orig, W_orig)

            # Get target map (on transformed input to align with how it is applied in the dataset class)
            target_map = target_map_fun(coords, H_t, W_t,
                                        **cfg["target_map"]["target_map_args"])     
            
            # Upsample target maps back to original image space
            target_map = F.interpolate(
                target_map.unsqueeze(0).unsqueeze(0),
                size=(H_orig, W_orig),
                mode="bilinear",
                align_corners=False
            ).squeeze()  # (H_orig, W_orig)

            total_count = len(coords)
            if total_count > 0:
                # match scale
                pred  = pred * (target_map.sum() / pred.sum().clamp(1e-6))  
                # Get blended map
                blended = alpha_blend * pred + (1 - alpha_blend) * target_map
                # Normalize blended map
                blended = blended / blended.sum().clamp(1e-6) * total_count
            else:
                # No coords -> flat target
                blended = torch.zeros_like(pred)
    
            # save compressed maps
            np.savez_compressed(out_dir / f"{img_path.stem}.npz",
                                map=blended.numpy().astype(np.float32))

    print(f"[INFO] Saved {len(image_paths)} blend maps to '{out_dir}'")