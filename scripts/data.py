"""
Class for creating custom PyTorch dataset and dataloader setup.
"""

from pathlib import Path
from typing import Callable, Any

import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import scripts.data_transformations as transforms
from scripts.utils import collate_fn
from scripts.target_maps import generate_density_map



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
    def __init__(self, 
                 root: str | Path, 
                 target_map_fun: Callable, 
                 transform: Callable = None, 
                 *target_map_args: Any, 
                 **target_map_kwargs: Any
                 ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.target_map_fun = target_map_fun
        self.target_map_args = target_map_args
        self.target_map_kwargs = target_map_kwargs
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

        # Apply transforms if provided
        if self.transform:
            img, coords = self.transform(img, coords)

        # Generate target map from coordinates
        _, H, W = img.shape
        target_map = self.target_map_fun(coords, H, W, *self.target_map_args, **self.target_map_kwargs)
        target_map = target_map.unsqueeze(0) # Add dim s.t. target maps are later of dim (B,1,H,W) matching model output

        return img, target_map, coords
    


    def get_dataloader(split: str,
                       cfg: dict[str, Any]
                       ) -> torch.utils.data.DataLoader:
        """
        TODO: Add function info.
        """

        # Set target map function 
        tmf = generate_density_map if cfg.training.target_map_fun == "generate_density_map" else None

        if split == "train":
            # Create dataset
            ds = TrichomeDataset(root=cfg.paths.train_path,  
                            transform=transforms.Compose(
                                [transforms.ResizeShortSide(cfg.transforms.short_side),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomBrightness(cfg.transforms.brightness)]),
                            target_map_fun=tmf,
                            sigma=cfg.training.target_map_args) 
            # Create dataloader
            dataloader = DataLoader(dataset=ds,
                                batch_size=cfg.training.batch_size,
                                shuffle=True,
                                collate_fn=collate_fn)
        elif split == "val" or split == "test":
            # Create dataset
            ds = TrichomeDataset(root=cfg.paths.train_path,  
                            transform=transforms.Compose(
                                [transforms.ResizeShortSide(cfg.transforms.short_side),
                                 transforms.PadToMultipleOf32()]),
                            target_map_fun=tmf,
                            sigma=cfg.training.target_map_args) 
            # Create dataloader
            dataloader = DataLoader(dataset=ds,
                                batch_size=cfg.training.batch_size,
                                shuffle=False,
                                collate_fn=collate_fn)
        else:
            raise KeyError("split must be one of {'train', 'val', 'test'}.")

        return dataloader