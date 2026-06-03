
"""
Custom transformation classes to transform images and coordinates.
"""

import random
from typing import Callable

import torch
import torch.nn.functional as F

class Compose:
    """
    Compose multiple transforms together for image and coordinate data.

    Parameters
    ----------
    transforms : list
        List of transform objects to apply sequentially.

    Methods
    -------
    __call__(image, coords, target_map=None)
        Apply all transforms in sequence to image, coordinates, and
        optionally a target map.
    """
    def __init__(self, transforms: list[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, 
                 image: torch.Tensor, 
                 coords: torch.Tensor,
                 target_map: torch.Tensor = None
                 ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply all transforms sequentially.

        Parameters
        ----------
        image : torch.Tensor
            Input image.
        coords : torch.Tensor
            Nx2 array of point coordinates.
        target_map : torch.Tensor, optional
            Target for training. If None, only image and coords are 
            returned.

        Returns
        -------
        image : torch.Tensor
            Transformed image.
        coords : torch.Tensor
            Transformed point coordinates.
        target_map : torch.Tensor
            Transformed target map. Only returned if a target_map was
            provided as input.
        """
        # Apply transforms sequentially
        if target_map is None:
            for t in self.transforms:
                image, coords = t(image, coords)
            return image, coords
        else:
            for t in self.transforms:
                image, coords, target_map = t(image, coords, target_map)
            return image, coords, target_map
    

class ResizeShortSide:
    """
    Resize image and scale coordinates so the shorter side matches target length.

    Parameters
    ----------
    short_side : int
        Target length for the shorter side of the image.

    Methods
    -------
    __call__(image, coords, target_map=None)
        Resize image, scale point coordinates proportionally, and optionally
        resize target map.
    """
    def __init__(self, short_side: int) -> None:
        self.short_side = short_side

    def __call__(self, 
                 image: torch.Tensor, 
                 coords: torch.Tensor,
                 target_map: torch.Tensor = None
                 ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resize image and scale coordinates.

        Parameters
        ----------
        image : torch.Tensor
            Input image.
        coords : torch.Tensor
            Nx2 array of point coordinates.
        target_map : torch.Tensor, optional
            Target for training. If None, only image and coords are 
            returned.

        Returns
        -------
        image : torch.Tensor
            Resized image.
        coords : torch.Tensor
            Resized point coordinates.
        target_map : torch.Tensor
            Resized target map. Only returned if a target_map was
            provided as input.
        """
        # Get scaling via minimal side length (short side length)
        _, H, W = image.shape
        scale = self.short_side / min(H, W)

        # Apply scaling to get new W and H
        new_W = round(W * scale)
        new_H = round(H * scale)

        # Resize image
        image = F.interpolate(
            image.unsqueeze(0),
            size=(new_H, new_W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Scale coordinates
        coords = coords * scale

        if target_map is not None:
            target_map = F.interpolate(
                target_map.unsqueeze(0), size=(new_H, new_W),
                mode="bilinear", align_corners=False
            ).squeeze(0)
            # Normalize (sum target_map = n_coords)
            total_count = len(coords)
            if total_count > 0:
                target_map = target_map / target_map.sum() * total_count

            return image, coords, target_map
        else:
            return image, coords


class RandomCrop:
    """
    Randomly crop image and filter coordinates to those within the crop.

    Parameters
    ----------
    crop_w : int
        Width of the crop.
    crop_h : int
        Height of the crop.

    Methods
    -------
    __call__(image, coords, target_map=None)
        Randomly crop image, adjust point coordinates, and optionally
        a target map.

    Raises
    ------
    ValueError
        If crop size is larger than image dimensions.
    """
    def __init__(self, 
                 crop_w: int, 
                 crop_h: int
                 ) -> None:
        self.crop_w = crop_w
        self.crop_h = crop_h

    def __call__(self, 
                 image: torch.Tensor, 
                 coords: torch.Tensor,
                 target_map: torch.Tensor = None
                 ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply random crop to image and coordinates.

        Parameters
        ----------
        image : torch.Tensor
            Input image.
        coords : torch.Tensor
            Nx2 array of point coordinates.
        target_map : torch.Tensor, optional
            Target for training. If None, only image and coords are 
            returned.

        Returns
        -------
        image : torch.Tensor
            Cropped image.
        coords : torch.Tensor
            Cropped point coordinates.
        target_map : torch.Tensor
            Cropped target map. Only returned if a target_map was
            provided as input.

        Raises
        ------
        ValueError
            If crop size exceeds image dimensions.
        """
        _, H, W = image.shape

        # Check if crop size is within image size
        if W < self.crop_w or H < self.crop_h:
            raise ValueError("Crop size larger than image.")

        # Get random upper left corner coordinates
        y0 = torch.randint(0, H - self.crop_h + 1, (1,)).item()
        x0 = torch.randint(0, W - self.crop_w + 1, (1,)).item()

        # Copy image and apply crop to copy
        image = image[:,y0:y0+self.crop_h,x0:x0+self.crop_w].clone()

        # Update cropped image coordinates based on sampled upper left corner
        if len(coords) > 0:
            coords = coords.clone()
            coords[:, 0] -= x0
            coords[:, 1] -= y0

            # define mask to keep only coordinates inside crop
            mask = (
                (coords[:, 0] >= 0) &
                (coords[:, 0] < self.crop_w) &
                (coords[:, 1] >= 0) &
                (coords[:, 1] < self.crop_h)
            )

            coords = coords[mask]

        if target_map is not None:
            target_map = target_map[:, y0:y0+self.crop_h, x0:x0+self.crop_w].clone()
            target_map = target_map / target_map.sum() * len(coords)
            return image, coords, target_map
        else:
            return image, coords
    

class RandomHorizontalFlip:
    """
    Randomly flip image and coordinates horizontally with given probability.

    Parameters
    ----------
    p : float, optional
        Probability of applying the flip (default: 0.5).

    Methods
    -------
    __call__(image, coords, target_map=None)
        Apply random horizontal flip to image, coordinates, and optionally a
        target map.
    """
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, 
                 image: torch.Tensor, 
                 coords: torch.Tensor,
                 target_map: torch.Tensor = None
                 ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply random horizontal flip.

        Parameters
        ----------
        image : torch.Tensor
            Input image.
        coords : torch.Tensor
            Nx2 array of point coordinates.
        target_map : torch.Tensor, optional
            Target for training. If None, only image and coords are 
            returned.

        Returns
        -------
        image : torch.Tensor
            Flipped or original image.
        coords : torch.Tensor
            Flipped or original point coordinates.
        target_map : torch.Tensor
            Flipped or original target map. Only returned if a 
            target_map was provided as input.
        """
        # By chance return data without flip
        if torch.rand(1).item() > self.p:
            return (image, coords, target_map) if target_map is not None else (image, coords)

        # Flip image
        _, _, W = image.shape
        image = torch.flip(image, dims=[2])

        # If coordinates exist, flip them
        if len(coords) > 0:
            coords = coords.clone()
            coords[:, 0] = W - 1 - coords[:, 0]

        if target_map is not None:
            target_map = torch.flip(target_map, dims=[2])
            return image, coords, target_map
        else:
            return image, coords
    

class RandomVerticalFlip:
    """
    Randomly flip image and coordinates vertically with given probability.

    Parameters
    ----------
    p : float, optional
        Probability of applying the flip (default: 0.5).

    Methods
    -------
    __call__(image, coords, target_map=None)
        Apply random vertical flip to image, coordinates, and optionally
        a target map.
    """
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, 
                 image: torch.Tensor, 
                 coords: torch.Tensor,
                 target_map: torch.Tensor = None
                 ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply random vertical flip.

        Parameters
        ----------
        image : torch.Tensor
            Input image.
        coords : torch.Tensor
            Nx2 array of point coordinates.
        target_map : torch.Tensor, optional
            Target for training. If None, only image and coords are 
            returned.

        Returns
        -------
        image : torch.Tensor
            Flipped or original image.
        coords : torch.Tensor
            Flipped or original point coordinates.
        target_map : torch.Tensor
            Flipped or original target map. Only returned if a 
            target_map was provided as input.
        """
        # By chance return data without flip
        if torch.rand(1).item() > self.p:
            return (image, coords, target_map) if target_map is not None else (image, coords)
        
        # Flip image along vertical axis (=0)
        _, H, _ = image.shape
        image = torch.flip(image, dims=[1])

        # If coordinates exist, flip them
        if len(coords) > 0:
            coords = coords.clone()
            coords[:, 1] = H - 1 - coords[:, 1]

        if target_map is not None:
            target_map = torch.flip(target_map, dims=[1])
            return image, coords, target_map
        else:
            return image, coords
    



class RandomBrightness:
    """
    Randomly adjust image brightness.

    Parameters
    ----------
    brightness_factor : float
        Maximum relative brightness change.
        Example: 0.2 -> brightness in [0.8, 1.2]

    Methods
    -------
    __call__(image, coords, target_map=None)
        Adjust brightness without modifying coordinates and the optional
        target map.
    """

    def __init__(self, brightness_factor: float = 0.2) -> None:
        self.brightness_factor = brightness_factor

    def __call__(self, 
                 image: torch.Tensor, 
                 coords: torch.Tensor,
                 target_map: torch.Tensor = None
                 ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adjust image brightness.

        Parameters
        ----------
        image : torch.Tensor
            Input image (C, H, W), expected in range [0,1]
        coords : torch.Tensor
            Nx2 point coordinates
        target_map : torch.Tensor, optional
            Target for training. If None, only image and coords are 
            returned.

        Returns
        -------
        image : torch.Tensor
            Brightness-adjusted image
        coords : torch.Tensor
            Unchanged coordinates
        target_map : torch.Tensor
            Unchanged target map. Only returned if a target_map was
            provided as input.
        """

        # Sample brightness factor
        factor = 1.0 + random.uniform(
            -self.brightness_factor,
            self.brightness_factor
        )

        image = image * factor
        image = torch.clamp(image, 0.0, 1.0)

        return (image, coords, target_map) if target_map is not None else (image, coords)
    
class RandomRotate90:
    """
    Randomly rotate image and coordinates by 90 degrees clockwise
    with given probability.

    Parameters
    ----------
    p : float, optional
        Probability of applying the rotation (default: 0.5).

    Methods
    -------
    __call__(image, coords, target_map=None)
        Apply random 90° rotation to image, coordinates, and optionally
        a target map.
    """
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, 
                 image: torch.Tensor, 
                 coords: torch.Tensor,
                 target_map: torch.Tensor = None
                 ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply random 90° clockwise rotation.

        Parameters
        ----------
        image : torch.Tensor
            Input image of shape (C, H, W).
        coords : torch.Tensor
            Nx2 array of point coordinates (x, y).
        target_map : torch.Tensor, optional
            Target for training. If None, only image and coords are 
            returned.

        Returns
        -------
        image : torch.Tensor
            Rotated or original image.
        coords : torch.Tensor
            Rotated or original point coordinates.
        target_map : torch.Tensor
            Totated or original target map. Only returned if a 
            target_map was provided as input.
        """
        # By chance return data without rotation
        if torch.rand(1).item() > self.p:
            return (image, coords, target_map) if target_map is not None else (image, coords)

        _, H, _ = image.shape

        # Rotate image 90° clockwise
        image = torch.rot90(image, k=-1, dims=[1, 2])

        # If coordinates exist, rotate them
        if len(coords) > 0:
            coords = coords.clone()
            x = coords[:, 0].clone()
            y = coords[:, 1].clone()

            # 90° clockwise rotation:
            # (x, y) -> (H - 1 - y, x)
            coords[:, 0] = H - 1 - y
            coords[:, 1] = x

        if target_map is not None:
            target_map = torch.rot90(target_map, k=-1, dims=[1, 2])
            return image, coords, target_map
        else:
            return image, coords


class PadToMultipleOf32:
    """
    Pad image so height and width are divisible by 32.

    This is useful for encoder-decoder architectures (e.g. U-Net with
    ResNet backbone) that downsample the feature maps by a total factor
    of 32. Padding avoids shape mismatches in skip connections.

    Padding is applied to the bottom and right side of the image.
    Coordinates remain unchanged.

    Methods
    -------
    __call__(image, coords, target_map=None)
        Pad image spatial dimensions to next multiple of 32.
    """

    def __call__(self, 
                 image: torch.Tensor, 
                 coords: torch.Tensor,
                 target_map: torch.Tensor = None
                 ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pad image to nearest multiple of 32.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape (C, H, W).
        coords : torch.Tensor
            Nx2 array of point coordinates (unchanged).
        target_map : torch.Tensor, optional
            Target for training. If None, only image and coords are 
            returned.

        Returns
        -------
        image : torch.Tensor
            Padded image tensor of shape (C, H_new, W_new),
            where H_new and W_new are divisible by 32.
        coords : torch.Tensor
            Unchanged point coordinates.
        target_map : torch.Tensor
            Padded target map of shape (H_new, W_new). Only returned if a 
            target_map was provided as input.
        """

        _, H, W = image.shape

        # Compute next multiple of 32 for height and width
        new_H = (H + 31) // 32 * 32
        new_W = (W + 31) // 32 * 32

        pad_h = new_H - H
        pad_w = new_W - W

        # Pad format: (left, right, top, bottom)
        image = F.pad(
            image,
            (0, pad_w, 0, pad_h),
            mode="constant",
            value=0
        )

        if target_map is not None:
            target_map = F.pad(target_map, 
                               (0, pad_w, 0, pad_h), 
                               mode="constant", 
                               value=0
                               )
            return image, coords, target_map
        else:
            return image, coords
    
class Normalize:
    """
    Normalize image tensor with mean and standard deviation.

    Parameters
    ----------
    mean : list of float
        Per-channel mean values (e.g. [0.485, 0.456, 0.406] for ImageNet).
    std : list of float
        Per-channel standard deviation values (e.g. [0.229, 0.224, 0.225] for ImageNet).

    Methods
    -------
    __call__(image, coords, target_map=None)
        Normalize image, pass coordinates unchanged, and optionally a
        target map.
    """
    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self,
                 image: torch.Tensor,
                 coords: torch.Tensor,
                 target_map: torch.Tensor = None
                 ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize image and pass coordinates through unchanged.

        Parameters
        ----------
        image : torch.Tensor
            Input image of shape (C, H, W), expected to be in [0, 1].
        coords : torch.Tensor
            Nx2 tensor of point coordinates.
        target_map : torch.Tensor, optional
            Target for training. If None, only image and coords are 
            returned.

        Returns
        -------
        image : torch.Tensor
            Normalized image.
        coords : torch.Tensor
            Unchanged point coordinates.
        target_map : torch.Tensor
            Unchanged target map. Only returned if a target_map was
            provided as input.
        """
        image = (image - self.mean) / self.std

        return (image, coords, target_map) if target_map is not None else (image, coords)