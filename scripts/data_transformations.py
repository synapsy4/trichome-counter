
"""
Custom transformation classes to transform images and coordinates.
"""
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
    __call__(image, coords)
        Apply all transforms in sequence to image and coordinates.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, coords):
        """
        Apply all transforms sequentially.

        Parameters
        ----------
        image : torch.Tensor
            Input image.
        coords : torch.Tensor
            Nx2 array of point coordinates.

        Returns
        -------
        image : torch.Tensor
            Transformed image.
        coords : torch.Tensor
            Transformed point coordinates.
        """
        # Apply transforms sequentially
        for t in self.transforms:
            image, coords = t(image, coords)
        return image, coords
    

class ResizeShortSide:
    """
    Resize image and scale coordinates so the shorter side matches target length.

    Parameters
    ----------
    short_side : int
        Target length for the shorter side of the image.

    Methods
    -------
    __call__(image, coords)
        Resize image and scale point coordinates proportionally.
    """
    def __init__(self, short_side):
        self.short_side = short_side

    def __call__(self, image, coords):
        """
        Resize image and scale coordinates.

        Parameters
        ----------
        image : torch.Tensor
            Input image.
        coords : torch.Tensor
            Nx2 array of point coordinates.

        Returns
        -------
        image : torch.Tensor
            Resized image.
        coords : torch.Tensor
            Scaled point coordinates.
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
    __call__(image, coords)
        Randomly crop image and adjust point coordinates.

    Raises
    ------
    RuntimeError
        If crop size is larger than image dimensions.
    """
    def __init__(self, crop_w, crop_h):
        self.crop_w = crop_w
        self.crop_h = crop_h

    def __call__(self, image, coords):
        """
        Apply random crop to image and coordinates.

        Parameters
        ----------
        image : torch.Tensor
            Input image.
        coords : torch.Tensor
            Nx2 array of point coordinates.

        Returns
        -------
        image : torch.Tensor
            Cropped image.
        coords : torch.Tensor
            Point coordinates within crop bounds, adjusted to crop origin.

        Raises
        ------
        RuntimeError
            If crop size exceeds image dimensions.
        """
        _, H, W = image.shape

        # Check if crop size is within image size
        if W < self.crop_w or H < self.crop_h:
            raise RuntimeError("Crop size larger than image.")

        # Get random upper left corner coordinates
        y0 = torch.randint(0, H - self.crop_h + 1, (1,)).item()
        x0 = torch.randint(0, W - self.crop_w + 1, (1,)).item()

        # Copy image and apply crop to copy
        image = image.clone()
        image = image[:,y0:y0+self.crop_h,x0:x0+self.crop_w]

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

            return image, coords[mask]
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
    __call__(image, coords)
        Apply random horizontal flip to image and coordinates.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, coords):
        """
        Apply random horizontal flip.

        Parameters
        ----------
        image : torch.Tensor
            Input image.
        coords : torch.Tensor
            Nx2 array of point coordinates.

        Returns
        -------
        image : torch.Tensor
            Flipped or original image.
        coords : torch.Tensor
            Flipped or original point coordinates.
        """
        # By chance return data without flip
        if torch.rand(1).item() > self.p:
            return image, coords

        # Flip image
        _, _, W = image.shape
        image = torch.flip(image, dims=[2])

        # If coordinates exist, flip them
        if len(coords) > 0:
            coords = coords.clone()
            coords[:, 0] = W - 1 - coords[:, 0]

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
    __call__(image, coords)
        Apply random vertical flip to image and coordinates.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, coords):
        """
        Apply random vertical flip.

        Parameters
        ----------
        image : torch.Tensor
            Input image.
        coords : torch.Tensor
            Nx2 array of point coordinates.

        Returns
        -------
        image : torch.Tensor
            Flipped or original image.
        coords : torch.Tensor
            Flipped or original point coordinates.
        """
        # By chance return data without flip
        if torch.rand(1).item() > self.p:
            return image, coords
        
        # Flip image along vertical axis (=0)
        _, H, _ = image.shape
        image = torch.flip(image, dims=[1])

        # If coordinates exist, flip them
        if len(coords) > 0:
            coords = coords.clone()
            coords[:, 1] = H - 1 - coords[:, 1]

        return image, coords