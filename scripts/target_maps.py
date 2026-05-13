"""
Functions to generate and visualize target maps based on trichome coordinates.
"""

from typing import Any

import torch
import matplotlib
import matplotlib.pyplot as plt


def generate_density_map(coords: torch.Tensor, 
                         H: int, 
                         W: int, 
                         sigma: float, 
                         *target_map_args: Any, 
                         **target_map_kwargs: Any
                         ) -> torch.Tensor:
    """
    Generate a 2D Gaussian density map from point coordinates.

    Creates a density map by placing a normalized 2D Gaussian kernel at each
    coordinate location. The density map represents the spatial distribution
    of points as a continuous heatmap.

    Parameters
    ----------
    coords : torch.Tensor
        Nx2 tensor of (x, y) coordinates.
    H : int
        Height of the output density map.
    W : int
        Width of the output density map.
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    density : torch.Tensor
        HxW density map with accumulated Gaussian kernels.
    """
    # Ensure device consistency
    device = coords.device

    # Initialize empty density map
    density = torch.zeros((H, W), dtype=torch.float32, device=device)

    # Return empty map if no coordinates provided
    if len(coords) == 0:
        return density

    # Radius for local windo creation (3 sigma rule)
    radius = int(3 * sigma)

    # Add Gaussian kernel for each coordinate
    for x, y in coords:

        # Define local window
        x0 = max(0, int(x) - radius)
        x1 = min(W, int(x) + radius + 1)
        y0 = max(0, int(y) - radius)
        y1 = min(H, int(y) + radius + 1)

        if x0 >= x1 or y0 >= y1:
            continue

        # Create local coordinate grids
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=device),
            torch.arange(x0, x1, device=device),
            indexing="ij"
        )

        # Compute 2D Gaussian centered at (x, y)
        g = torch.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)
        )

        # Normalize s.t. mass reflects trichome count
        g /= g.sum()

        # Accumulate into density map
        density[y0:y1, x0:x1] += g

    return density


def generate_density_map_adaptive(coords: torch.Tensor, 
                                  H: int, 
                                  W: int, 
                                  k: int = 3, 
                                  beta: float = 0.4, 
                                  sigma_min: float = 2.0, 
                                  sigma_max: float = 12.0, 
                                  *target_map_args: Any, 
                                  **target_map_kwargs: Any
                                  ) -> torch.Tensor:
    """
    Generate adaptive Gaussian density map using kNN-based sigma.

    Parameters
    ----------
    coords : torch.Tensor (Nx2)
        (x, y) coordinates
    H, W : int
        Output dimensions
    k : int
        Number of nearest neighbors
    beta : float
        Scaling factor for sigma
    sigma_min, sigma_max : float
        Clipping bounds for sigma

    Returns
    -------
    density : torch.Tensor (H x W)
    """
    # Ensure device consistency
    device = coords.device

    # Initialize empty density map
    density = torch.zeros((H, W), dtype=torch.float32, device=device)

    # Return empty map if no coordinates provided
    if len(coords) == 0:
        return density
    # Lower k for images with very few trichome labels
    elif k > len(coords):
        k = len(coords)

    # Compute pairwise distances
    dists = torch.cdist(coords, coords, p=2)

    # Replace self-distance with large number
    dists.fill_diagonal_(float("inf"))

    # Compute kNN mean distance per point
    knn_dists, _ = torch.topk(dists, k, largest=False)
    mean_knn = knn_dists.mean(dim=1)

    # Compute adaptive sigmas
    sigmas = beta * mean_knn
    sigmas = torch.clamp(sigmas, sigma_min, sigma_max)

    for idx, (x, y) in enumerate(coords):
        sigma = sigmas[idx]

        # Define local window
        radius = int(3 * sigma)
        x0 = max(0, int(x) - radius)
        x1 = min(W, int(x) + radius + 1)
        y0 = max(0, int(y) - radius)
        y1 = min(H, int(y) + radius + 1)

        if x0 >= x1 or y0 >= y1:
            continue

        # Create local coordinate grids
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=device),
            torch.arange(x0, x1, device=device),
            indexing="ij"
        )

        g = torch.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)
        )

        # Normalize s.t. mass reflects trichome count
        g /= g.sum()  

        density[y0:y1, x0:x1] += g

    return density


def plot_density_map(img: torch.Tesnor, 
                     density: torch.Tesnor, 
                     alpha: float = 0.5, 
                     title: str = "", 
                     ax: matplotlib.axes.Axes = None
                     ) -> None:
    """
    Visualize a density map overlaid on an image.

    Parameters
    ----------
    img : torch.Tensor
        RGB image to display as background.
    density : torch.Tensor
        2D density map to overlay using jet colormap.
    alpha : float, optional
        Transparency level for density overlay (default: 0.5).
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, creates a new figure with
        size (10, 5). Default is None.
    """
    # If no axes provided, create one
    if ax is None:
        _, ax = plt.subplots(figsize=(10,5))
    # Plot image
    ax.imshow(img)
    # Overlay density map with transparency
    ax.imshow(density, cmap="jet", alpha=alpha)
    #plt.colorbar(label="Density")
    # Set title and turn off axis
    ax.set_title(title)
    ax.axis("off")