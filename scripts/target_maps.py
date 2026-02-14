"""
Functions to generate and visualize target maps based on trichome coordinates.
"""

import torch
import matplotlib.pyplot as plt


def generate_density_map(coords, H, W, sigma):
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
    # Initialize empty density map
    density = torch.zeros((H, W), dtype=torch.float32)

    # Return empty map if no coordinates provided
    if len(coords) == 0:
        return density

    # Create coordinate grids for entire image
    yy, xx = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing="ij"
    )

    # Add Gaussian kernel for each coordinate
    for x, y in coords:
        # Compute 2D Gaussian centered at (x, y)
        g = torch.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)
        )
        # Normalize
        g_sum = g.sum()
        g /= g_sum 
        # Accumulate into density map
        density += g

    return density


def plot_density_map(img, density, alpha=0.5, title="", ax=None):
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
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, creates a new figure with
        size (10, 5). Default is None.
    """
    # If no axes provided, create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,5))
    # Plot image
    ax.imshow(img)
    # Overlay density map with transparency
    ax.imshow(density, cmap="jet", alpha=alpha)
    #plt.colorbar(label="Density")
    # Set title and turn off axis
    ax.set_title(title)
    ax.axis("off")