"""
Functions that create visualizations.
"""

from pathlib import Path
from typing import Any

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from scripts.evaluations import predict_single_image


def plot_image(img: torch.Tensor | np.ndarray,
               coords: torch.Tensor | np.ndarray = None,
               title: str = "", 
               ax: plt.Axes = None
               ) -> None:
    """
    Visualize image + optionally its trichome coordinates.

    Parameters
    ----------
    img : torch.Tensor or numpy.ndarray
        RGB image.
    coords : torch.Tensor or numpy.ndarray, optional
        Nx2 (x, y) coordinates. Default is None.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, creates a new figure with
        size (10, 5). Default is None.
    """

    # If no axes provided, create one
    if ax is None:
        _, ax = plt.subplots(figsize=(10,5))
    
    # Make sure img has the right format
    if isinstance(img, torch.Tensor):
        img_np = img.numpy()
    else: 
        img_np = img

    if img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    
    if img_np.min() < 0 or img_np.max() > 1:
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # Plot image
    ax.imshow(img_np)

    # Plot coordinates
    if coords is not None and len(coords) > 0:
        ax.scatter(coords[:,0], coords[:,1], s=20, c="red", marker="o")  
  
    # Set title and turn off axis
    ax.set_title(title)
    ax.axis("off")


def plot_density_map(density: torch.Tensor, 
                     img: torch.Tensor | np.ndarray = None,
                     alpha: float = 0.5, 
                     title: str = "",
                     cmap: str = "jet", 
                     ax: plt.Axes = None
                     ) -> None:
    """
    Visualize a density map overlaid on an image.

    Parameters
    ----------
    density : torch.Tensor
        2D density map.
    img : torch.Tensor or numpy.ndarray, optional
        RGB image to optionally display as background.
    alpha : float, optional
        Transparency level for density map (default: 0.5).
    title : str, optional
        Plot title.
    cmap : str, optional
        The colormap key, e.g. "jet", "hot", etc. Default is "jet".
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, creates a new figure with
        size (10, 5). Default is None.
    """
    # If no axes provided, create one
    if ax is None:
        _, ax = plt.subplots(figsize=(10,5))
    
    if img is not None:
        # Make sure img has the right format
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
        else: 
            img_np = img

        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        
        if img_np.min() < 0 or img_np.max() > 1:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # Plot image
        ax.imshow(img_np)
        
    if isinstance(density, torch.Tensor):
        density_np = density.numpy()
    else:
        density_np = density
    
    if density_np.shape[0] == 1:
        density_np = np.transpose(density_np, (1, 2, 0))

    # Plot/overlay density map with transparency
    ax.imshow(density_np, cmap=cmap, alpha=alpha)
    
    # Set title and turn off axis
    ax.set_title(title)
    ax.axis("off")


def visualize_test_samples(model: torch.nn.Module,
                           dataset: torch.utils.data.Dataset,
                           device: torch.device,
                           cfg: dict[str, Any],
                           cp: str,
                           n_samples: int = 4,
                           alpha_over: float = 0.5,
                           alpha_raw: float = 1.0,
                           cmap_over: str = "hot",
                           cmap_raw: str = "hot",
                           save_fig: bool = False
                           ) -> None:
    """
    Randomly sample images from a dataset and visualise model predictions
    in a three-panel-per-row grid:
        (A) Original + GT coords
        (B) Densitymap overlay      
        (C) Raw density map

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    dataset : torch.utils.data.Dataset
        Dataset with (img, target_map, coords) entries.
    device : torch.device
        Device to run inference on.
    cfg : dict
        Config dict.
    cp : str
        The model checkpoint id {"best" or "last"}.
    n_samples : int, optional
        Number of random samples to visualise. Default 4.
    alpha_over : float, optional
        Densitymap overlay opacity for panel B.
    alpha_raw : float, optional
        Densitymap overlay opacity for panel C.
    cmap_over : str, optional
        Matplotlib colormap name for the panel B.
    cmap_raw : str, optional
        Matplotlib colormap name for the panel C.
    save_fig : bool, optional
        If true, the figure is saved.
    """

    # Get random sample indices
    n_samples_all = len(dataset)
    indices = np.random.choice(range(n_samples_all), 
                                   size=min(n_samples_all, n_samples),
                                   replace=False)

    fig, axes = plt.subplots(
        nrows=len(indices),
        ncols=3,
        figsize=(14, 4 * len(indices))
    )

    # Ensure axes is always 2D
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):

        # Get sample
        img, _, coords = dataset[idx]
        gt_count = len(coords)

        # Get predictions
        pred_density, pred_count = predict_single_image(
            model=model,
            img=img,
            device=device
            )
        
        # Panel A: original image + GT coords
        ax = axes[row, 0]
        
        plot_image(img=img,
                   coords=coords,
                   title=f"Sample {idx}  |  GT count: {gt_count}",
                   ax=ax)

        # Panel B: Densitymap overlay
        ax = axes[row, 1]

        plot_density_map(density=pred_density,
                         img=img,
                         alpha=alpha_over,
                         title=f"Predicted density | Pred count={pred_count}",
                         cmap=cmap_over,
                         ax=ax)

        # Panel C: Blank density map
        ax = axes[row, 2]

        plot_density_map(density=1-pred_density,
                         alpha=alpha_raw,
                         title="Raw predicted density",
                         cmap=cmap_raw,
                         ax=ax)

    fig.suptitle("Trichome detection — test samples", fontsize=16)
    fig.tight_layout()

    # Save figure to model outputs if specified
    if save_fig:
        save_path = Path(cfg["paths"]["outputs"]) / cfg["model"]["model_name"]
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f"rnd_predictions_cp_{cp}.png"
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight")
        print(f"[INFO] Figure saved to '{save_path}'")


def visualize_test_sample_trichomes(model: torch.nn.Module,
                                    dataset: torch.utils.data.Dataset,
                                    device: torch.device,
                                    cfg: dict[str, Any],
                                    cp: str,
                                    n_samples: int = 4,
                                    cutout_size: int = 100,
                                    alpha_over: float = 0.5,
                                    alpha_raw: float = 1.0,
                                    cmap_over: str = "hot",
                                    cmap_raw: str = "hot",
                                    save_fig: bool = False
                                    ) -> None:
    """
    Randomly sample images that have at least 1 trichome from a dataset, zoom in on a random trichome 
    and visualise model predictions in a three-panel-per-row grid:
        (A) Original + GT coords at zoom region
        (B) Densitymap overlay      
        (C) Raw density map

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    dataset : torch.utils.data.Dataset
        Dataset with (img, target_map, coords) entries.
    device : torch.device
        Device to run inference on.
    cfg : dict
        Config dict.
    cp : str
        The model checkpoint id {"best" or "last"}.
    n_samples : int, optional
        Number of random samples to visualise. Default is 4.
    cutout_size : int, optional
        The sidelength in px of the cutout region around a trichome.
        Default is 100.
    alpha_over : float, optional
        Densitymap overlay opacity for panel B.
    alpha_raw : float, optional
        Densitymap overlay opacity for panel C.
    cmap_over : str, optional
        Matplotlib colormap name for the panel B.
    cmap_raw : str, optional
        Matplotlib colormap name for the panel C.
    save_fig : bool, optional
        If true, the figure is saved.
    """

    # Get random sample indices
    n_samples_all = len(dataset)
    indices = []
    tolerance = 20
    fails = 0
    while len(indices) < n_samples and fails < tolerance:
        idx = np.random.randint(0, n_samples_all)
        _, _, coords = dataset[idx]
        if idx in indices or len(coords) == 0:
            fails += 1
        else:
            fails = 0
            indices.append(idx)
 
    fig, axes = plt.subplots(
        nrows=len(indices),
        ncols=3,
        figsize=(14, 4 * len(indices))
    )

    # Ensure axes is always 2D
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):

        # Get sample
        img, _, coords = dataset[idx]
        gt_count = len(coords)

        # Get predictions
        pred_density, pred_count = predict_single_image(
            model=model,
            img=img,
            device=device
            )
        
        # Cutout region around a random trichome
        coord_idx = np.random.choice(len(coords))
        x, y = coords[coord_idx]
        _, H, W = img.shape
        lower_x = int(max(0, min(x - cutout_size // 2, W - cutout_size)))
        lower_y = int(max(0, min(y - cutout_size // 2, H - cutout_size)))
        cutout_img = img[:, lower_y:lower_y+cutout_size, lower_x:lower_x+cutout_size]
        cutout_pred_density = pred_density[lower_y:lower_y+cutout_size, lower_x:lower_x+cutout_size]
        
        # Panel A: original image + GT coord
        ax = axes[row, 0]
        
        # Select all coords within the cutout region and shift to local coordinates
        mask = (
            (coords[:, 0] >= lower_x) & (coords[:, 0] < lower_x + cutout_size) &
            (coords[:, 1] >= lower_y) & (coords[:, 1] < lower_y + cutout_size)
        )
        local_coords = coords[mask] - np.array([lower_x, lower_y])
        plot_image(img=cutout_img,
                   coords=local_coords,
                   title=f"Sample {idx}  |  GT count: {gt_count}",
                   ax=ax)

        # Panel B: Densitymap overlay
        ax = axes[row, 1]

        plot_density_map(density=cutout_pred_density,
                         img=cutout_img,
                         alpha=alpha_over,
                         title=f"Predicted density | Pred count={pred_count}",
                         cmap=cmap_over,
                         ax=ax)

        # Panel C: Blank density map
        ax = axes[row, 2]
        plot_density_map(density=cutout_pred_density,
                         alpha=alpha_raw,
                         title="Raw predicted density",
                         cmap=cmap_raw,
                         ax=ax)

    fig.suptitle("Trichome detection — test samples — zoomed on trichomes", fontsize=16, y=1.01)
    fig.tight_layout()

    # Save figure to model outputs if specified
    if save_fig:
        save_path = Path(cfg["paths"]["outputs"]) / cfg["model"]["model_name"]
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f"rnd_trichome_predictions_cp_{cp}.png"
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight")
        print(f"[INFO] Figure saved to '{save_path}'")


def plot_error_distribution(eval_results: dict,
                            cfg: dict,
                            cp: str,
                            save_fig: bool = False
                            ) -> None:
    """
    Produce a two-panel diagnostic figure from the output of evaluate_on_testset:
        | (A) Scatter: predicted vs GT | (B) Signed error histogram + KDE |
 
    Panel A includes the identity line (perfect prediction) and a linear
    regression fit to reveal systematic bias or scaling errors.
    Panel B shows the distribution of signed errors (pred - GT) with a KDE
    curve, a zero-reference line, and a mean-error marker.
 
    Parameters
    ----------
    eval_results : dict
        Dictionary returned by evaluate_on_testset. Must contain keys:
        "gt_counts", "pred_counts", "mae_sum", "rmse_sum", "me_sum".
    cfg : dict
        Config dict.
    cp : str
        The model checkpoint id {"best" or "last"}.
    save_fig : bool, optional
        If true, the figure is saved.
    """
 
    gt = eval_results["gt_counts"]
    pred = eval_results["pred_counts"]
    errors = pred - gt
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    ## Panel A: Predicted vs GT scatter
    ax = axes[0]
    color = "red"
    lim = (min(gt.min(), pred.min()) * 0.95,
           max(gt.max(), pred.max()) * 1.05)
 
    ax.scatter(gt, pred, alpha=0.5, s=25, color=color, edgecolors="none")
 
    # Identity line (perfect prediction)
    ax.plot(lim, lim, "k--", linewidth=1, label="Perfect prediction")
 
    # Linear regression fit
    m, b = np.polyfit(gt, pred, 1)
    x_fit = np.linspace(lim[0], lim[1], 200)
    ax.plot(x_fit, m * x_fit + b, color=color, linewidth=1.5,
            label=f"Fit: y = {m:.2f}x {b:+.1f}")
 
    # Metric annotation
    ax.text(0.04, 0.95,
            f"MAE  = {eval_results['mae_sum']:.2f}\n"
            f"RMSE = {eval_results['rmse_sum']:.2f}\n"
            f"Bias = {eval_results['me_sum']:+.2f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
 
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("GT count", fontsize=9)
    ax.set_ylabel("Predicted count", fontsize=9)
    ax.set_title("Predicted vs GT", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
 
    ## Panel B: Signed error histogram + KDE
    ax = axes[1]
    color = "blue"
    ax.hist(errors, bins=30, color=color, alpha=0.6, edgecolor="white", density=True, label="Error dist.")
 
    if errors.std() > 0:
        kde = gaussian_kde(errors, bw_method="scott")
        x_kde = np.linspace(errors.min(), errors.max(), 300)
        ax.plot(x_kde, kde(x_kde), color=color, linewidth=2, label="KDE")
 
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Zero error")
    ax.axvline(errors.mean(), color="darkorange", linestyle=":", linewidth=1.5, 
               label=f"Mean = {errors.mean():+.2f}")
 
    ax.set_xlabel("Signed error  (pred − GT)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Error distribution", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
 
    fig.suptitle("Trichome count — error analysis", fontsize=13)
    fig.tight_layout()
 
    # Save figure to model outputs if specified
    if save_fig:
        save_path = Path(cfg["paths"]["outputs"]) / cfg["model"]["model_name"]
        save_path.mkdir(parents=True, exist_ok=True)
        #save_path = save_path / f"error_distribution_cp_{cp}.png"
        save_path = save_path / f"error_distribution_cp_{cp}.pdf"
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight")
        print(f"[INFO] Figure saved to '{save_path}'")