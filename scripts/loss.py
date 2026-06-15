"""
Custom loss function classes.
"""

import torch
import torch.nn as nn


class DensityCountLoss(nn.Module):
    """
    Combined Density + Count loss.

    L_total = MSE(density_map) + lambda_count * L1(total_count)

    - MSE encourages spatially correct density distribution
    - L1 count loss directly optimizes total object count
    """

    def __init__(self, lambda_count: float = 0.5):
        super().__init__()

        # Weight for count loss contribution
        self.lambda_count = lambda_count

        # Pixel-wise regression loss
        self.mse = nn.MSELoss()

        # Absolute error for count
        self.l1 = nn.L1Loss()

    def forward(self, 
                pred_density: torch.Tensor, 
                gt_density: torch.Tensor,
                gt_coords: tuple[torch.Tensor, ...],
                pred_count: torch.Tensor | None
                ) -> dict[str, torch.Tensor | None]:
        """
        Compute combined loss.

        Parameters
        ----------
        pred_density : torch.Tensor
            Predicted density map (B, 1, H, W)
        gt_density : torch.Tensor
            Ground truth density map (B, H, W) 
        gt_coords : tuple[torch.Tensor, ...]
            Ground truth trichome coordinate tensors.
        pred_count : torch.Tensor | None
            Predicted trichome counts.

        Returns
        -------
        dict
            total_loss : torch.Tensor
                Combined scalar loss
            loss_density : torch.Tensor
                Pixel-wise MSE (detached for logging)
            loss_count : torch.Tensor
                Count L1 loss (detached for logging)
        """

        # Pixel-wise density regression loss
        loss_density = self.mse(pred_density, gt_density)

        # Compute total predicted counts & gt counts
        pred_count = pred_density.sum(dim=[1, 2, 3]) if pred_count is None else pred_count
        gt_count = torch.tensor([len(coords) for coords in gt_coords], 
                         dtype=pred_count.dtype, device=pred_count.device)

        # Count-level loss (robust to outliers)
        loss_count = self.l1(pred_count, gt_count)

        # Final weighted loss
        total_loss = loss_density + self.lambda_count * loss_count

        return {"total_loss": total_loss, 
                "loss_density": loss_density.detach(), 
                "loss_count": loss_count.detach()}
    

class PointMassAllocationLoss(nn.Module):
    """
    Combined Likelihood + Count loss.

    L_total = Likelihood(density_map, gt_coords) + lambda_count * L1(pred_count)

    - Likelihood directly maximizes the likelihood that each gt count is 
      "covered" by the predicted density
    - L1 count loss directly optimizes total object count
    """

    def __init__(self, lambda_count: float = 0.5):
        super().__init__()

        # Weight for count loss contribution
        self.lambda_count = lambda_count

        # Absolute error for count
        self.l1 = nn.L1Loss()

    def forward(self, 
                pred_density: torch.Tensor, 
                gt_density: torch.Tensor,
                gt_coords: tuple[torch.Tensor, ...],
                pred_count: torch.Tensor | None = None
                ) -> dict[str, torch.Tensor | None]:
        """
        Compute combined loss.

        Parameters
        ----------
        pred_density : torch.Tensor
            Predicted density map (B, 1, H, W)
        gt_density : torch.Tensor
            Ground truth density map (B, H, W) 
        gt_coords : tuple[torch.Tensor, ...]
            Ground truth trichome coordinate tensors.
        pred_count : torch.Tensor | None
            Predicted trichome counts.
            Default is None as it is calculated from pred_density.

        Returns
        -------
        dict
            total_loss : torch.Tensor
                Combined scalar loss
            loss_density : torch.Tensor
                Bayesian loss (detached for logging)
            loss_count : torch.Tensor
                Count L1 loss (detached for logging)
        """

        # Bayesian loss
        loss_density = self.pma_loss(pred_density, gt_coords)

        # Compute total predicted counts & gt counts
        pred_count = pred_density.sum(dim=[1, 2, 3]) if pred_count is None else pred_count
        gt_count = torch.tensor([len(coords) for coords in gt_coords], 
                         dtype=pred_count.dtype, device=pred_count.device)

        # Count-level loss
        normalizer = (gt_count + 1)
        loss_count_norm = ((pred_count - gt_count).abs() / normalizer).mean()
        loss_count = self.l1(pred_count, gt_count)
        

        # Final weighted loss 
        total_loss = loss_density + self.lambda_count * loss_count_norm

        return {"total_loss": total_loss, 
                "loss_density": loss_density.detach(), 
                "loss_count": loss_count.detach()}
    
    def pma_loss(self,
                      pred_density: torch.Tensor, 
                      gt_coords: tuple[torch.Tensor, ...], 
                      sigma: float = 8.0
                      ) -> torch.Tensor:
        """
        point_annotations: list of (N_i, 2) tensors of trichome locations per image
        Computes loss directly from point locations, not GT density map.
        TODO: complete docstring
        """
        loss = pred_density.new_tensor(0.0)
        total_trichomes = 0

        for b, coords in enumerate(gt_coords):
            if len(coords) == 0:
                continue

            pred = pred_density[b, 0]  # (H, W)

            y_grid, x_grid = torch.meshgrid(
                    torch.arange(pred.shape[0], device=pred.device),
                    torch.arange(pred.shape[1], device=pred.device),
                    indexing='ij'
                )
            
            # For each coordinate, integrate predicted density in its neighborhood
            for (x, y) in coords:
                # Gaussian expectation under the predicted density
                gauss = torch.exp(-((y_grid - y)**2 + (x_grid - x)**2) / (2 * sigma**2))
                gauss = gauss / gauss.sum()
                expected = (pred * gauss).sum()
                loss += torch.abs(expected - 1.0)
                total_trichomes += 1

        return loss / max(total_trichomes, 1)



class BayesianDensityCountLoss(nn.Module):
    """
    Bayesian loss (Ma et al., "Bayesian Loss for Crowd Count Estimation
    with Point Supervision", ICCV 2019), combined with a direct count loss.

    For each ground-truth point y_m, defines an (unnormalized) likelihood
    that pixel x belongs to it: p(x|y_m) = exp(-||x - y_m||^2 / (2*sigma^2)).
    A background likelihood p(x|y_0) = 1 - max_m p(x|y_m) captures pixels
    far from all points. Normalizing these per pixel gives a posterior
    p(y_m|x): a soft assignment of each pixel to an instance or background.

    The predicted density is then expected to sum to 1 under each
    instance's posterior, and to 0 under the background posterior.

    Note: `gt_density` is unused here -- this loss is computed directly
    from the point annotations, so no target density map needs to be
    constructed at all.
    """

    def __init__(self, lambda_count: float = 0.5, sigma: float = 8.0):
        super().__init__()
        self.lambda_count = lambda_count
        self.sigma = sigma
        self.l1 = nn.L1Loss()

    def _per_image_loss(self, density_map: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        density_map : (H, W) predicted density for one image
        points : (M, 2) tensor of (row, col) point coordinates; may have M=0
        """
        H, W = density_map.shape
        device, dtype = density_map.device, density_map.dtype

        if points.numel() == 0:
            # No annotated objects: every pixel is background,
            # expected background count = total predicted mass, target 0.
            return density_map.sum().abs()

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )

        points = points.to(device=device, dtype=dtype)  # (M, 2), assumed (row, col)
        dy = ys.unsqueeze(0) - points[:, 0].view(-1, 1, 1)
        dx = xs.unsqueeze(0) - points[:, 1].view(-1, 1, 1)
        sq_dist = dy ** 2 + dx ** 2  # (M, H, W)

        fg_likelihood = torch.exp(-sq_dist / (2 * self.sigma ** 2))  # (M, H, W)
        bg_likelihood = (1.0 - fg_likelihood.max(dim=0).values).clamp_min(0.0).unsqueeze(0)  # (1, H, W)

        likelihood = torch.cat([bg_likelihood, fg_likelihood], dim=0)  # (M+1, H, W)
        posterior = likelihood / likelihood.sum(dim=0, keepdim=True).clamp_min(1e-8)

        expected_counts = (posterior * density_map.unsqueeze(0)).sum(dim=(1, 2))  # (M+1,)
        targets = torch.ones_like(expected_counts)
        targets[0] = 0.0  # background

        return self.l1(expected_counts, targets)

    def forward(self,
                 pred_density: torch.Tensor,
                 gt_density: torch.Tensor,
                 gt_coords: tuple[torch.Tensor, ...],
                 pred_count: torch.Tensor | None
                 ) -> dict[str, torch.Tensor | None]:
        density_maps = pred_density.squeeze(1) if pred_density.dim() == 4 else pred_density  # (B, H, W)

        loss_density = torch.stack([
            self._per_image_loss(dmap, pts) for dmap, pts in zip(density_maps, gt_coords)
        ]).mean()

        pred_count = pred_density.sum(dim=[1, 2, 3]) if pred_count is None else pred_count
        gt_count = torch.tensor([len(coords) for coords in gt_coords],
                                 dtype=pred_count.dtype, device=pred_count.device)
        loss_count = self.l1(pred_count, gt_count)

        total_loss = loss_density + self.lambda_count * loss_count

        return {"total_loss": total_loss,
                "loss_density": loss_density.detach(),
                "loss_count": loss_count.detach()}
