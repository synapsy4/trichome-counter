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
                Likelihood loss (detached for logging)
            loss_count : torch.Tensor
                Count L1 loss (detached for logging)
        """

        # PMA loss
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
