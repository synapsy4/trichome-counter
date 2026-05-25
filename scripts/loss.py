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

        # Robust absolute error for count
        self.l1 = nn.L1Loss()

    def forward(self, 
                pred_density: torch.Tensor, 
                gt_density: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Parameters
        ----------
        pred_density : torch.Tensor
            Predicted density map (B, 1, H, W)
        gt_density : torch.Tensor
            Ground truth density map (B, H, W)

        Returns
        -------
        total_loss : torch.Tensor
            Combined scalar loss
        loss_density : torch.Tensor
            Pixel-wise MSE (detached for logging)
        loss_count : torch.Tensor
            Count L1 loss (detached for logging)
        """

        # Pixel-wise density regression loss
        loss_density = self.mse(pred_density, gt_density)

        # Compute total predicted and ground truth counts
        pred_count = pred_density.sum(dim=[1, 2, 3])
        gt_count = gt_density.sum(dim=[1, 2, 3])

        # Count-level loss (robust to outliers)
        loss_count = self.l1(pred_count, gt_count)

        # Final weighted loss
        total_loss = loss_density + self.lambda_count * loss_count

        return total_loss, loss_density.detach(), loss_count.detach()
