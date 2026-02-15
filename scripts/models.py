"""
Contains PyTorch model code to instantiate models.
"""

import torch.nn as nn
import segmentation_models_pytorch as smp


class DensityModel(nn.Module):
    """
    U-Net based density regression model.

    Uses a ResNet34 encoder (ImageNet pretrained) and outputs
    a single-channel density map. ReLU activation ensures
    non-negative density predictions.
    """
    def __init__(self):
        super().__init__()

        # Base U-Net model for fully convolutional regression
        # - encoder: ResNet34 backbone
        # - classes=1: single output channel (density map)
        # - activation=None: raw regression output
        self.base_model = smp.Unet(
                                encoder_name="resnet34",
                                encoder_weights="imagenet",
                                in_channels=3,
                                classes=1,          
                                activation=None,   
                            )
        # Enforce non-negative density values
        # Density maps represent counts → must be >= 0
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images (B, 3, H, W)

        Returns
        -------
        torch.Tensor
            Predicted density maps (B, 1, H, W)
        """
        return self.relu(self.base_model(x))