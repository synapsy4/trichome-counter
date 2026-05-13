"""
Contains PyTorch model code to instantiate models.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DensityModel(nn.Module):
    """
    U-Net based density regression model.

    Uses a ResNet34 encoder (ImageNet pretrained) and outputs
    a single-channel density map. ReLU activation ensures
    non-negative density predictions.

    Parameters
    ----------
    activation : {"ReLU", "ReLUTanh", "Sigmoid"}, optional
        Last layer activation function, default "ReLU".

    Raises
    ------
    TypeError
        If the activation type is not implemented.

    """
    def __init__(self, activation: str = "ReLU") -> None:
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
        if activation == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "ReLUTanh":
            self.activation = nn.Sequential(nn.ReLU(inplace=True), 
                                            nn.Tanh())
        elif activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        else: 
            raise TypeError(f"Activation function type {activation} not known.")
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.activation(self.base_model(x))