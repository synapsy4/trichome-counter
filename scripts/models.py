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
    a single-channel density map.

    Parameters
    ----------
    activation : {"ReLU", "ReLUTanh", "Sigmoid"}, optional
        Last layer activation function, default "ReLU".

    Raises
    ------
    ValueError
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
            raise ValueError(f"Activation function type {activation} not known.")
        

    def forward(self, x: torch.Tensor
                ) -> dict[str, torch.Tensor | None]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images (B, 3, H, W)

        Returns
        -------
        dict
            density : torch.Tensor
                Predicted density maps (B, 1, H, W)
            count : None
                None as count is not calculated here
        """
        return {"density": self.activation(self.base_model(x)),
                "count": None}



class DensityCountModel(nn.Module):
    """
    U-Net based density regression model with explicit count head.

    Uses a ResNet34 encoder (ImageNet pretrained) and outputs
    a single-channel density map as well as a count.

    Parameters
    ----------
    activation : {"ReLU", "ReLUTanh", "Sigmoid"}, optional
        Last layer activation function, default "ReLU".

    Raises
    ------
    ValueError
        If the activation type is not implemented.

    """
    def __init__(self, activation: str = "ReLU") -> None:
        super().__init__()

        # Base U-Net model for fully convolutional regression
        # - encoder: ResNet34 backbone
        # - classes=1: single output channel (density map)
        # - activation=None: raw regression output
        base_model = smp.Unet(
                            encoder_name="resnet34",
                            encoder_weights="imagenet",
                            in_channels=3,
                            classes=1,          
                            activation=None,   
                        )
        
        # Split backbone into submodules
        self.encoder = base_model.encoder # ResNet34
        self.decoder = base_model.decoder # U-Net
        self.seg_head = base_model.segmentation_head

        # Bottleneck = last encoder stage, 512 channels for resnet34
        BOTTLENECK_CH = 512
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),              # (B, 512, 4, 4)
            nn.Flatten(),                          # (B, 512*16)
            nn.Linear(BOTTLENECK_CH * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.ReLU()
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
            raise ValueError(f"Activation function type {activation} not known.")
        

    def forward(self, x: torch.Tensor
                ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images (B, 3, H, W)

        Returns
        -------
        dict
            density : torch.Tensor
                Predicted density maps (B, 1, H, W)
            count : torch.Tensor
                Predicted count (B,)
        """
        # Get list of features
        features = self.encoder(x) #(B, 512, H/32, W/32)

        # Count head on bottleneck 
        count = self.count_head(features[-1]).squeeze(-1)   # (B,)

        # Decode 
        decoder_out = self.decoder(features)               # (B, 16, H, W)
        density_raw = self.seg_head(decoder_out)            # (B, 1, H, W)

        return {"density": self.activation(density_raw),
                "count": count}