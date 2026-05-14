"""
Training of a trichome counter model.
"""
import torch

from scripts.data import get_dataloader
from scripts.engine import train
from scripts.utils import load_config, init_model, init_loss


if __name__ == "__main__":

    # Load config
    cfg = load_config()

    # Create dataloaders
    train_dataloader = get_dataloader(split="train", cfg=cfg)
    val_dataloader = get_dataloader(split="val", cfg=cfg)
   
    # Init model
    model = init_model(model_name=cfg["model"]["model_name"], 
                       model_type=cfg["model"]["model_type"], 
                       activation=cfg["model"]["activation"]) 

    # Init loss
    criterion = init_loss(cfg)

    # Init optimizer
    optimizer = torch.optim.AdamW( 
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Send model to device
    model.to(device)

    # Train model
    train(model=model,
          cfg=cfg,
          train_dataloader=train_dataloader, 
          val_dataloader=val_dataloader, 
          optimizer=optimizer, 
          criterion=criterion, 
          device=device)
