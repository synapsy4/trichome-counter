"""
Train, validate and inference functions.
"""

from typing import Any 

from tqdm.auto import tqdm
import torch

from scripts.logging import TrainingLogger

def train_one_epoch(model: torch.nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    criterion: torch.nn.Module, 
                    device: torch.device,
                    accumulation_steps: int = 1
                    ) -> tuple[float, float]:
    """
    Train model for one epoch.
    TODO: Update docstring

    accumulation_steps : int, optional
        Steps for gradient accumulation. Default is 1 (= no acumulation)

    Returns
    -------
    avg_loss : float
        Average training loss
    avg_mae : float
        Average count MAE
    """

    model.train()

    total_loss = 0.0
    total_mae = 0.0

    for i, batch in enumerate(dataloader):

        # Move batch to GPU/CPU
        images, gt_density, gt_coords = batch
        images = images.to(device)
        gt_density = gt_density.to(device)
        gt_coords = tuple(c.to(device) for c in gt_coords)

        # Forward pass
        pred = model(images)

        # Compute loss
        loss_dict = criterion(pred_density=pred["density"], 
                               gt_density=gt_density,
                               gt_coords=gt_coords,
                               pred_count=pred["count"])
        loss = loss_dict["total_loss"]
        loss_count = loss_dict["loss_count"]

        # Backpropagation
        (loss / accumulation_steps).backward()

        # Update wnbs
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Monitor loss + count-level MAE
        total_loss += loss.item()
        total_mae += loss_count.item()

    # Handle leftover gradients if dataset size isn't divisible by accumulation_steps
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return (
        total_loss / len(dataloader),
        total_mae / len(dataloader)
    )


def validate(model: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             criterion: torch.nn.Module, 
             device: torch.device
             ) -> tuple[float, float]:
    """
    Evaluate a model on a validation or test dataset.
    
    Performs a forward pass on the dataset without computing gradients, 
    computes the loss using the provided criterion, and tracks count-level 
    metrics such as mean absolute error (MAE) for object counts.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model to evaluate.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of validation/test data.
    criterion : callable
        Loss function.
    device : torch.device
        Device on which the model and data are processed ("cpu" or "cuda").
    
    Returns
    -------
    avg_loss : float
        Average loss over all batches in the dataset.
    avg_mae : float
        Average count-level mean absolute error (MAE) over all batches.
    """

    model.eval()

    # Init return variables
    total_loss = 0.0
    total_mae = 0.0

    # Loop through batches
    with torch.inference_mode():
        for images, gt_density, gt_coords in dataloader:

            # Move batch to GPU/CPU
            images = images.to(device)
            gt_density = gt_density.to(device)
            gt_coords = tuple(c.to(device) for c in gt_coords)

            # Forward pass
            pred = model(images)

            # Compute loss
            loss_dict = criterion(pred_density=pred["density"], 
                        gt_density=gt_density,
                        gt_coords=gt_coords,
                        pred_count=pred["count"])
            loss = loss_dict["total_loss"]
            loss_count = loss_dict["loss_count"]

            total_loss += loss.item()

            # Monitor count-level MAE
            total_mae += loss_count.item()

    return (
        total_loss / len(dataloader),
        total_mae / len(dataloader)
    )

def train(model: torch.nn.Module, 
          cfg: dict[str, Any], 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer, 
          criterion: torch.nn.Module, 
          device: torch.device
          ) -> None:
    """
    Train a model for a specified number of epochs and track performance metrics.

    Identifies the best-performing epoch based on validation MAE, and saves the 
    best and last model checkpoints each epoch along with training metadata.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    cfg : dict
        Dictionary with the hyperparameters.
    train_dataloader : torch.utils.data.DataLoader
        DataLoader providing the training dataset.
    val_dataloader : torch.utils.data.DataLoader
        DataLoader providing the validation dataset.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    criterion : callable
        Loss function used to compute training and validation loss.
    device : torch.device
        Device on which the model and data are processed ("cpu" or "cuda").
    """

    # Init writer
    logger = TrainingLogger(cfg=cfg)

    # Get steps for gradient accumulation (set to 1 if non existent)
    accumulation_steps = cfg["training"].get("accu_steps", 1)
    
    # Def progress bar
    n_epochs = cfg["training"]["epochs"]
    pbar = tqdm(range(1,n_epochs+1), desc="Epochs")
    
    for epoch in pbar:

        # Train step
        train_dl = tqdm(train_dataloader, desc=f"Train epoch {epoch}", leave=False)
        train_loss, train_mae = train_one_epoch(model=model,
                                                dataloader=train_dl,
                                                optimizer=optimizer,
                                                criterion=criterion,
                                                device=device,
                                                accumulation_steps=accumulation_steps)
        
        # Validation step
        val_dl = tqdm(val_dataloader, desc=f"Val epoch {epoch}", leave=False)
        val_loss, val_mae = validate(model=model,
                                     dataloader=val_dl,
                                     criterion=criterion,
                                     device=device)

        # Save epoch logs
        metrics = {
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_mae": val_mae
        }
        logger.log_epoch(epoch=epoch,
                         metrics=metrics,
                         model=model,
                         optimizer=optimizer)

        # Print status
        pbar.set_postfix({
            "tr-loss": f"{train_loss:.2f}", "tr-mae": f"{train_mae:.2f}",
            "val-loss": f"{val_loss:.2f}", "val-mae": f"{val_mae:.2f}",
        })

    # Close tb summary writers
    logger.close()
