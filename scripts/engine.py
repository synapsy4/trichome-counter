"""
Train, validate and inference functions.
"""

from typing import Any 

from tqdm.auto import tqdm
import torch

from scripts import utils

def train_one_epoch(model: torch.nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    criterion: torch.nn.Module, 
                    device: torch.device
                    ) -> tuple[float, float]:
    """
    Train model for one epoch.

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

    for images, gt_density, _ in dataloader:

        # Move batch to GPU/CPU
        images = images.to(device)
        gt_density = gt_density.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_density = model(images)

        # Compute loss
        loss, _, _ = criterion(pred_density, gt_density)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss+= loss.item()

        # Monitor count-level MAE
        pred_count = pred_density.sum(dim=[1, 2, 3])
        gt_count = gt_density.sum(dim=[1, 2, 3])

        mae = torch.abs(pred_count - gt_count).mean()
        total_mae += mae.item()

    return (
        total_loss / len(dataloader),
        total_mae / len(dataloader)
    )


def validate(model: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             criterion: torch.nn.Module, 
             device: torch.device
             ) -> tuple[float, float, list[torch.Tensor], list[torch.Tensor]]:
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
        Device to run computations on (e.g., "cuda" or "cpu").
    
    Returns
    -------
    avg_loss : float
        Average loss over all batches in the dataset.
    avg_mae : float
        Average count-level mean absolute error (MAE) over all batches.
    gt_counts : list of torch.Tensor
        List of total counts per batch computed from ground truth density maps.
    pred_counts : list of torch.Tensor
        List of total counts per batch computed from predicted density maps.
    """

    model.eval()

    # Init return variables
    total_loss = 0.0
    total_mae = 0.0
    gt_counts = []
    pred_counts = []

    # Loop through batches
    with torch.inference_mode():
        for images, gt_density, _ in dataloader:

            images = images.to(device)
            gt_density = gt_density.to(device)

            # Forward pass
            pred_density = model(images)

            # Compute loss
            loss, _, _ = criterion(pred_density, gt_density)

            total_loss+= loss.item()

            # Monitor count-level MAE
            pred_count = pred_density.sum(dim=[1, 2, 3])
            gt_count = gt_density.sum(dim=[1, 2, 3])

            mae = torch.abs(pred_count - gt_count).mean()
            total_mae += mae.item()

            # Update counts
            gt_counts.extend(gt_count)
            pred_counts.extend(pred_count)

    return (
        total_loss / len(dataloader),
        total_mae / len(dataloader),
        gt_counts,
        pred_counts
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
    best and last model checkpoints along with training metadata.

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
    # Things we want to save
    best_epoch = 1
    best_epoch_val_mae = None
    best_model_cp = model.state_dict()
    train_loss_list = []
    train_mae_list = []
    val_loss_list = []
    val_mae_list = []

    n_epochs = cfg["training"]["epochs"]
    
    # Run training loop
    for epoch in tqdm(range(1,n_epochs+1)):

        # Train step
        train_loss, train_mae = train_one_epoch(model=model,
                                                dataloader=train_dataloader,
                                                optimizer=optimizer,
                                                criterion=criterion,
                                                device=device)
        # Validation step
        val_loss, val_mae, _, _ = validate(model=model,
                                        dataloader=val_dataloader,
                                        criterion=criterion,
                                        device=device)
        
        # Update best val epoch if lowest mae reached
        if best_epoch_val_mae is None or val_mae < best_epoch_val_mae:
            best_epoch = epoch
            best_epoch_val_mae = val_mae
            best_model_cp = model.state_dict()
        
        # Store losses and maes
        train_loss_list.append(train_loss)
        train_mae_list.append(train_mae)
        val_loss_list.append(val_loss)
        val_mae_list.append(val_mae)
        
        # Print status
        print(f"Epoch {epoch:03d} | Train-loss={train_loss:.2f}, train-mae={train_mae:.2f} | Val-loss={val_loss:.2f}, val-mae={val_mae:.2f}")

    # Write metric dict
    metrics = {
        "best_epoch": best_epoch,
        "best_epoch_val_mae": best_epoch_val_mae,
        "train_loss_list": train_loss_list,
        "train_mae_list": train_mae_list,
        "val_loss_list": val_loss_list,
        "val_mae_list": val_mae_list
        }

    # Write metadata dict
    metadata = {
        "config": cfg,
        "metrics": metrics
    }


    # Save model
    utils.save_model(model=model,
               model_name=cfg["model"]["model_name"],
               metadata=metadata,
               best_cp=best_model_cp)