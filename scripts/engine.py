"""
Train, validate and inference functions.
"""

from typing import Any 

from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

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

    for images, gt_density, gt_coords in dataloader:

        # Move batch to GPU/CPU
        images = images.to(device)
        gt_density = gt_density.to(device)
        gt_coords = tuple(c.to(device) for c in gt_coords)

        optimizer.zero_grad()

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
        loss.backward()
        optimizer.step()

        # Monitor loss + count-level MAE
        total_loss += loss.item()
        total_mae += loss_count.item()

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

    # Init writers
    model_name = cfg["model"]["model_name"]
    writer = SummaryWriter(log_dir=f"tb_logs/{model_name}")   
    overview = utils.load_overview(model_name, cfg["paths"]["models"])
    epoch_offset = overview["epochs"]  if overview else 0
    run_name = f"{model_name}_ep{epoch_offset+1}-{epoch_offset+cfg['training']['epochs']}" 
    hparam_writer = SummaryWriter(log_dir=f"tb_logs/{model_name}/hparams/{run_name}")
    
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
    pbar = tqdm(range(1,n_epochs+1), desc="Epochs")
    for epoch in pbar:

        # Train step
        train_dl = tqdm(train_dataloader, desc=f"Train epoch {epoch}", leave=False)
        train_loss, train_mae = train_one_epoch(model=model,
                                                dataloader=train_dl,
                                                optimizer=optimizer,
                                                criterion=criterion,
                                                device=device)
        # Validation step
        val_dl = tqdm(val_dataloader, desc=f"Val epoch {epoch}", leave=False)
        val_loss, val_mae = validate(model=model,
                                     dataloader=val_dl,
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

        # Add scalars to writer
        global_step = epoch_offset + epoch
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, global_step)
        writer.add_scalars("MAE",  {"train": train_mae,  "val": val_mae},  global_step)

        # Print status
        pbar.set_postfix({
            "tr-loss": f"{train_loss:.2f}", "tr-mae": f"{train_mae:.2f}",
            "val-loss": f"{val_loss:.2f}", "val-mae": f"{val_mae:.2f}",
        })

    # Write metric dict
    metrics = {
        "best_epoch": best_epoch,
        "best_epoch_val_mae": best_epoch_val_mae,
        "train_loss_list": train_loss_list,
        "train_mae_list": train_mae_list,
        "val_loss_list": val_loss_list,
        "val_mae_list": val_mae_list
        }

    # Add hparams to writer
    if overview:
        global_best_mae = min(overview["best_val_mae"], round(best_epoch_val_mae, 3))
        global_best_epoch = overview["best_epoch"] if global_best_mae < best_epoch_val_mae else epoch_offset + best_epoch
    else:
        global_best_mae = round(best_epoch_val_mae, 3)
        global_best_epoch = best_epoch
    hparam_writer.add_hparams(utils.flatten_dict(cfg), {
        "hparam/best_val_mae": global_best_mae,
        "hparam/best_epoch":   global_best_epoch,
    })
    
    # Save model
    utils.save_model(last_cp=model.state_dict(),
                     optim_cp=optimizer.state_dict(),
                     best_cp=best_model_cp,
                     cfg=cfg,
                     metrics=metrics)
