"""
Train, validate and inference functions.
"""
import torch
from . import utils
from tqdm.auto import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
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


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set.

    Returns
    -------
    avg_loss : float
        Average validation loss
    avg_mae : float
        Average count MAE
    """

    model.eval()

    total_loss = 0.0
    total_mae = 0.0

    for images, gt_density, _ in dataloader:

        images = images.to(device)
        gt_density = gt_density.to(device)

        # Forward pass (no gradients)
        pred_density = model(images)

        # Compute loss
        loss, _, _ = criterion(pred_density, gt_density)

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

def train(model, model_name, train_dataloader, val_dataloader, epochs, optimizer, criterion, hparams, device):
    """
    Train a model for a specified number of epochs and track performance metrics.

    Identifies the best-performing epoch based on validation MAE, and saves the 
    best and last model checkpoints along with training metadata.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    model_name : str
        Name used when saving the trained model.
    train_dataloader : torch.utils.data.DataLoader
        DataLoader providing the training dataset.
    val_dataloader : torch.utils.data.DataLoader
        DataLoader providing the validation dataset.
    epochs : int
        Number of training epochs.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    criterion : callable
        Loss function used to compute training and validation loss.
    hparams : dict
        Dictionary used to store the hyperparameters.
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
    
    # Run training loop
    for epoch in tqdm(range(1,epochs+1)):

        # Train step
        train_loss, train_mae = train_one_epoch(model=model,
                                                dataloader=train_dataloader,
                                                optimizer=optimizer,
                                                criterion=criterion,
                                                device=device)
        # Validation step
        val_loss, val_mae = validate(model=model,
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
        print(f"Epoch {epoch:03d} | Train-loss={train_loss:.0f}, train-mae={train_mae:.0f} | Val-loss={val_loss:.0f}, val-mae={val_mae:.0f}")

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
        "hyperparameters": hparams,
        "metrics": metrics
    }


    # Save model
    utils.save_model(model=model,
               model_name=model_name,
               metadata=metadata,
               best_cp=best_model_cp)