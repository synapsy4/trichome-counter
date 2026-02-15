"""
Train, validate and inference functions.
"""
import torch
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

def train(model, train_dataloader, val_dataloader, epochs, optimizer, criterion, device):

    for epoch in tqdm(range(epochs)):

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
        
        # Print status
        print(f"Epoch {epoch+1:03d} | Train-loss={train_loss:.0f}, train-mae={train_mae:.0f} | Val-loss={val_loss:.0f}, val-mae={val_mae:.0f}")