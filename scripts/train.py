"""
Training of a trichome counter model.
"""

from pathlib import Path

import torch

from scripts.loss import DensityCountLoss
from scripts.data import get_dataloader
from scripts.engine import train
from scripts.utils import parse_train_args, init_model


if __name__ == "__main__":

    # Setup hyperparameters
    args = parse_train_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    WEIGHT_DECAY = args.weight_decay
    SHORT_SIDE = args.short_side
    MODEL_NAME = args.model_name
    MODEL_TYPE = args.model_type
    ACTIVATION = args.activation
    TARGET_MAP_FUN = args.target_map_fun

    LAMBDA_COUNT = args.lbda_count
    SIGMA = args.sigma


    # Create hyperparameter dict
    hparams = {"model_name": MODEL_NAME,
                "model_type": MODEL_TYPE,
                "activation": ACTIVATION,
                "target_map_fun": TARGET_MAP_FUN,
                "target_map_args": {"sigma": SIGMA},
                "epochs": EPOCHS,
                "loss_fun": "DensityCountLoss",
                "loss_args": {"lambda_count": LAMBDA_COUNT},
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "short_side_len": SHORT_SIDE}
        
    
    # Set data paths
    train_path = Path("data/preprocessed/train")
    val_path = Path("data/preprocessed/val")

    """
    # Create datasets
    train_ds = TrichomeDataset(root=train_path,  
                            transform=transforms.Compose(
                                [transforms.ResizeShortSide(SHORT_SIDE),
                                    transforms.RandomCrop(SHORT_SIDE,SHORT_SIDE),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomBrightness(0.2)]),
                            target_map_fun=tmf,
                            sigma=SIGMA) 
    val_ds = TrichomeDataset(root=val_path,
                            transform=transforms.Compose([
                                transforms.ResizeShortSide(SHORT_SIDE),
                                transforms.PadToMultipleOf32()]), # NOTE: Add padding s.t. image W of 1023 is padded to 1024 (=divisible by 32)
                            target_map_fun=tmf,
                            sigma=SIGMA)
    
    # Create dataloaders
    train_dataloader = DataLoader(dataset=train_ds,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_ds,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                collate_fn=collate_fn)
    """
    
    cfg = {"training": {"batch_size": BATCH_SIZE, "target_map_fun": TARGET_MAP_FUN, "target_map_args": SIGMA}, 
           "transforms": {"short_side": SHORT_SIDE, "crop": SHORT_SIDE, "brightness": 0.2}, 
           "paths": {"train_path": train_path, "val_path": val_path}}
    train_dataloader = get_dataloader(split="train", cfg=cfg)
    val_dataloader = get_dataloader(split="val", cfg=cfg)
   
    
    # Init model
    model = init_model(model_name=MODEL_NAME, 
                       model_type=MODEL_TYPE, 
                       activation=ACTIVATION) 

    # Init loss
    criterion = DensityCountLoss(lambda_count=LAMBDA_COUNT)

    # Init optimizer
    optimizer = torch.optim.AdamW( 
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Send model to device
    model.to(device)

    # Train model
    train(model=model,
        model_name=MODEL_NAME, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        epochs=EPOCHS, 
        optimizer=optimizer, 
        criterion=criterion, 
        hparams=hparams,
        device=device)
