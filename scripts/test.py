"""
Testing a trichome counter model.
"""

from pathlib import Path

import torch
from matplotlib import pyplot as plt

from scripts.loss import DensityCountLoss
from scripts.data import get_dataloader
from scripts.utils import  parse_test_args, load_model
from scripts.engine import validate



if __name__ == "__main__":

    # Setup hyperparameters
    args = parse_test_args()

    SHORT_SIDE = args.short_side
    MODEL_NAME = args.model_name
    SIGMA = args.sigma
    LAMBDA_COUNT = args.lbda_count
    BATCH_SIZE = args.batch_size
    RUN_ID = args.run_id
    CP = args.cp
    TARGET_MAP_FUN = args.target_map_fun

    TARGET_DIR = "models"





    # Set data path
    test_path = Path("data/preprocessed/test")
    """
    # Create dataset
    test_ds = TrichomeDataset(root=test_path,
                                transform=transforms.Compose([
                                    transforms.ResizeShortSide(SHORT_SIDE),
                                    transforms.PadToMultipleOf32()]), # NOTE: Add padding s.t. image W of 1023 is padded to 1024 (=divisible by 32)
                                target_map_fun=tmf,
                                sigma=SIGMA)

    # Create dataloader
    test_dataloader = DataLoader(dataset=test_ds,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    collate_fn=collate_fn)
    """

    cfg = {"training": {"batch_size": BATCH_SIZE, "target_map_fun": TARGET_MAP_FUN, "target_map_args": SIGMA}, 
        "transforms": {"short_side": SHORT_SIDE, "brightness": 0.2}, 
        "paths": {"test_path": test_path}}
    test_dataloader = get_dataloader(split="test", cfg=cfg)

    # Init model
    model = load_model(MODEL_NAME, RUN_ID, CP, TARGET_DIR) 

    # Init loss
    criterion = DensityCountLoss(lambda_count=LAMBDA_COUNT)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Send model to device
    model.to(device)

    # Test model
    test_results = validate(model=model,
        dataloader=test_dataloader, 
        criterion=criterion, 
        device=device)

    # Visualize test results
    fig, ax = plt.subplots(2,1, figsize=(10,6))
    fig.suptitle(f"Avg. Loss = {test_results[0]:.2f} | Avg. MAE = {test_results[1]:.2f}")
    ax[0].bar(x=range(len(test_results[2])), height=test_results[2])
    ax[0].set_title("GT counts")
    ax[1].bar(x=range(len(test_results[2])), height=test_results[3])
    ax[1].set_title("Pred counts")
    plt.show();

