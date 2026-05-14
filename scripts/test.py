"""
Testing a trichome counter model.
"""

from pathlib import Path

import torch
import numpy as np
from matplotlib import pyplot as plt

from scripts.loss import DensityCountLoss
from scripts.data import get_dataloader
from scripts.utils import  parse_test_args, load_model, load_config, init_loss
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
    TARGET_MAP_FUN = args.target_map_fun

    TARGET_DIR = "models"

    # Load config
    cfg = load_config()
    
    # Create dataloader
    test_dataloader = get_dataloader(split="test", cfg=cfg)

    # Init model
    model = load_model(MODEL_NAME, RUN_ID, TARGET_DIR) 

    # Init loss
    criterion = init_loss(cfg)

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
    fig, ax = plt.subplots(3,1, figsize=(10,19))
    fig.suptitle(f"Avg. Loss = {test_results[0]:.2f} | Avg. MAE = {test_results[1]:.2f}")
    ax[0].bar(x=range(len(test_results[2])), height=test_results[2])
    ax[0].set_title("GT counts")
    ax[1].bar(x=range(len(test_results[2])), height=test_results[3])
    ax[1].set_title("Pred counts")
    ax[2].bar(x=range(len(test_results[2])), height=np.array(test_results[3])-np.array(test_results[2]))
    ax[2].set_title("Pred counts - GT counts")

    # Save figure
    figure_path = Path("outputs") / MODEL_NAME
    figure_path.mkdir(parents=True, exist_ok=True)
    figure_path = figure_path / f"run{RUN_ID}_counts_on_testset.png"

    fig.savefig(figure_path, dpi=200)
    plt.show();

