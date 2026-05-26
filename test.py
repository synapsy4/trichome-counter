"""
Testing a trichome counter model.
"""

from pathlib import Path

import torch

from scripts.data import get_dataloader
from scripts.utils import  parse_test_args, load_model, load_config, init_loss, get_model_instance_path
from scripts.engine import validate
from scripts.evaluations import visualize_test_summary



if __name__ == "__main__":

    # Setup hyperparameters
    args = parse_test_args()

    MODEL_NAME = args.model_name
    CP = args.cp
    ROOT_DIR = args.model_root_dir


    # Load config
    model_dir = Path(ROOT_DIR) / MODEL_NAME
    model_instance_path = get_model_instance_path(model_dir=model_dir, cp=CP)
    config_path = model_instance_path / "config.yaml"
    cfg = load_config(config_path)
    
    # Create dataloader
    test_dataloader = get_dataloader(split="test", cfg=cfg)

    # Init model
    model = load_model(model_name=MODEL_NAME, 
                        cp=CP, 
                        root_dir=ROOT_DIR) 

    # Init loss
    criterion = init_loss(cfg)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test model
    test_results = validate(model=model.to(device),
        dataloader=test_dataloader, 
        criterion=criterion, 
        device=device)

    # Visualize test results
    visualize_test_summary(avg_loss=test_results[0],
                           avg_mae=test_results[1],
                           gt_counts=test_results[2],
                           pred_counts=test_results[3],
                           cfg=cfg,
                           cp=CP)

