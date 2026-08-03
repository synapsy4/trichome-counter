"""
Testing a trichome counter model.
"""

from pathlib import Path

import torch

from trichomecounter.data.data import get_dataloader
from trichomecounter.utils.utils import  parse_test_args, load_model, load_config, get_model_instance_path
from trichomecounter.model.evaluations import evaluate_on_testset
from trichomecounter.utils.visualizations import plot_error_distribution, visualize_test_samples, visualize_test_sample_trichomes
from trichomecounter.paths import MODELS_DIR



if __name__ == "__main__":

    # Setup hyperparameters
    args = parse_test_args()

    MODEL_NAME = args.model_name
    CP = args.cp


    # Load config
    model_dir = MODELS_DIR / MODEL_NAME
    model_instance_path = get_model_instance_path(model_dir=model_dir, cp=CP)
    config_path = model_instance_path / "config.yaml"
    cfg = load_config(config_path)
    
    # Create dataloader
    test_dataloader = get_dataloader(split="test", cfg=cfg)

    # Init model
    model = load_model(model_name=MODEL_NAME, 
                        cp=CP) 

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test model
    test_results = evaluate_on_testset(model=model, 
                                       dataloader=test_dataloader, 
                                       device=device)

    # Visualize test results
    plot_error_distribution(eval_results=test_results,
                            cfg=cfg,
                            cp=CP,
                            save_fig=True)
    
    # Visualize random samples
    visualize_test_samples(model=model, 
                           dataset=test_dataloader.dataset, 
                           device=device, 
                           cfg=cfg, 
                           cp=CP, 
                           n_samples=5,
                           save_fig=True)
    
    # Visualize random samples zoomed in on trichome region
    visualize_test_sample_trichomes(model=model, 
                                    dataset=test_dataloader.dataset, 
                                    device=device, 
                                    cfg=cfg, 
                                    cp=CP, 
                                    n_samples=5,
                                    save_fig=True)


