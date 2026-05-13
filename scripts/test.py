"""
Testing a trichome counter model.
"""

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import data_transformations as transforms
from pathlib import Path
from utils import collate_fn, parse_test_args, load_model
from target_maps import generate_density_map
from engine import validate
from data_setup import TrichomeDataset
from loss import DensityCountLoss



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
tmf = generate_density_map if TARGET_MAP_FUN == "generate_density_map" else None

TARGET_DIR = "models"





# Set data path
test_path = Path("data/preprocessed/test")

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
ax[0].bar(x=range(len(test_results[2])), height=test_results[2])
ax[0].set_title("GT counts")
ax[1].bar(x=range(len(test_results[2])), height=test_results[3])
ax[1].set_title("Pred counts")
ax[1].legend(label=f"Avg. Loss = {test_results[0]:.2f}\nAvg. MAE = {test_results[1]:.2f}")
plt.show();

