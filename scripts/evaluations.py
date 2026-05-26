"""
Evaluation functions
"""

from pathlib import Path
from collections import Any

import numpy as np
from matplotlib import pyplot as plt

def visualize_test_summary(avg_loss: float,
                           avg_mae: float,
                           gt_counts: list,
                           pred_counts: list,
                           cfg: dict[str, Any],
                           cp: str
                           ) -> None:
    """
    TODO: Add docsting
    """

    fig, ax = plt.subplots(3,1, figsize=(10,19))
    fig.suptitle(f"Avg. Loss = {avg_loss:.2f} | Avg. MAE = {avg_mae:.2f}")
    ax[0].bar(x=range(len(gt_counts)), height=gt_counts)
    ax[0].set_title("GT counts")
    ax[1].bar(x=range(len(pred_counts)), height=pred_counts)
    ax[1].set_title("Pred counts")
    ax[2].bar(x=range(len(gt_counts)), height=np.array(pred_counts)-np.array(gt_counts))
    ax[2].set_title("Pred counts - GT counts")

    # Save figure
    figure_path = Path(cfg["paths"]["outputs"]) / cfg["model"]["model_name"]
    figure_path.mkdir(parents=True, exist_ok=True)
    figure_path = figure_path / f"{cp}_cp_counts_on_testset.png"

    fig.savefig(figure_path, dpi=200)
    print(f"[INFO] Test summary saved to '{figure_path}'")
