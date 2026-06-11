"""
Evaluation functions
"""

import torch
import numpy as np
from tqdm.auto import tqdm


def predict_single_image(model: torch.nn.Module,
                         img: torch.Tensor,
                         device: torch.device
                         ) -> tuple[np.ndarray, int]:
    """
    Run inference on a single image tensor and return the predicted densitymap
    and trichome counts derived via densitymap summation.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    img : torch.Tensor
        RGB image tensor of shape [3, H, W].
    device : torch.device
        Device to run inference on.

    Returns
    -------
    pred_density : np.ndarray
        Raw predicted density.
    pred_count : int
        Predicted trichome count.
    """
    
    model.eval()
    
    # Make sure img + model are on the same device
    model.to(device), img.to(device)

    # Make inference + directly convert to numpy
    with torch.inference_mode():
        pred = model(img.unsqueeze(0))
        pred_density = pred["density"].squeeze().cpu().numpy()
        pred_count = round(pred_density.sum()) if pred["count"] is None else pred["count"].squeeze().cpu().numpy()

    return pred_density, pred_count
    

def evaluate_on_testset(model: torch.nn.Module,
                        dataloader: torch.utils.data.DataLoader,
                        device: torch.device
                        ) -> dict:
    """
    Run inference over a full test DataLoader and compute quantitative
    counting metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    dataloader : torch.utils.data.DataLoader
        DataLoader whose dataset returns: (img, target_map, coords)
    device : torch.device
        Device to run inference on.

    Returns
    -------
    dict with keys:
        "gt_counts" : np.ndarray [N]
            Ground-truth counts per sample.
        "pred_counts_sum" : np.ndarray [N]
            Sum-based predicted counts.
        "mae_sum" : float
            MAE using densitymap summation.
        "rmse_sum" : float
            RMSE using densitymap summation.
        "me_sum" : float
            Mean signed error (bias) for densitymap summation.
    """

    model.eval()
    model.to(device)

    gt_counts = []
    pred_counts = []

    # Make inference on test set
    with torch.inference_mode():
        for images, _, coords in tqdm(dataloader):
            
            images = images.to(device)       
            pred = model(images) 
            pred_densities = pred["density"]
            pred_counts_batch = pred["count"]

            for i in range(images.size(0)):

                # GT counts
                coords_i = coords[i]
                gt_counts.append(len(coords_i))

                # Pred counts
                if pred_counts_batch is None:
                    pred_density = pred_densities[i].squeeze().cpu().numpy()
                    pred_counts.append(round(pred_density.sum()))
                else:
                    pred_counts.append(pred_counts_batch[i].squeeze().cpu().numpy())
        
    # Convert counts to numpy
    gt = np.array(gt_counts, dtype=np.float32)
    pred = np.array(pred_counts, dtype=np.float32)


    # Make results dict
    results = {
        # Per-sample arrays
        "gt_counts": gt,
        "pred_counts": pred,
        # Aggregate metrics
        "mae_sum": float(np.mean(np.abs(pred - gt))),
        "rmse_sum": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "me_sum": float(np.mean(pred - gt)),
    }

    return results