"""metrics computation for GP reconstruction"""
import numpy as np
import torch


def compute_metrics(pred_mean, ground_truth):
    """compute reconstruction metrics

    args:
        pred_mean: (N,) predicted temperatures [torch tensor or numpy]
        ground_truth: (N,) ground truth temperatures [torch tensor or numpy]

    returns:
        dict with mse, rmse, mae, nrmse
    """
    if isinstance(pred_mean, torch.Tensor):
        pred_mean = pred_mean.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()

    errors = pred_mean - ground_truth
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    temp_range = ground_truth.max() - ground_truth.min()
    nrmse = rmse / temp_range

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'nrmse': nrmse
    }
