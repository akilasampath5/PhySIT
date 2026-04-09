
"""
utils.py
--------
Utility functions for PhySIT: data loading, preprocessing, and evaluation metrics.

Reference:
    Sampath et al., "Physics-Informed Machine Learning for Sea Ice Thickness Prediction,"
    IEEE ICKG 2024. https://doi.org/10.1109/ICKG63256.2024.00048
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        SMAPE score as a percentage.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100 / len(y_true)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAPE score as a percentage.
    """
    diff = np.abs(y_true - y_pred) / np.abs(y_true)
    diff[np.isnan(diff)] = 0.0
    diff[np.isinf(diff)] = 0.0
    return np.mean(diff) * 100 / len(y_true)


def theil_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Theil's Inequality Coefficient.

    A value of 0 indicates perfect prediction.
    A value of 1 indicates predictions no better than naive forecast.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Theil's U coefficient.
    """
    diff_true = y_true - np.mean(y_true)
    diff_pred = y_pred - np.mean(y_pred)
    num   = np.sqrt(np.mean(diff_true ** 2)) + np.sqrt(np.mean(diff_pred ** 2))
    denom = np.sqrt(np.mean(y_true ** 2))    + np.sqrt(np.mean(y_pred ** 2))
    return num / denom


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all evaluation metrics in one call.

    Args:
        y_true: Ground truth values (flattened).
        y_pred: Predicted values (flattened).

    Returns:
        Dictionary with MSE, RMSE, SMAPE, MAPE, and Theil coefficient.
    """
    mse_val = mean_squared_error(y_true, y_pred)
    return {
        "MSE"   : mse_val,
        "RMSE"  : np.sqrt(mse_val),
        "SMAPE" : smape(y_true, y_pred),
        "MAPE"  : mape(y_true, y_pred),
        "Theil" : theil_coefficient(y_true, y_pred),
    }


def print_metrics(metrics: dict, label: str = "") -> None:
    """Pretty-print evaluation metrics."""
    prefix = f"[{label}] " if label else ""
    for k, v in metrics.items():
        print(f"{prefix}{k}: {v:.4f}")


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def batch_predict(
    model: torch.nn.Module,
    X: torch.Tensor,
    batch_size: int = 64,
) -> tuple:
    """
    Run inference in batches to avoid OOM on large datasets.

    Args:
        model     : Trained PyTorch model.
        X         : Input tensor of shape (N, ...).
        batch_size: Number of samples per batch.

    Returns:
        (predictions tensor, list of per-batch MSE losses if y provided)
    """
    model.eval()
    predictions = []
    n = X.size(0)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            batch = X[start:end]
            pred  = model(batch)
            predictions.append(pred)
    return torch.cat(predictions, dim=0)
