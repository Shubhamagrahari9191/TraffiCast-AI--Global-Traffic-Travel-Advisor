import numpy as np

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Mean Absolute Error (MAE) with robust NaN filtering.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Root Mean Squared Error (RMSE) with robust NaN filtering.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 1.0) -> float:
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Ignores ground-truth values below the threshold to prevent division by zero or explosive ratios.
    
    Args:
        y_true (np.ndarray): True target traffic measurements.
        y_pred (np.ndarray): Predicted traffic measurements.
        threshold (float): Ignore ground truth values below this threshold (e.g. 1.0).
        
    Returns:
        float: MAPE in percentage (0 to 100).
    """
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    threshold_mask = (y_true >= threshold) & valid_mask
    if np.sum(threshold_mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[threshold_mask] - y_pred[threshold_mask]) / y_true[threshold_mask])) * 100.0)

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 1.0) -> float:
    """
    Calculates Symmetric Mean Absolute Percentage Error (sMAPE).
    Ignores ground-truth values below threshold.
    
    Args:
        y_true (np.ndarray): True target traffic measurements.
        y_pred (np.ndarray): Predicted traffic measurements.
        threshold (float): Ignore values below this threshold.
        
    Returns:
        float: sMAPE in percentage (0 to 100).
    """
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    threshold_mask = (y_true >= threshold) & valid_mask
    if np.sum(threshold_mask) == 0:
        return 0.0
    denominator = np.abs(y_true[threshold_mask]) + np.abs(y_pred[threshold_mask])
    zero_denom = (denominator == 0.0)
    denominator[zero_denom] = 1.0
    return float(np.mean(2.0 * np.abs(y_true[threshold_mask] - y_pred[threshold_mask]) / denominator) * 100.0)

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 1.0) -> dict:
    """
    Computes all standard forecasting evaluation metrics safely.
    """
    return {
        'MAE': calculate_mae(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred, threshold),
        'sMAPE': calculate_smape(y_true, y_pred, threshold)
    }
