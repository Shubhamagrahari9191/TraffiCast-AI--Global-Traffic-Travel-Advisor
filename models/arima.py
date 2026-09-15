import os
import json
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

class ARIMABaseline:
    """
    ARIMA baseline model for traffic flow forecasting.
    Fits independent autoregressive models on each sensor's historical time series.
    Tracks fitting failures and naive persistence fallbacks for full research disclosure.
    """
    def __init__(self, order: tuple = (2, 1, 0)):
        self.order = order
        self.stats = {
            'order': list(order),
            'total_sensors_evaluated': 0,
            'fit_attempts': 0,
            'successful_initial_fits': 0,
            'failed_initial_fits': 0,
            'total_horizon_predictions': 0,
            'fallback_predictions': 0,
            'fallback_percentage': 0.0
        }

    def predict(self, train_data: np.ndarray, test_data: np.ndarray, 
                input_window: int = 12, horizons: list = [3, 6, 12],
                sensor_indices: list = None, step_size: int = 50,
                dataset_name: str = "dataset"):
        """
        Runs rolling ARIMA prediction on selected sensors.
        
        Args:
            train_data (np.ndarray): Training data of shape (num_train_timestamps, num_sensors)
            test_data (np.ndarray): Test data of shape (num_test_timestamps, num_sensors)
            input_window (int): Historical window size (e.g. 12)
            horizons (list of int): Offsets (e.g. [3, 6, 12])
            sensor_indices (list of int): Sensor indices to evaluate.
            step_size (int): Temporal sampling rate to evaluate ARIMA predictions.
            dataset_name (str): Name of dataset for saving audit statistics.
            
        Returns:
            predictions (np.ndarray): shape (num_samples, len(horizons), len(sensor_indices))
            targets (np.ndarray): shape (num_samples, len(horizons), len(sensor_indices))
        """
        num_sensors = train_data.shape[1]
        if sensor_indices is None:
            sensor_indices = list(range(num_sensors))
            
        test_len = len(test_data)
        max_horizon = max(horizons)
        
        # Calculate evaluation indices
        num_samples = test_len - input_window - max_horizon + 1
        sample_indices = list(range(0, num_samples, step_size))
        
        if len(sample_indices) == 0:
            raise ValueError("Test set is too small for the specified step_size, input_window, and horizons.")
            
        preds_all = np.zeros((len(sample_indices), len(horizons), len(sensor_indices)), dtype=np.float32)
        targets_all = np.zeros((len(sample_indices), len(horizons), len(sensor_indices)), dtype=np.float32)
        
        print(f"Running rolling ARIMA on {len(sensor_indices)} sensors, evaluating {len(sample_indices)} test timestamps each (step_size={step_size})...")
        
        self.stats['total_sensors_evaluated'] = len(sensor_indices)
        self.stats['step_size'] = step_size
        self.stats['num_timestamps_evaluated'] = len(sample_indices)
        
        total_preds = 0
        fallback_preds = 0
        
        for s_idx, sensor in enumerate(tqdm(sensor_indices, desc="ARIMA Sensors")):
            train_series = train_data[:, sensor]
            test_series = test_data[:, sensor]
            full_series = np.concatenate([train_series, test_series])
            train_len = len(train_series)
            
            # Initial parameter estimation on training slice
            params = None
            self.stats['fit_attempts'] += 1
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(train_series, order=self.order)
                    res = model.fit()
                    params = res.params
                    self.stats['successful_initial_fits'] += 1
            except Exception as e:
                self.stats['failed_initial_fits'] += 1
                
            # Perform rolling prediction
            for idx_out, i in enumerate(sample_indices):
                history_end = train_len + i + input_window
                history = full_series[:history_end]
                used_fallback = False
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if params is not None:
                            mod = ARIMA(history, order=self.order)
                            res = mod.filter(params)
                        else:
                            mod = ARIMA(history, order=self.order)
                            res = mod.fit()
                            
                        forecast = res.forecast(steps=max_horizon)
                except Exception:
                    # Fallback to naive persistence when ARIMA numerical optimization diverges
                    forecast = np.full(max_horizon, history[-1])
                    used_fallback = True
                    
                for j, h in enumerate(horizons):
                    total_preds += 1
                    if used_fallback:
                        fallback_preds += 1
                    preds_all[idx_out, j, s_idx] = forecast[h - 1]
                    targets_all[idx_out, j, s_idx] = full_series[history_end + h - 1]
                    
        self.stats['total_horizon_predictions'] = total_preds
        self.stats['fallback_predictions'] = fallback_preds
        self.stats['fallback_percentage'] = float((fallback_preds / total_preds * 100.0) if total_preds > 0 else 0.0)
        
        # Save audit statistics
        os.makedirs("outputs/tables", exist_ok=True)
        stats_path = f"outputs/tables/arima_fallback_statistics_{dataset_name}.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)
            
        print(f"\n[ARIMA Audit] Predictions: {total_preds} | Fallbacks: {fallback_preds} ({self.stats['fallback_percentage']:.2f}%)")
        print(f"[ARIMA Audit] Fallback report saved to: {stats_path}")
        
        return preds_all, targets_all
