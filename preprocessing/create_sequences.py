import numpy as np
import pandas as pd
from preprocessing.load_data import load_raw_data
from preprocessing.clean_data import clean_traffic_data
from preprocessing.normalize import StandardScaler

def generate_sequences(data: np.ndarray, input_window: int = 12, horizons: list = [3, 6, 12]):
    """
    Generates sliding window sequences.
    
    Args:
        data (np.ndarray): Normalised data of shape (num_timestamps, num_sensors)
        input_window (int): Historical window size (e.g., 12)
        horizons (list of int): Future step offsets (e.g., [3, 6, 12] for 15, 30, 60 mins)
        
    Returns:
        X (np.ndarray): Input sequences of shape (num_samples, input_window, num_sensors)
        Y (np.ndarray): Target values of shape (num_samples, len(horizons), num_sensors)
    """
    max_horizon = max(horizons)
    num_samples = len(data) - input_window - max_horizon + 1
    if num_samples <= 0:
        raise ValueError("Data size is too small for input_window and horizons.")
        
    num_sensors = data.shape[1]
    
    X = np.zeros((num_samples, input_window, num_sensors), dtype=np.float32)
    Y = np.zeros((num_samples, len(horizons), num_sensors), dtype=np.float32)
    
    for i in range(num_samples):
        X[i] = data[i : i + input_window]
        for j, h in enumerate(horizons):
            # i + input_window is the step immediately following the input window
            # h is the offset (e.g. 3, 6, 12), so we take step (i + input_window + h - 1)
            Y[i, j] = data[i + input_window + h - 1]
            
    return X, Y

def get_train_val_test_data(file_path: str, input_window: int = 12, horizons: list = [3, 6, 12],
                            train_ratio: float = 0.7, val_ratio: float = 0.1, test_ratio: float = 0.2,
                            treat_zeros_as_missing: bool = True):
    """
    Loads raw data, cleans it, splits it temporally, fits scaler only on train set,
    normalises all sets, and generates temporal sequences.
    
    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test (np.ndarray)
        scaler (StandardScaler): The scaler fitted on the training set
    """
    # 1. Load raw data
    df, _ = load_raw_data(file_path)
    
    # 2. Clean data (impute NaNs and zeros)
    cleaned_df = clean_traffic_data(df, treat_zeros_as_missing=treat_zeros_as_missing)
    cleaned_data = cleaned_df.values
    
    # 3. Temporal split (no shuffling)
    num_timestamps = len(cleaned_data)
    train_end = int(num_timestamps * train_ratio)
    val_end = int(num_timestamps * (train_ratio + val_ratio))
    
    train_data = cleaned_data[:train_end]
    val_data = cleaned_data[train_end:val_end]
    test_data = cleaned_data[val_end:]
    
    # 4. Normalize data (fitting scaler ONLY on training data)
    scaler = StandardScaler()
    scaler.fit(train_data)
    
    train_norm = scaler.transform(train_data)
    val_norm = scaler.transform(val_data)
    test_norm = scaler.transform(test_data)
    
    # 5. Generate sequences
    X_train, Y_train = generate_sequences(train_norm, input_window, horizons)
    X_val, Y_val = generate_sequences(val_norm, input_window, horizons)
    X_test, Y_test = generate_sequences(test_norm, input_window, horizons)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler
