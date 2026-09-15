import os
import yaml
import numpy as np
import pandas as pd
from preprocessing.load_data import load_raw_data

def read_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_dataset_file(dataset_name, raw_dir):
    dataset_dir = os.path.join(raw_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        return None
    # Look for files inside dataset_dir
    files = [f for f in os.listdir(dataset_dir) if f.endswith(('.h5', '.hdf5', '.npz', '.csv'))]
    if not files:
        return None
    return os.path.join(dataset_dir, files[0])

def inspect():
    config = read_config()
    dataset_name = config['dataset']['name']
    raw_dir = config['dataset']['raw_dir']
    
    print("=" * 60)
    print(f"Dataset Inspection: {dataset_name}")
    print("=" * 60)
    
    file_path = find_dataset_file(dataset_name, raw_dir)
    
    if file_path is None:
        print(f"Dataset files for '{dataset_name}' not found under '{os.path.join(raw_dir, dataset_name)}'.")
        print("\nINSTRUCTIONS FOR PLACING DATASET:")
        print(f"1. Download the '{dataset_name}' dataset (usually metr-la.h5 or pems-bay.h5).")
        print(f"2. Place the file inside the directory:")
        print(f"   {os.path.abspath(os.path.join(raw_dir, dataset_name))}/")
        print(f"3. Make sure the file name ends with .h5, .npz, or .csv.")
        print(f"\nSupported file formats: .h5 (Pandas HDF5), .npz (NumPy archive), .csv")
        print("=" * 60)
        return
        
    print(f"Found dataset file: {file_path}")
    print("Loading dataset...")
    try:
        df, timestamps = load_raw_data(file_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    shape = df.shape
    num_timestamps = shape[0]
    num_sensors = shape[1]
    
    print("\n[DATASET PROPERTIES]")
    print(f"Shape: {shape} (Timestamps: {num_timestamps}, Sensors: {num_sensors})")
    
    if timestamps is not None:
        print(f"Timestamps: Yes (Type: {type(timestamps).__name__})")
        print(f"Start Time: {timestamps[0]}")
        print(f"End Time:   {timestamps[-1]}")
        
        # Calculate sampling interval
        if len(timestamps) > 1:
            time_diffs = pd.Series(timestamps).diff().dropna()
            mode_diff = time_diffs.mode()
            if not mode_diff.empty:
                interval_minutes = mode_diff.iloc[0].total_seconds() / 60
                print(f"Sampling Interval: {interval_minutes} minutes")
            else:
                print(f"Sampling Interval: Variable/Unknown")
    else:
        print("Timestamps: None (Integer indices used)")
        print("Sampling Interval: Unknown (No timestamp indexing)")
        
    # Check for missing values
    # Standard traffic datasets: NaNs represent missing values, and sometimes 0s represent missing values.
    nans = df.isna().sum().sum()
    zeros = (df == 0.0).sum().sum()
    total_elements = df.size
    
    print(f"Missing Values (NaNs): {nans} ({nans / total_elements * 100:.4f}%)")
    print(f"Zero Values:           {zeros} ({zeros / total_elements * 100:.4f}%)")
    
    # Summary stats
    print("\n[SAMPLE VALUES (First 5 sensors, first 5 timesteps)]")
    print(df.iloc[:5, :5])
    
    print("\n[STATISTICAL SUMMARY (overall)]")
    flat_values = df.values.flatten()
    print(f"Min speed:    {np.min(flat_values):.4f}")
    print(f"Max speed:    {np.max(flat_values):.4f}")
    print(f"Mean speed:   {np.mean(flat_values):.4f}")
    print(f"Median speed: {np.median(flat_values):.4f}")
    print(f"Std dev:      {np.std(flat_values):.4f}")
    print("=" * 60)

if __name__ == '__main__':
    inspect()
