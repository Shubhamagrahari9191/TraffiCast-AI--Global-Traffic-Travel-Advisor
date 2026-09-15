import os
import numpy as np
import pandas as pd

def resolve_dataset_path(raw_dir: str, dataset_name: str) -> str:
    """
    Locates the dataset file dynamically inside the target folder raw_dir/dataset_name.
    Supported extensions: .h5, .hdf5, .npz, .csv
    """
    target_dir = os.path.join(raw_dir, dataset_name)
    if os.path.exists(target_dir):
        for f in os.listdir(target_dir):
            if f.lower().endswith(('.h5', '.hdf5', '.npz', '.csv')):
                return os.path.join(target_dir, f)
    raise FileNotFoundError(f"No valid dataset file (.h5, .npz, .csv) found in {target_dir}")


def load_raw_data(file_path: str):
    """
    Loads raw traffic flow dataset.
    Supports HDF5 (.h5), NPZ (.npz), and CSV (.csv) formats.
    
    Args:
        file_path (str): Path to the dataset file.
        
    Returns:
        df (pd.DataFrame): Traffic speed/flow data, shape (num_timestamps, num_sensors)
        timestamps (pd.DatetimeIndex or None): Timestamps corresponding to the rows
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.h5', '.hdf5']:
        try:
            # Try loading with standard pandas read_hdf
            df = pd.read_hdf(file_path)
            return df, df.index
        except Exception as pandas_err:
            # Fall back to manual parsing using h5py (highly robust, bypassing PyTables issues on Python 3.14)
            import h5py
            try:
                with h5py.File(file_path, 'r') as f:
                    keys = list(f.keys())
                    # Look for standard keys like 'df' or 'speed'
                    group_key = None
                    for key in ['df', 'speed', 'data']:
                        if key in keys:
                            group_key = key
                            break
                    if group_key is None:
                        group_key = keys[0]
                    
                    group = f[group_key]
                    
                    # If it's a pandas dataframe saved in HDF5 format
                    if 'block0_values' in group:
                        data = group['block0_values'][:]
                        
                        # Load columns
                        if 'axis0' in group:
                            columns = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in group['axis0'][:]]
                        else:
                            columns = None
                            
                        # Load index (timestamps)
                        if 'axis1' in group:
                            idx_data = group['axis1'][:]
                            try:
                                # Convert int64 timestamps to datetime
                                timestamps = pd.to_datetime(idx_data)
                            except Exception:
                                timestamps = pd.Index(idx_data)
                        else:
                            timestamps = None
                            
                        df = pd.DataFrame(data, index=timestamps, columns=columns)
                        return df, df.index
                    else:
                        # Fallback for generic datasets in HDF5
                        if isinstance(group, h5py.Dataset):
                            data = group[:]
                        else:
                            # Search first dataset inside group
                            data_key = None
                            for gk in group.keys():
                                if isinstance(group[gk], h5py.Dataset):
                                    data_key = gk
                                    break
                            if data_key is not None:
                                data = group[data_key][:]
                            else:
                                raise ValueError("Could not find a valid dataset in HDF5 structure.")
                        
                        if len(data.shape) == 3:
                            data = data[..., 0]  # Take first feature (speed)
                        
                        df = pd.DataFrame(data)
                        return df, None
            except Exception as h5py_err:
                raise RuntimeError(
                    f"Failed to parse HDF5 file using pandas.read_hdf (Error: {pandas_err}) "
                    f"and h5py (Error: {h5py_err})"
                )
            
    elif ext == '.npz':
        data_dict = np.load(file_path)
        keys = list(data_dict.keys())
        
        # Look for standard keys like 'data' or 'speed'
        data_key = None
        for k in ['data', 'speed', 'x', 'y']:
            if k in data_dict:
                data_key = k
                break
        if data_key is None:
            data_key = keys[0]
            
        data = data_dict[data_key]
        if len(data.shape) == 3:
            data = data[..., 0]  # Take first feature (speed)
            
        # Try to find timestamps
        timestamps = None
        for k in ['timestamps', 'time', 'dates', 'index']:
            if k in data_dict:
                timestamps = data_dict[k]
                break
                
        if timestamps is not None:
            try:
                if np.issubdtype(timestamps.dtype, np.number):
                    timestamps = pd.to_datetime(timestamps, unit='s')
                else:
                    timestamps = pd.to_datetime(timestamps)
                df = pd.DataFrame(data, index=timestamps)
                return df, df.index
            except Exception:
                pass
                
        df = pd.DataFrame(data)
        return df, None
        
    elif ext == '.csv':
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df, df.index
        
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .h5, .npz, .csv")
