import numpy as np
import pandas as pd

def clean_traffic_data(df: pd.DataFrame, treat_zeros_as_missing: bool = True):
    """
    Cleans the raw traffic speed dataframe.
    Replaces NaNs (and optionally zeros) with linear interpolation.
    
    Args:
        df (pd.DataFrame): Raw traffic data, shape (num_timestamps, num_sensors)
        treat_zeros_as_missing (bool): If True, treats 0.0 values as missing readings.
        
    Returns:
        pd.DataFrame: Cleaned traffic data.
    """
    cleaned_df = df.copy()
    
    # Treat zeros as missing
    if treat_zeros_as_missing:
        # We replace exactly 0.0 with NaN
        cleaned_df = cleaned_df.replace(0.0, np.nan)
        
    # Detect missing values
    num_missing = cleaned_df.isna().sum().sum()
    if num_missing > 0:
        # Interpolate missing values column-wise (sensor-wise)
        cleaned_df = cleaned_df.interpolate(method='linear', axis=0)
        # Fill any remaining NaNs at the boundaries (start/end) with forward-fill or backward-fill
        cleaned_df = cleaned_df.ffill().bfill()
        
    # Verify no NaNs or zeros remain (unless it was completely empty)
    remaining_missing = cleaned_df.isna().sum().sum()
    if remaining_missing > 0:
        # Fallback to fill with overall mean
        cleaned_df = cleaned_df.fillna(cleaned_df.mean().fillna(0.0))
        
    return cleaned_df
