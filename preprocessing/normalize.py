import numpy as np
import pandas as pd

class StandardScaler:
    """
    Standard scaler for traffic data.
    Fits mean and standard deviation sensor-wise on training data.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, data):
        """
        Fits the scaler on the input data.
        
        Args:
            data (np.ndarray or pd.DataFrame): Data of shape (num_timestamps, num_sensors)
        """
        if isinstance(data, pd.DataFrame):
            val = data.values
        else:
            val = data
            
        self.mean = np.mean(val, axis=0, keepdims=True)
        self.std = np.std(val, axis=0, keepdims=True)
        
        # Prevent division by zero
        self.std[self.std == 0.0] = 1.0
        self.fitted = True

    def transform(self, data):
        """
        Normalizes the input data.
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transforming data.")
        if isinstance(data, pd.DataFrame):
            val = data.values
        else:
            val = data
        return (val - self.mean) / self.std

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """
        Denormalizes the input data.
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before inverse transforming.")
        return data * self.std + self.mean
