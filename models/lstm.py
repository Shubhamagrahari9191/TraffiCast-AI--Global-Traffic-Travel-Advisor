import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Standard PyTorch LSTM model for multi-horizon traffic flow forecasting.
    Processes input of shape (batch_size, input_window, num_sensors)
    and predicts output of shape (batch_size, num_horizons, num_sensors)
    """
    def __init__(self, num_sensors: int, num_horizons: int = 3, 
                 hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.num_sensors = num_sensors
        self.num_horizons = num_horizons
        
        # LSTM Layer
        # Input shape: (batch_size, input_window, num_sensors)
        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Fully Connected output layer to predict (num_horizons * num_sensors) values
        self.fc = nn.Linear(hidden_size, num_horizons * num_sensors)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input of shape (batch_size, input_window, num_sensors)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, num_horizons, num_sensors)
        """
        # x: [B, T_in, N]
        # lstm_out: [B, T_in, H], hidden: tuple of ([L, B, H], [L, B, H])
        lstm_out, _ = self.lstm(x)
        
        # Last hidden representation: [B, H]
        last_hidden = lstm_out[:, -1, :]
        
        # Apply dropout and FC projection
        out = self.dropout(last_hidden)
        out = self.fc(out)  # [B, num_horizons * num_sensors]
        
        # Reshape to [B, num_horizons, num_sensors]
        out = out.view(-1, self.num_horizons, self.num_sensors)
        
        return out
