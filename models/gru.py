import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """
    Standard PyTorch GRU model for multi-horizon traffic flow forecasting.
    Processes input of shape (batch_size, input_window, num_sensors)
    and predicts output of shape (batch_size, num_horizons, num_sensors)
    """
    def __init__(self, num_sensors: int, num_horizons: int = 3, 
                 hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super(GRUModel, self).__init__()
        
        self.num_sensors = num_sensors
        self.num_horizons = num_horizons
        
        # GRU Layer
        self.gru = nn.GRU(
            input_size=num_sensors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Fully Connected output layer
        self.fc = nn.Linear(hidden_size, num_horizons * num_sensors)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input of shape (batch_size, input_window, num_sensors)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, num_horizons, num_sensors)
        """
        # x: [B, T_in, N]
        # gru_out: [B, T_in, H]
        gru_out, _ = self.gru(x)
        
        # Last hidden representation: [B, H]
        last_hidden = gru_out[:, -1, :]
        
        # Apply dropout and FC projection
        out = self.dropout(last_hidden)
        out = self.fc(out)  # [B, num_horizons * num_sensors]
        
        # Reshape to [B, num_horizons, num_sensors]
        out = out.view(-1, self.num_horizons, self.num_sensors)
        
        return out
