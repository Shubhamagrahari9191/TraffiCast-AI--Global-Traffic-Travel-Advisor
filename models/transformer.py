import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding for Transformer models.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        
        # Create constant positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input embedding of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Embedding with positional encoding added of same shape
        """
        # x.shape[1] is seq_len
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerModel(nn.Module):
    """
    Standard Transformer Encoder model for multi-horizon traffic flow forecasting.
    Processes input of shape (batch_size, input_window, num_sensors)
    and predicts output of shape (batch_size, num_horizons, num_sensors)
    """
    def __init__(self, num_sensors: int, num_horizons: int = 3, 
                 d_model: int = 64, num_heads: int = 4, transformer_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.2):
        super(TransformerModel, self).__init__()
        
        self.num_sensors = num_sensors
        self.num_horizons = num_horizons
        
        # 1. Linear Embedding
        self.embedding = nn.Linear(num_sensors, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder Layer & Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # 5. Fully Connected output projection
        self.fc = nn.Linear(d_model, num_horizons * num_sensors)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input of shape (batch_size, input_window, num_sensors)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, num_horizons, num_sensors)
        """
        # x: [B, T_in, N]
        
        # 1. Embed input: [B, T_in, d_model]
        out = self.embedding(x)
        
        # 2. Add Positional Encoding
        out = self.pos_encoder(out)
        
        # 3. Pass through Transformer Encoder: [B, T_in, d_model]
        out = self.transformer_encoder(out)
        
        # 4. Temporal Pooling (Mean Pooling over time steps T_in): [B, d_model]
        pooled = out.mean(dim=1)
        
        # 5. Apply Dropout and Projection: [B, num_horizons * num_sensors]
        proj = self.dropout(pooled)
        proj = self.fc(proj)
        
        # Reshape to [B, num_horizons, num_sensors]
        output = proj.view(-1, self.num_horizons, self.num_sensors)
        
        return output
