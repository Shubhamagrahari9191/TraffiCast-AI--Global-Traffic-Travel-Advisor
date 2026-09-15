import torch
import torch.nn as nn
from models.transformer import PositionalEncoding

class TransformerBiLSTMModel(nn.Module):
    """
    Proposed Hybrid Model: Transformer Encoder + BiLSTM
    For multi-horizon multivariate spatiotemporal traffic forecasting.
    
    Conceptual Architecture Pipeline:
    Input [B, T_in, N]  (Joint historical readings from all N sensors)
    ↓
    Linear Sensor Embedding [B, T_in, d_model]  (Implicit cross-sensor dependency mixing)
    ↓
    Positional Encoding [B, T_in, d_model]  (Temporal sequence lag encoding)
    ↓
    [TRANSFORMER ENCODER BLOCK]
    Multi-Head Temporal Self-Attention + FFN [B, T_in, d_model]  (Captures long-range temporal dependencies)
    ↓
    [BIDIRECTIONAL LSTM BLOCK]
    BiLSTM Sequence Processing [B, T_in, 2 * bilstm_hidden]  (Models fine-grained forward & backward temporal dynamics)
    ↓
    Last Hidden Representation [B, 2 * bilstm_hidden]  (Context vector from final temporal step)
    ↓
    Dropout Regularization
    ↓
    [MULTI-HORIZON DECODER]
    Dense Projection Layer [B, num_horizons * num_sensors]
    ↓
    Output [B, num_horizons, num_sensors]  (Simultaneous forecasts for all horizons and sensors)
    
    Note on Spatiotemporal Modeling:
    - Temporal dependencies: Modeled explicitly via Transformer Temporal Self-Attention and BiLSTM recurrent units.
    - Spatial dependencies: Modeled via joint multivariate sensor representations and dense linear projections
      (implicit cross-sensor dependency learning).
    - Road topology / Graph Adjacency: An explicit road network graph or GNN adjacency matrix is NOT used.
    """
    def __init__(self, num_sensors: int, num_horizons: int = 3, 
                 d_model: int = 64, num_heads: int = 4, transformer_layers: int = 2,
                 dim_feedforward: int = 128, bilstm_hidden: int = 64, bilstm_layers: int = 1,
                 dropout: float = 0.2):
        super(TransformerBiLSTMModel, self).__init__()
        
        self.num_sensors = num_sensors
        self.num_horizons = num_horizons
        
        # 1. EMBEDDING & POSITIONAL ENCODING BLOCK
        # Maps all N sensor readings at each time step to a joint hidden embedding
        self.embedding = nn.Linear(num_sensors, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. TRANSFORMER ENCODER BLOCK (Temporal Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 3. BIDIRECTIONAL LSTM BLOCK
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if bilstm_layers > 1 else 0.0
        )
        
        # 4. DECODER & PROJECTION BLOCK
        self.dropout = nn.Dropout(p=dropout)
        # BiLSTM is bidirectional, so hidden dimension is doubled
        self.fc = nn.Linear(bilstm_hidden * 2, num_horizons * num_sensors)

    def forward(self, x):
        """
        Forward pass of the Hybrid model.
        
        Args:
            x (torch.Tensor): Joint input tensor of shape (batch_size, input_window, num_sensors)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, num_horizons, num_sensors)
        """
        # Step 1: Joint Sensor Embedding & Positional Encoding: [B, T_in, N] -> [B, T_in, d_model]
        embed = self.embedding(x)
        embed = self.pos_encoder(embed)
        
        # Step 2: Temporal Self-Attention via Transformer Encoder: [B, T_in, d_model] -> [B, T_in, d_model]
        trans_out = self.transformer_encoder(embed)
        
        # Step 3: Bidirectional Sequential Dynamics via BiLSTM: [B, T_in, d_model] -> [B, T_in, 2 * bilstm_hidden]
        bilstm_out, _ = self.bilstm(trans_out)
        
        # Step 4: Extract the final timestep representation: [B, 2 * bilstm_hidden]
        last_hidden = bilstm_out[:, -1, :]
        
        # Step 5: Multi-horizon, multi-sensor projection: [B, 2 * bilstm_hidden] -> [B, num_horizons * num_sensors]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        
        # Step 6: Reshape to [B, num_horizons, num_sensors]
        output = out.view(-1, self.num_horizons, self.num_sensors)
        
        return output
