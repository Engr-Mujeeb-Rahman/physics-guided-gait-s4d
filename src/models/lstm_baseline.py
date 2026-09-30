"""Bidirectional LSTM baseline model for gait kinetics prediction.

Parameter-matched to GaitS4D (~173k parameters) to provide a rigorous,
apples-to-apples benchmark against conventional recurrent neural networks.
"""

import torch
import torch.nn as nn
from typing import Optional

from src.models.film import FiLM


class GaitLSTM(nn.Module):
    """2-layer Bidirectional LSTM for gait joint-kinetics prediction.
    
    Args:
        in_channels: Input feature dimension (default: 102).
        out_channels: Output target dimension (default: 3).
        hidden_size: Hidden dimension per direction (default: 64 -> 128 bidirectional).
        num_layers: Number of stacked LSTM layers (default: 2).
        dropout: Dropout probability between LSTM layers (default: 0.1).
        use_film: If True, apply FiLM speed conditioning on temporal representations.
    """
    
    def __init__(
        self,
        in_channels: int = 102,
        out_channels: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_film: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_film = use_film
        
        # Linear projection to hidden dimension
        self.in_proj = nn.Linear(in_channels, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 2-layer Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        out_dim = hidden_size * 2  # Bidirectional doubles output feature size
        
        # Optional FiLM speed conditioning
        if use_film:
            self.film = FiLM(cond_dim=1, d_model=out_dim)
        else:
            self.film = None
            
        # Prediction head
        self.norm = nn.LayerNorm(out_dim)
        self.head = nn.Linear(out_dim, out_channels)

    def forward(self, x: torch.Tensor, speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
            speed: Optional walking speed tensor of shape (batch_size, 1) or (batch_size,)
            
        Returns:
            Predicted joint moments of shape (batch_size, seq_len, out_channels)
        """
        # Ensure speed shape is (batch_size, 1) if provided
        if speed is not None and speed.ndim == 1:
            speed = speed.unsqueeze(1)
            
        h = self.dropout(self.in_proj(x))
        h, _ = self.lstm(h)
        
        if self.use_film and self.film is not None and speed is not None:
            h = self.film(h, speed)
            
        h = self.norm(h)
        out = self.head(h)
        return out

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
