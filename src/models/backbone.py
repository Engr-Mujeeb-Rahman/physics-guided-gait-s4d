"""Full Model Architecture: S4D Backbone with optional FiLM conditioning and Linear Output Head.

Supports all three ablation configurations:
1. baseline: use_film=False, plain supervised regression on joint moments.
2. +conditioning: use_film=True (speed conditioning via FiLM).
3. +conditioning+physics: use_film=True (speed conditioning + physics loss terms in training loop).
"""

import torch
import torch.nn as nn
from typing import Optional

from src.models.s4d import S4DBlock
from src.models.film import FiLM


class GaitS4D(nn.Module):
    """S4D-based sequence model for gait kinetics prediction.
    
    Args:
        in_channels: Input feature dimension (default: 102 for full pipeline).
        out_channels: Output target dimension (default: 3 for hip, knee, ankle sagittal moments).
        d_model: Model hidden dimension (default: 64 per design.md §3).
        d_state: SSM state dimension (default: 64).
        n_layers: Number of S4D blocks (default: 4 per design.md §3).
        expand: MLP expansion ratio in each S4D block (default: 2).
        dropout: Dropout rate (default: 0.1).
        use_film: If True, apply FiLM speed conditioning at each block.
    """
    
    def __init__(
        self,
        in_channels: int = 102,
        out_channels: int = 3,
        d_model: int = 64,
        d_state: int = 64,
        n_layers: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_film: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_film = use_film
        
        # Input projection
        self.in_proj = nn.Linear(in_channels, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # S4D blocks and optional FiLM layers
        self.blocks = nn.ModuleList([
            S4DBlock(d_model=d_model, d_state=d_state, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        if use_film:
            self.films = nn.ModuleList([
                FiLM(cond_dim=1, d_model=d_model)
                for _ in range(n_layers)
            ])
        else:
            self.films = None
            
        # Output prediction head
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_channels)

    def forward(self, x: torch.Tensor, speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
            speed: Optional walking speed scalar tensor of shape (batch_size, 1) or (batch_size,)
            
        Returns:
            Predicted joint moments of shape (batch_size, seq_len, out_channels)
        """
        # Ensure speed has shape (batch_size, 1) if provided
        if speed is not None and speed.ndim == 1:
            speed = speed.unsqueeze(1)
            
        # Project inputs to hidden dimension
        h = self.dropout(self.in_proj(x))
        
        # Pass through S4D blocks
        for i, block in enumerate(self.blocks):
            h = block(h)
            if self.use_film and self.films is not None and speed is not None:
                h = self.films[i](h, speed)
                
        # Final norm and linear projection
        h = self.final_norm(h)
        out = self.head(h)
        return out

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
