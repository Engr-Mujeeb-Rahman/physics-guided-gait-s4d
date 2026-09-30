"""Temporal Convolutional Network (TCN) baseline for gait kinetics prediction.

Parameter-matched to GaitS4D (~171k parameters) to provide a rigorous benchmark
against dilated convolutional sequence models (Bai et al., 2018).
"""

import torch
import torch.nn as nn
from typing import Optional, List

from src.models.film import FiLM


class TemporalBlock(nn.Module):
    """Residual block with dilated 1D convolutions and weight normalization."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x shape: (B, C, L)"""
        out = self.net(x)
        # Ensure sequence length matches after padded dilated convolutions
        if out.shape[-1] != x.shape[-1]:
            out = out[:, :, :x.shape[-1]]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class GaitTCN(nn.Module):
    """4-layer Dilated Temporal Convolutional Network for gait kinetics.
    
    Args:
        in_channels: Input feature dimension (default: 102).
        out_channels: Output target dimension (default: 3).
        num_channels: List of channel widths per level (default: [64, 64, 64, 64]).
        kernel_size: Convolution kernel size (default: 5).
        dropout: Dropout rate (default: 0.1).
        use_film: If True, apply FiLM speed conditioning at output.
    """
    
    def __init__(
        self,
        in_channels: int = 102,
        out_channels: int = 3,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 5,
        dropout: float = 0.1,
        use_film: bool = False,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64, 64]
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_film = use_film
        
        # Linear projection on input
        self.in_proj = nn.Linear(in_channels, num_channels[0])
        self.dropout = nn.Dropout(dropout)
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = num_channels[0] if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size // 2
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )
            
        self.network = nn.ModuleList(layers)
        
        if use_film:
            self.film = FiLM(cond_dim=1, d_model=num_channels[-1])
        else:
            self.film = None
            
        self.norm = nn.LayerNorm(num_channels[-1])
        self.head = nn.Linear(num_channels[-1], out_channels)

    def forward(self, x: torch.Tensor, speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
            speed: Optional walking speed tensor of shape (batch_size, 1)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, out_channels)
        """
        if speed is not None and speed.ndim == 1:
            speed = speed.unsqueeze(1)
            
        # Project inputs: (B, L, C_in) -> (B, L, C_hidden)
        h = self.dropout(self.in_proj(x))
        
        # Transpose for Conv1d: (B, C_hidden, L)
        h = h.transpose(1, 2)
        for block in self.network:
            h = block(h)
            
        # Transpose back: (B, L, C_hidden)
        h = h.transpose(1, 2)
        
        if self.use_film and self.film is not None and speed is not None:
            h = self.film(h, speed)
            
        h = self.norm(h)
        out = self.head(h)
        return out

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
