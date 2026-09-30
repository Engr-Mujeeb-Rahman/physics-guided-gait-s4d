"""Diagonal State Space (S4D) layer in pure PyTorch.

Implements the S4D (Structured State Space Diagonal) model architecture
per Gu et al. (2022) "On the Parameterization and Initialization of Diagonal State Space Models".
Uses pure PyTorch operations (no custom CUDA kernels or external compilation required).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class S4DKernel(nn.Module):
    """Generates the 1D convolution filter K of length L from diagonal SSM parameters."""
    
    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Timescale parameter log(dt)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        
        # Real part of A: constrained strictly negative via -exp(A_real)
        # Initialized to -0.5 (standard S4D-Lin init)
        self.A_real = nn.Parameter(-0.5 * torch.ones(d_model, d_state))
        
        # Imaginary part of A: initialized with HiPPO frequencies pi * n
        self.A_imag = nn.Parameter(math.pi * torch.arange(d_state, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1))
        
        # Input projection B (complex): initialized ~ N(0, 1) / sqrt(d_state)
        self.B = nn.Parameter(torch.randn(d_model, d_state, 2) / math.sqrt(d_state))
        
        # Output projection C (complex): initialized ~ N(0, 1) / sqrt(d_state)
        self.C = nn.Parameter(torch.randn(d_model, d_state, 2) / math.sqrt(d_state))

    def forward(self, L: int) -> torch.Tensor:
        """Compute the convolution kernel of length L.
        
        Returns:
            K: (d_model, L) real-valued convolution kernel
        """
        # Shape: (d_model, 1)
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        
        # Complex A: (d_model, d_state)
        A = -torch.exp(self.A_real) + 1j * self.A_imag
        
        # Complex B and C: (d_model, d_state)
        B_c = torch.view_as_complex(self.B)
        C_c = torch.view_as_complex(self.C)
        
        # Zero-Order Hold (ZOH) discretization
        # dtA: (d_model, d_state)
        dtA = dt * A
        exp_dtA = torch.exp(dtA)
        
        # B_bar = (exp(dtA) - 1) / A * B
        B_bar = ((exp_dtA - 1.0) / A) * B_c
        
        # v = C * B_bar: (d_model, d_state)
        v = C_c * B_bar
        
        # Evaluate power series across sequence length L
        # t: (L, 1, 1)
        t = torch.arange(L, dtype=torch.float32, device=self.A_real.device).unsqueeze(-1).unsqueeze(-1)
        
        # exp_t_dtA: (L, d_model, d_state)
        exp_t_dtA = torch.exp(t * dtA.unsqueeze(0))
        
        # Sum over state dimension: (L, d_model)
        K = 2.0 * torch.sum(v.unsqueeze(0) * exp_t_dtA, dim=-1).real
        
        # Transpose to (d_model, L)
        return K.transpose(0, 1)


class S4D(nn.Module):
    """Single S4D channel-mixing state space layer."""
    
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.kernel = S4DKernel(d_model, d_state)
        self.D = nn.Parameter(torch.randn(d_model))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass for sequence tensor.
        
        Args:
            u: (batch_size, seq_len, d_model)
            
        Returns:
            y: (batch_size, seq_len, d_model)
        """
        B, L, H = u.shape
        # Transpose to (B, H, L) for 1D convolution
        u_conv = u.transpose(1, 2)
        
        # Generate kernel: (H, L) -> (H, 1, L) for grouped conv1d
        K = self.kernel(L).unsqueeze(1)
        
        # Causal convolution with padding L - 1
        y_conv = F.conv1d(u_conv, K, padding=L - 1, groups=H)[:, :, :L]
        
        # Add direct skip D * u
        y = y_conv + self.D.unsqueeze(-1) * u_conv
        
        # Transpose back to (B, L, H) and apply dropout
        return self.dropout(y.transpose(1, 2))


class S4DBlock(nn.Module):
    """Full S4D Block with Pre-LayerNorm, Feedforward Projection, and Residual Connection."""
    
    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s4d = S4D(d_model, d_state=d_state, dropout=dropout)
        self.act = nn.GELU()
        
        # Pointwise MLP expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_model, expand * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expand * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Residual branch 1: S4D state space
        norm_x = self.norm(x)
        h = x + self.s4d(norm_x)
        
        # Residual branch 2: Pointwise FFN
        out = h + self.ffn(self.act(h))
        return out
