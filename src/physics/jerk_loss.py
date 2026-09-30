"""Torque jerk minimization loss for gait kinetics prediction.

Governed by design.md §4 and architecture.md §4.
Penalizes the discrete second time-derivative of predicted joint moments:
d²tau/dt² = (tau[t+1] - 2*tau[t] + tau[t-1]) / dt²
using the real per-trial physical stride duration (dt = stride_duration / 99).
Encourages smooth, physiologically plausible torque trajectories without
distorting high-frequency physiological impact peaks.
"""

import torch
import torch.nn as nn


class TorqueJerkLoss(nn.Module):
    """Computes mean squared torque jerk (N*m / (kg * s²))² across the gait cycle."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_moments: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_moments: Tensor of shape (B, T, 3) representing predicted sagittal
                          moments (hip, knee, ankle in N*m/kg).
            dt: Tensor of shape (B,) or (B, 1) or (B, 1, 1) representing the real physical
                time spacing in seconds for each trial in the batch.
                
        Returns:
            torch.Tensor: Scalar jerk loss.
        """
        # Central second difference along time dimension T (dimension 1)
        # d²tau shape: (B, T - 2, 3)
        d2tau = pred_moments[:, 2:, :] - 2.0 * pred_moments[:, 1:-1, :] + pred_moments[:, :-2, :]
        
        # Ensure dt has shape (B, 1, 1) for broadcast division
        if dt.dim() == 1:
            dt = dt.view(-1, 1, 1)
        elif dt.dim() == 2:
            dt = dt.view(-1, 1, 1)
            
        dt_sq = torch.clamp(dt ** 2, min=1e-8)
        jerk = d2tau / dt_sq
        
        return torch.mean(jerk ** 2)
