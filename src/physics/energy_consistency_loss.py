"""Energy-consistency residual loss for gait kinetics prediction.

Governed by design.md §4 and architecture.md §4.
Penalizes the residual between predicted mechanical joint power
P_pred = sign_j * tau_pred_j * q_dot_j
and the ground-truth inverse dynamics joint power (P_meas) from the dataset.
Uses real radian-based angular velocity (rad/s) and moments (N*m/kg),
producing power residuals in (W/kg)².
"""

import torch
import torch.nn as nn


class EnergyConsistencyLoss(nn.Module):
    """Computes mean squared residual between predicted and reference joint power (W/kg)²."""
    
    def __init__(self, signs: tuple[float, float, float] = (1.0, -1.0, -1.0)):
        """
        Args:
            signs: Coordinate alignment factors for [hip, knee, ankle] sagittal power:
                   (+1.0 for hip flexion, -1.0 for knee flexion, -1.0 for ankle plantarflexion).
        """
        super().__init__()
        self.register_buffer("signs", torch.tensor(signs, dtype=torch.float32).view(1, 1, 3))
        
    def forward(
        self,
        pred_moments: torch.Tensor,
        angular_vel: torch.Tensor,
        power_meas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_moments: Tensor of shape (B, T, 3) representing predicted sagittal
                          moments (hip, knee, ankle in N*m/kg).
            angular_vel: Tensor of shape (B, T, 3) representing measured sagittal
                         angular velocities (rad/s).
            power_meas: Tensor of shape (B, T, 3) representing reference joint mechanical
                        powers (W/kg).
                        
        Returns:
            torch.Tensor: Scalar energy-consistency loss in (W/kg)².
        """
        # Predicted joint power: P_j = sign_j * tau_j * q_dot_j (W/kg)
        pred_power = self.signs * pred_moments * angular_vel
        power_residual = pred_power - power_meas
        return torch.mean(power_residual ** 2)
