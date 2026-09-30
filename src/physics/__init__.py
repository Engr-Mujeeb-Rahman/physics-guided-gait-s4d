"""Physics-guided auxiliary loss functions."""

from src.physics.jerk_loss import TorqueJerkLoss
from src.physics.energy_consistency_loss import EnergyConsistencyLoss

__all__ = ["TorqueJerkLoss", "EnergyConsistencyLoss"]
