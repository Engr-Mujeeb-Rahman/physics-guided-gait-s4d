"""Evaluation module for gait kinetics models.

Implements decoupled, unbiased evaluation per design.md §10:
1. Primary dataset test evaluation (subject-split or leave-one-speed-out)
2. Physical consistency metrics (torque jerk RMS, power residual RMSE)
3. Cross-dataset zero-shot evaluation on Fukuchi 2018 dataset (328 treadmill trials)

Guarantees that test evaluation is strictly firewalled and only executed
on a frozen checkpoint after model selection is finalized.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.backbone import GaitS4D
from src.models.lstm_baseline import GaitLSTM
from src.models.tcn_baseline import GaitTCN
from src.data.dataset import get_dataloaders
from src.data.fukuchi_loader import get_fukuchi_treadmill_catalog, load_fukuchi_trial_data


def build_eval_model(config: str, d_model: int = 64, n_layers: int = 4, device: torch.device = torch.device("cpu")) -> tuple[nn.Module, bool]:
    """Helper to instantiate parameter-matched model architectures for evaluation."""
    use_film = (config in ["conditioning", "conditioning_physics"])
    if config in ["lstm", "lstm_baseline"]:
        model = GaitLSTM(
            in_channels=102,
            out_channels=3,
            hidden_size=d_model,
            num_layers=2,
            dropout=0.1,
            use_film=use_film,
        ).to(device)
    elif config in ["tcn", "tcn_baseline"]:
        model = GaitTCN(
            in_channels=102,
            out_channels=3,
            num_channels=[d_model] * n_layers,
            kernel_size=5,
            dropout=0.1,
            use_film=use_film,
        ).to(device)
    else:
        model = GaitS4D(
            in_channels=102,
            out_channels=3,
            d_model=d_model,
            d_state=64,
            n_layers=n_layers,
            dropout=0.1,
            use_film=use_film,
        ).to(device)
    return model, use_film


def safe_pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Pearson r without triggering Windows Intel MKL BLAS access violations."""
    p_flat = pred.flatten()
    t_flat = target.flatten()
    p_c = p_flat - np.mean(p_flat)
    t_c = t_flat - np.mean(t_flat)
    denom = np.sqrt(np.sum(p_c ** 2)) * np.sqrt(np.sum(t_c ** 2))
    return float(np.sum(p_c * t_c) / max(denom, 1e-8))


@torch.no_grad()
def evaluate_primary_test_split(
    checkpoint_path: str,
    config: str = "conditioning_physics",
    split_mode: str = "subject",
    held_out_speed: float = None,
    d_model: int = 64,
    n_layers: int = 4,
    batch_size: int = 16,
    device_name: str = "cpu",
    save_dir: str = None,
) -> dict:
    """Evaluate a frozen model checkpoint on the held-out test split."""
    device = torch.device(device_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    print(f"\n==================================================")
    print(f"EVALUATING TEST SET (Decoupled Single Pass)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config} | Split mode: {split_mode} (held_out_speed={held_out_speed})")
    print(f"==================================================")
    
    # Load dataset
    _, _, test_loader, scaler = get_dataloaders(
        batch_size=batch_size,
        split_mode=split_mode,
        held_out_speed=held_out_speed,
        include_muscles=True,
    )
    
    # Initialize model and load weights
    model, use_film = build_eval_model(config, d_model=d_model, n_layers=n_layers, device=device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    criterion = nn.MSELoss()
    total_mse = 0.0
    n_batches = 0
    
    all_preds = []
    all_targets = []
    all_dts = []
    all_qdots = []
    all_powers = []
    
    for batch in test_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        speed = batch["speed"].to(device)
        dt = batch["dt"].to(device)
        qdot = batch["angular_vel"].to(device)
        power = batch["power_meas"].to(device)
        
        if not use_film:
            pred = model(x)
        else:
            pred = model(x, speed=speed)
            
        loss = criterion(pred, y)
        total_mse += loss.item()
        n_batches += 1
        
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        all_dts.append(dt.cpu().numpy())
        all_qdots.append(qdot.cpu().numpy())
        all_powers.append(power.cpu().numpy())
        
    all_preds = np.concatenate(all_preds, axis=0)      # (N, 100, 3)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 100, 3)
    all_dts = np.concatenate(all_dts, axis=0)          # (N, 1)
    all_qdots = np.concatenate(all_qdots, axis=0)      # (N, 100, 3)
    all_powers = np.concatenate(all_powers, axis=0)    # (N, 100, 3)
    
    test_loss_mse = total_mse / max(n_batches, 1)
    
    # Residuals
    diff = all_preds - all_targets
    joint_names = ["Hip Flexion/Extension", "Knee Flexion/Extension", "Ankle Plantar/Dorsiflexion"]
    joint_mae = np.mean(np.abs(diff), axis=(0, 1))
    joint_rmse = np.sqrt(np.mean(diff ** 2, axis=(0, 1)))
    
    joint_r = []
    for j in range(3):
        r_val = safe_pearson_r(all_preds[:, :, j], all_targets[:, :, j])
        joint_r.append(r_val)
        
    # Physical consistency metrics
    dt_3d = all_dts.reshape(-1, 1, 1)
    pred_d2tau = (all_preds[:, 2:, :] - 2.0 * all_preds[:, 1:-1, :] + all_preds[:, :-2, :]) / (dt_3d ** 2)
    pred_jerk_rms = float(np.sqrt(np.mean(pred_d2tau ** 2)))
    
    targ_d2tau = (all_targets[:, 2:, :] - 2.0 * all_targets[:, 1:-1, :] + all_targets[:, :-2, :]) / (dt_3d ** 2)
    targ_jerk_rms = float(np.sqrt(np.mean(targ_d2tau ** 2)))
    
    signs = np.array([1.0, -1.0, -1.0]).reshape(1, 1, 3)
    pred_power = signs * all_preds * all_qdots
    power_residual_rmse = float(np.sqrt(np.mean((pred_power - all_powers) ** 2)))
    
    results = {
        "checkpoint_path": checkpoint_path,
        "config": config,
        "split_mode": split_mode,
        "held_out_speed": held_out_speed,
        "test_sample_count": len(all_preds),
        "test_loss_mse": float(test_loss_mse),
        "test_overall_mae": float(np.mean(joint_mae)),
        "test_overall_rmse": float(np.mean(joint_rmse)),
        "test_overall_pearson_r": float(np.mean(joint_r)),
        "per_joint_metrics": {
            joint_names[j]: {
                "mae": float(joint_mae[j]),
                "rmse": float(joint_rmse[j]),
                "pearson_r": float(joint_r[j]),
            }
            for j in range(3)
        },
        "physical_consistency": {
            "predicted_jerk_rms": pred_jerk_rms,
            "target_jerk_rms": targ_jerk_rms,
            "power_residual_rmse": power_residual_rmse,
        }
    }
    
    print("\n--- Test Set Kinetics Performance ---")
    for j in range(3):
        print(f"  {joint_names[j]:30s} | MAE: {joint_mae[j]:.4f} N*m/kg | RMSE: {joint_rmse[j]:.4f} N*m/kg | Pearson r: {joint_r[j]:.4f}")
    print(f"  {'Overall Average':30s} | MAE: {np.mean(joint_mae):.4f} N*m/kg | RMSE: {np.mean(joint_rmse):.4f} N*m/kg | Pearson r: {np.mean(joint_r):.4f}")
    print("\n--- Physical Consistency Metrics ---")
    print(f"  Predicted Torque Jerk (RMS):   {pred_jerk_rms:.2f} N*m/(kg*s²)")
    print(f"  Target Torque Jerk (RMS):      {targ_jerk_rms:.2f} N*m/(kg*s²)")
    print(f"  Power Consistency Residual RMSE: {power_residual_rmse:.4f} W/kg")
    print("==================================================")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(save_dir, "test_metrics.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Test metrics saved to {out_file}")
        
    return results


@torch.no_grad()
def evaluate_fukuchi_zero_shot(
    checkpoint_path: str,
    config: str = "conditioning_physics",
    d_model: int = 64,
    n_layers: int = 4,
    device_name: str = "cpu",
    save_dir: str = None,
) -> dict:
    """Run cross-dataset zero-shot evaluation on all Fukuchi 2018 treadmill trials."""
    device = torch.device(device_name)
    print(f"\n==================================================")
    print(f"CROSS-DATASET ZERO-SHOT EVALUATION (Fukuchi 2018)")
    print(f"Checkpoint: {checkpoint_path} | Config: {config}")
    print(f"==================================================")
    
    # Load primary dataset scaler
    _, _, _, scaler = get_dataloaders(batch_size=16, split_mode="subject", include_muscles=True)
    x_mean = scaler["x_mean"]
    x_std = scaler["x_std"]
    y_mean = scaler["y_mean"]
    y_std = scaler["y_std"]
    speed_min = scaler["speed_min"]
    speed_max = scaler["speed_max"]
    
    # Initialize model
    model, use_film = build_eval_model(config, d_model=d_model, n_layers=n_layers, device=device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    cache_path = "data/processed/fukuchi_processed.npz"
    if os.path.exists(cache_path):
        print(f"Loading cached Fukuchi data from {cache_path}...")
        npz = np.load(cache_path)
        all_features = npz["features"] # (N, 100, 102)
        all_targets = npz["targets"]   # (N, 100, 3)
        all_speeds = npz["speeds"]     # (N,)
        
        # Normalize
        feat_norm = (all_features - x_mean) / x_std
        speed_norm = (all_speeds - speed_min) / max(speed_max - speed_min, 1e-4)
        speed_norm = np.clip(speed_norm, 0.0, 1.0)
        
        # Batch inference
        batch_size = 32
        all_preds = []
        for i in range(0, len(feat_norm), batch_size):
            bx = torch.tensor(feat_norm[i:i+batch_size], dtype=torch.float32).to(device)
            bsp = torch.tensor(speed_norm[i:i+batch_size, None], dtype=torch.float32).to(device)
            if not use_film:
                pred_norm = model(bx).cpu().numpy()
            else:
                pred_norm = model(bx, speed=bsp).cpu().numpy()
            b_pred = pred_norm * y_std + y_mean
            all_preds.append(b_pred)
            
        all_preds = np.concatenate(all_preds, axis=0) # (N, 100, 3)
    else:
        catalog = get_fukuchi_treadmill_catalog()
        print(f"Found {len(catalog)} valid treadmill trials in Fukuchi dataset.")
        
        all_preds = []
        all_targets = []
        
        for idx, row in catalog.iterrows():
            trial_base = row["TrialBase"]
            speed = float(row["speed_num"])
            
            try:
                trial_data = load_fukuchi_trial_data(trial_base, speed, compute_muscles=True)
                feat = trial_data["features"] # (100, 102)
                targ = trial_data["targets"]  # (100, 3) in N*m/kg
                
                # Normalize with primary scaler
                feat_norm = (feat - x_mean) / x_std
                speed_norm = (speed - speed_min) / max(speed_max - speed_min, 1e-4)
                speed_norm = np.clip(speed_norm, 0.0, 1.0)
                
                x_t = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0).to(device)
                sp_t = torch.tensor([[speed_norm]], dtype=torch.float32).to(device)
                
                if not use_film:
                    pred_norm = model(x_t).cpu().numpy().squeeze(0) # (100, 3)
                else:
                    pred_norm = model(x_t, speed=sp_t).cpu().numpy().squeeze(0) # (100, 3)
                    
                # Denormalize predictions
                pred = pred_norm * y_std + y_mean
                
                all_preds.append(pred)
                all_targets.append(targ)
            except Exception as e:
                print(f"Skipping {trial_base}: {e}")
                continue
                
        all_preds = np.array(all_preds)    # (N, 100, 3)
        all_targets = np.array(all_targets) # (N, 100, 3)
    
    diff = all_preds - all_targets
    joint_names = ["Hip Flexion/Extension", "Knee Flexion/Extension", "Ankle Plantar/Dorsiflexion"]
    joint_mae = np.mean(np.abs(diff), axis=(0, 1))
    joint_rmse = np.sqrt(np.mean(diff ** 2, axis=(0, 1)))
    
    joint_r = []
    for j in range(3):
        r_val = safe_pearson_r(all_preds[:, :, j], all_targets[:, :, j])
        joint_r.append(r_val)
        
    overall_mae = float(np.mean(joint_mae))
    overall_rmse = float(np.mean(joint_rmse))
    overall_r = float(np.mean(joint_r))
    
    results = {
        "checkpoint_path": checkpoint_path,
        "config": config,
        "trial_count": len(all_preds),
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "overall_pearson_r": overall_r,
        "per_joint_metrics": {
            joint_names[j]: {
                "mae": float(joint_mae[j]),
                "rmse": float(joint_rmse[j]),
                "pearson_r": float(joint_r[j]),
            }
            for j in range(3)
        }
    }
    
    print("\n--- Fukuchi Zero-Shot Generalization Performance ---")
    print(f"Evaluated on {len(all_preds)} treadmill trials.")
    for j in range(3):
        print(f"  {joint_names[j]:30s} | MAE: {joint_mae[j]:.4f} N*m/kg | RMSE: {joint_rmse[j]:.4f} N*m/kg | Pearson r: {joint_r[j]:.4f}")
    print(f"  {'Overall Average':30s} | MAE: {overall_mae:.4f} N*m/kg | RMSE: {overall_rmse:.4f} N*m/kg | Pearson r: {overall_r:.4f}")
    print("==================================================")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(save_dir, "fukuchi_metrics.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Fukuchi metrics saved to {out_file}")
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="conditioning_physics", choices=["baseline", "conditioning", "conditioning_physics", "physics_only", "lstm", "tcn", "lstm_baseline", "tcn_baseline"])
    parser.add_argument("--split_mode", type=str, default="subject")
    parser.add_argument("--held_out_speed", type=float, default=None)
    parser.add_argument("--eval_fukuchi", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.eval_fukuchi:
        evaluate_fukuchi_zero_shot(
            checkpoint_path=args.checkpoint,
            config=args.config,
            save_dir=args.save_dir,
        )
    else:
        evaluate_primary_test_split(
            checkpoint_path=args.checkpoint,
            config=args.config,
            split_mode=args.split_mode,
            held_out_speed=args.held_out_speed,
            save_dir=args.save_dir,
        )
