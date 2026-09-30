"""Training script for gait kinetics prediction.

Supports 3-way ablation configurations via --config flag:
1. baseline: S4D backbone, no speed conditioning, no physics loss (plain MSE on moments).
2. conditioning: baseline + FiLM speed conditioning.
3. conditioning_physics: baseline + FiLM + jerk penalty and energy-consistency loss.

Also supports small-subset overfitting test (--subset_subjects N) to verify convergence
and measure per-epoch wall-clock training time on CPU/GPU.
"""

import os
import sys
import time
import json
import argparse
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Suppress duplicate OpenMP runtime error on Windows if PyTorch and OpenSim both loaded
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import get_dataloaders, GaitDataset
from src.data.splits import get_subject_splits
from src.models.backbone import GaitS4D
from src.models.lstm_baseline import GaitLSTM
from src.models.tcn_baseline import GaitTCN
from src.physics.jerk_loss import TorqueJerkLoss
from src.physics.energy_consistency_loss import EnergyConsistencyLoss


def get_physics_weights(
    epoch: int,
    warmup_epochs: int = 10,
    lambda_jerk_final: float = 1e-8,
    lambda_energy_final: float = 0.01,
) -> tuple[float, float]:
    """Compute linearly ramped physics loss weights."""
    ramp = min(1.0, float(epoch) / float(warmup_epochs))
    return lambda_jerk_final * ramp, lambda_energy_final * ramp


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    jerk_criterion: nn.Module,
    energy_criterion: nn.Module,
    device: torch.device,
    config: str = "baseline",
    lambda_jerk: float = 0.0,
    lambda_energy: float = 0.0,
) -> dict:
    """Train for one epoch and return loss breakdown and wall-clock duration."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_jerk = 0.0
    total_energy = 0.0
    total_w_jerk = 0.0
    total_w_energy = 0.0
    n_batches = 0
    
    t_start = time.perf_counter()
    for batch in dataloader:
        x = batch["x"].to(device)                      # (B, 100, 102)
        y = batch["y"].to(device)                      # (B, 100, 3)
        speed = batch["speed"].to(device)              # (B, 1)
        
        optimizer.zero_grad()
        
        # Forward pass
        if config in ["baseline", "physics_only", "lstm", "tcn", "lstm_baseline", "tcn_baseline"]:
            pred = model(x)
        else:
            pred = model(x, speed=speed)
            
        # Task MSE loss on joint moments
        mse_loss = criterion(pred, y)
        
        # Auxiliary physics losses
        if config in ["conditioning_physics", "physics_only"]:
            dt = batch["dt"].to(device)
            angular_vel = batch["angular_vel"].to(device)
            power_meas = batch["power_meas"].to(device)
            
            l_jerk = jerk_criterion(pred, dt)
            l_energy = energy_criterion(pred, angular_vel, power_meas)
            
            w_jerk = lambda_jerk * l_jerk
            w_energy = lambda_energy * l_energy
            
            loss = mse_loss + w_jerk + w_energy
            
            total_jerk += l_jerk.item()
            total_energy += l_energy.item()
            total_w_jerk += w_jerk.item()
            total_w_energy += w_energy.item()
        else:
            loss = mse_loss
            
        loss.backward()
        # Gradient clipping for SSM stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse_loss.item()
        n_batches += 1
        
    t_end = time.perf_counter()
    duration = t_end - t_start
    nb = max(n_batches, 1)
    
    return {
        "loss": total_loss / nb,
        "mse": total_mse / nb,
        "jerk_loss": total_jerk / nb,
        "energy_loss": total_energy / nb,
        "w_jerk_loss": total_w_jerk / nb,
        "w_energy_loss": total_w_energy / nb,
        "duration": duration,
    }


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: str = "baseline",
) -> float:
    """Evaluate MSE loss on a dataloader."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        speed = batch["speed"].to(device)
        
        if config in ["baseline", "physics_only", "lstm", "tcn", "lstm_baseline", "tcn_baseline"]:
            pred = model(x)
        else:
            pred = model(x, speed=speed)
            
        loss = criterion(pred, y)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def run_training(
    config: str = "baseline",
    subset_subjects: int = 0,
    split_mode: str = "subject",
    held_out_speed: float = None,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    d_model: int = 64,
    n_layers: int = 4,
    lambda_jerk_final: float = 1e-8,
    lambda_energy_final: float = 0.01,
    warmup_epochs: int = 10,
    seed: int = 42,
    device_name: str = "cpu",
    save_dir: str = "experiments/test_run",
):
    """Main training routine with per-component logging and validation evaluation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_name)
    print(f"==================================================")
    print(f"Starting Training: config={config}, split_mode={split_mode} (held_out_speed={held_out_speed}), seed={seed}")
    print(f"subset_subjects={subset_subjects} (0=full), epochs={epochs}, batch_size={batch_size}")
    if config in ["conditioning_physics", "physics_only"]:
        print(f"Physics losses: lambda_jerk={lambda_jerk_final}, lambda_energy={lambda_energy_final}, warmup={warmup_epochs} epochs")
    print(f"==================================================")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader, stats = get_dataloaders(
        batch_size=batch_size,
        split_mode=split_mode,
        held_out_speed=held_out_speed,
        include_muscles=True,
    )
    
    # If small-subset test is requested
    if subset_subjects > 0:
        splits = get_subject_splits()
        train_subjects = splits["train"]
        selected_subjects = set(train_subjects[:subset_subjects])
        print(f"Small-subset mode enabled: using {subset_subjects} subjects: {selected_subjects}")
        
        full_train_ds = train_loader.dataset
        subset_indices = [
            i for i, s in enumerate(full_train_ds.samples)
            if s["subject"] in selected_subjects
        ]
        subset_ds = Subset(full_train_ds, subset_indices)
        train_loader = DataLoader(
            subset_ds,
            batch_size=min(batch_size, len(subset_indices)),
            shuffle=True,
        )
        print(f"Subset dataset size: {len(subset_ds)} samples across {len(train_loader)} batch(es).")
    
    # Initialize model
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
        model_name = "GaitLSTM"
    elif config in ["tcn", "tcn_baseline"]:
        model = GaitTCN(
            in_channels=102,
            out_channels=3,
            num_channels=[d_model] * n_layers,
            kernel_size=5,
            dropout=0.1,
            use_film=use_film,
        ).to(device)
        model_name = "GaitTCN"
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
        model_name = "GaitS4D"
    
    param_count = model.get_num_params()
    print(f"Initialized {model_name} (config={config}, use_film={use_film}): {param_count:,} parameters.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    jerk_criterion = TorqueJerkLoss().to(device)
    energy_criterion = EnergyConsistencyLoss().to(device)
    
    best_val_loss = float("inf")
    epoch_times = []
    train_losses = []
    train_mse_losses = []
    train_w_jerk_losses = []
    train_w_energy_losses = []
    val_losses = []
    
    print("\n--- Training Loop Starting ---")
    for epoch in range(1, epochs + 1):
        if config in ["conditioning_physics", "physics_only"]:
            l_jerk_w, l_energy_w = get_physics_weights(
                epoch=epoch,
                warmup_epochs=warmup_epochs,
                lambda_jerk_final=lambda_jerk_final,
                lambda_energy_final=lambda_energy_final,
            )
        else:
            l_jerk_w, l_energy_w = 0.0, 0.0
            
        epoch_res = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            jerk_criterion=jerk_criterion,
            energy_criterion=energy_criterion,
            device=device,
            config=config,
            lambda_jerk=l_jerk_w,
            lambda_energy=l_energy_w,
        )
        scheduler.step()
        
        loss = epoch_res["loss"]
        duration = epoch_res["duration"]
        epoch_times.append(duration)
        train_losses.append(loss)
        train_mse_losses.append(epoch_res["mse"])
        train_w_jerk_losses.append(epoch_res["w_jerk_loss"])
        train_w_energy_losses.append(epoch_res["w_energy_loss"])
        
        # Evaluate validation loss (task MSE)
        v_loss = evaluate_loss(model, val_loader, criterion, device, config=config)
        val_losses.append(v_loss)
        
        # Save best model checkpoint
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pt"))
            
        # Logging: For physics runs, log every single epoch with component breakdown
        if config in ["conditioning_physics", "physics_only"]:
            print(
                f"Epoch {epoch:3d}/{epochs:3d} | Total: {loss:.5f} | "
                f"MSE: {epoch_res['mse']:.5f} | "
                f"lambda_j*L_jerk: {epoch_res['w_jerk_loss']:.5f} | "
                f"lambda_e*L_energy: {epoch_res['w_energy_loss']:.5f} | "
                f"Val MSE: {v_loss:.5f} | Time: {duration:.2f} s"
            )
        else:
            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch {epoch:3d}/{epochs:3d} | Train Loss: {loss:.5f} | Val Loss: {v_loss:.5f} | Wall Time: {duration*1000:6.1f} ms ({duration:.3f} s)")
            
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "model_last.pt"))
    
    # Final epoch completed
    mean_epoch_time = float(np.mean(epoch_times))
    median_epoch_time = float(np.median(epoch_times))
    total_train_time = float(np.sum(epoch_times))
    
    print("\n==================================================")
    print(f"Training Complete ({epochs} epochs in {total_train_time:.1f} s)")
    print(f"Initial Train Loss (Epoch 1):  {train_losses[0]:.5f}")
    print(f"Final Train Loss (Epoch {epochs}):    {train_losses[-1]:.5f}")
    print(f"Best Val Loss (MSE):           {best_val_loss:.5f}")
    print(f"Mean Wall-Clock Time per Epoch:{mean_epoch_time:.4f} s ({mean_epoch_time*1000:.1f} ms)")
    print(f"Best checkpoint saved to:      {os.path.join(save_dir, 'model_best.pt')}")
    print("NOTE: Test evaluation decoupled per design.md §10 anti-leakage protocol.")
    print("==================================================")
    
    # Save training metrics log (strictly training + validation, zero test metrics)
    metrics = {
        "config": config,
        "seed": seed,
        "device": device_name,
        "subset_subjects": subset_subjects,
        "epochs": epochs,
        "batch_size": batch_size,
        "param_count": param_count,
        "total_train_time_seconds": total_train_time,
        "mean_epoch_time_seconds": mean_epoch_time,
        "median_epoch_time_seconds": median_epoch_time,
        "initial_train_loss": float(train_losses[0]),
        "final_train_loss": float(train_losses[-1]),
        "best_val_loss": float(best_val_loss),
        "all_epoch_times_seconds": [float(t) for t in epoch_times],
        "train_losses": [float(l) for l in train_losses],
        "train_mse_losses": [float(l) for l in train_mse_losses],
        "train_w_jerk_losses": [float(l) for l in train_w_jerk_losses],
        "train_w_energy_losses": [float(l) for l in train_w_energy_losses],
        "val_losses": [float(l) for l in val_losses],
    }
    
    log_path = os.path.join(save_dir, "train_metrics.json")
    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Validation metrics saved to {log_path}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="baseline", choices=["baseline", "conditioning", "conditioning_physics", "physics_only", "lstm", "tcn", "lstm_baseline", "tcn_baseline"])
    parser.add_argument("--subset_subjects", type=int, default=0, help="Number of subjects for small-subset overfitting (0 for all)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_jerk", type=float, default=1e-8)
    parser.add_argument("--lambda_energy", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_mode", type=str, default="subject", choices=["subject", "loso"])
    parser.add_argument("--held_out_speed", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default="experiments/test_run")
    args = parser.parse_args()
    
    run_training(
        config=args.config,
        subset_subjects=args.subset_subjects,
        split_mode=args.split_mode,
        held_out_speed=args.held_out_speed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_jerk_final=args.lambda_jerk,
        lambda_energy_final=args.lambda_energy,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        device_name=args.device,
        save_dir=args.save_dir,
    )
