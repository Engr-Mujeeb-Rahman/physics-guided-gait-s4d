"""Leave-One-Speed-Out (LOSO) cross-validation runner.

Implements Phase 4 speed generalization evaluation per design.md §10:
- Evaluates all 3 ablation models across all 6 treadmill speeds (0.6 to 1.6 m/s).
- Enforces strict zero-speed-leakage: held-out speed is absent from both train and val.
- Checkpoints are selected solely on within-fold validation loss.
- Single test pass on held-out speed per fold.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.train import run_training
from src.evaluate import evaluate_primary_test_split
from src.data.splits import ALL_SPEEDS

CONFIGS = [
    {
        "name": "baseline",
        "label": "Model 1 (Baseline)",
        "lambda_jerk": 0.0,
        "lambda_energy": 0.0,
    },
    {
        "name": "physics_only",
        "label": "Model 1.5 (+Physics Only)",
        "lambda_jerk": 1e-8,
        "lambda_energy": 0.01,
    },
    {
        "name": "conditioning",
        "label": "Model 2 (+Conditioning)",
        "lambda_jerk": 0.0,
        "lambda_energy": 0.0,
    },
    {
        "name": "conditioning_physics",
        "label": "Model 3 (+Conditioning+Physics)",
        "lambda_jerk": 1e-8,
        "lambda_energy": 0.01,
    },
    {
        "name": "lstm",
        "label": "BiLSTM Baseline",
        "lambda_jerk": 0.0,
        "lambda_energy": 0.0,
    },
    {
        "name": "tcn",
        "label": "Dilated TCN Baseline",
        "lambda_jerk": 0.0,
        "lambda_energy": 0.0,
    },
]


def run_single_fold(held_out_speed: float, configs=None, epochs=50, device="cpu", seed=42):
    """Run all specified model configurations on a single held-out speed fold."""
    if configs is None:
        configs = CONFIGS
        
    print(f"\n========================================================")
    print(f"STARTING LOSO FOLD: Held-Out Speed = {held_out_speed:.1f} m/s (Seed: {seed})")
    print(f"========================================================")
    
    fold_results = {}
    
    for cfg in configs:
        cfg_name = cfg["name"]
        save_dir = f"experiments/loso/{cfg_name}_speed_{held_out_speed:.1f}"
        os.makedirs(save_dir, exist_ok=True)
        best_ckpt = os.path.join(save_dir, "model_best.pt")
        test_metric_file = os.path.join(save_dir, "test_metrics.json")
        
        if os.path.exists(best_ckpt) and os.path.exists(test_metric_file):
            print(f"\n>>> [Fold {held_out_speed:.1f} m/s] Found existing results for {cfg['label']}, loading cached metrics...")
            with open(test_metric_file, "r") as f:
                test_res = json.load(f)
            train_metric_file = os.path.join(save_dir, "train_metrics.json")
            best_val = 0.0
            if os.path.exists(train_metric_file):
                with open(train_metric_file, "r") as f:
                    best_val = json.load(f).get("best_val_loss", 0.0)
            train_res = {"best_val_loss": best_val}
        else:
            print(f"\n>>> [Fold {held_out_speed:.1f} m/s] Training {cfg['label']}...")
            train_res = run_training(
                config=cfg_name,
                subset_subjects=0,
                split_mode="loso",
                held_out_speed=held_out_speed,
                epochs=epochs,
                batch_size=16,
                lr=1e-3,
                lambda_jerk_final=cfg["lambda_jerk"],
                lambda_energy_final=cfg["lambda_energy"],
                warmup_epochs=10,
                device_name=device,
                seed=seed,
                save_dir=save_dir,
            )
            print(f">>> [Fold {held_out_speed:.1f} m/s] Training complete. Best Val Loss = {train_res['best_val_loss']:.5f}")
            
            # Single decoupled test evaluation on held-out speed
            test_res = evaluate_primary_test_split(
                checkpoint_path=best_ckpt,
                config=cfg_name,
                split_mode="loso",
                held_out_speed=held_out_speed,
                device_name=device,
                save_dir=save_dir,
            )
        
        fold_results[cfg_name] = {
            "best_val_loss": train_res["best_val_loss"],
            "test_overall_mae": test_res["test_overall_mae"],
            "test_overall_rmse": test_res["test_overall_rmse"],
            "test_overall_pearson_r": test_res["test_overall_pearson_r"],
            "hip_mae": test_res["per_joint_metrics"]["Hip Flexion/Extension"]["mae"],
            "hip_rmse": test_res["per_joint_metrics"]["Hip Flexion/Extension"]["rmse"],
            "hip_r": test_res["per_joint_metrics"]["Hip Flexion/Extension"]["pearson_r"],
            "knee_mae": test_res["per_joint_metrics"]["Knee Flexion/Extension"]["mae"],
            "knee_rmse": test_res["per_joint_metrics"]["Knee Flexion/Extension"]["rmse"],
            "knee_r": test_res["per_joint_metrics"]["Knee Flexion/Extension"]["pearson_r"],
            "ankle_mae": test_res["per_joint_metrics"]["Ankle Plantar/Dorsiflexion"]["mae"],
            "ankle_rmse": test_res["per_joint_metrics"]["Ankle Plantar/Dorsiflexion"]["rmse"],
            "ankle_r": test_res["per_joint_metrics"]["Ankle Plantar/Dorsiflexion"]["pearson_r"],
            "pred_jerk_rms": test_res["physical_consistency"]["predicted_jerk_rms"],
            "power_residual_rmse": test_res["physical_consistency"]["power_residual_rmse"],
        }
        
    return fold_results


def run_full_loso(epochs=50, device="cpu", selected_speeds=None, selected_models=None, seed=42):
    """Run full LOSO evaluation across all 6 treadmill walking speeds."""
    speeds = selected_speeds if selected_speeds is not None else ALL_SPEEDS
    active_configs = CONFIGS
    if selected_models is not None:
        active_configs = [c for c in CONFIGS if c["name"] in selected_models]
        
    summary_path = "experiments/results/loso_summary.json"
    all_fold_results = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                all_fold_results = json.load(f)
        except Exception:
            all_fold_results = {}
    
    t_start = time.time()
    for speed in speeds:
        sp_key = f"{speed:.1f}"
        fold_res = run_single_fold(speed, configs=active_configs, epochs=epochs, device=device, seed=seed)
        if sp_key not in all_fold_results:
            all_fold_results[sp_key] = {}
        all_fold_results[sp_key].update(fold_res)
        
    total_time = time.time() - t_start
    print(f"\n========================================================")
    print(f"ALL LOSO FOLDS COMPLETED in {total_time/60:.2f} minutes.")
    print(f"========================================================")
    
    # Save full results json
    os.makedirs("experiments/results", exist_ok=True)
    summary_path = "experiments/results/loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_fold_results, f, indent=2)
    print(f"Full LOSO results saved to {summary_path}")
    
    # Build summary table
    rows = []
    for sp_str, sp_data in all_fold_results.items():
        for cfg in CONFIGS:
            c_name = cfg["name"]
            c_res = sp_data[c_name]
            rows.append({
                "Speed (m/s)": sp_str,
                "Model": cfg["label"],
                "Val MSE": c_res["best_val_loss"],
                "Test MAE (N*m/kg)": c_res["test_overall_mae"],
                "Test RMSE (N*m/kg)": c_res["test_overall_rmse"],
                "Test Pearson r": c_res["test_overall_pearson_r"],
                "Jerk RMS": c_res["pred_jerk_rms"],
                "Power Residual": c_res["power_residual_rmse"],
            })
            
    df = pd.DataFrame(rows)
    csv_path = "experiments/results/loso_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary table saved to {csv_path}")
    
    # Compute cross-speed mean per model
    print("\n=== LOSO SPEED GENERALIZATION SUMMARY (Cross-Speed Means) ===")
    mean_df = df.groupby("Model")[["Test MAE (N*m/kg)", "Test RMSE (N*m/kg)", "Test Pearson r", "Jerk RMS", "Power Residual"]].mean()
    print(mean_df.to_string())
    print("========================================================\n")
    
    return all_fold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--speed", type=float, default=None, help="Run single speed fold only (e.g. 0.6)")
    parser.add_argument("--speeds", type=float, nargs="+", default=None, help="Run specific speeds (e.g. --speeds 0.6 0.8 1.0)")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Models to run (e.g. --models physics_only lstm tcn)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for weight init and training")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads per worker")
    args = parser.parse_args()
    
    import torch
    torch.set_num_threads(args.threads)
    
    if args.speeds is not None:
        run_full_loso(epochs=args.epochs, device=args.device, selected_speeds=args.speeds, selected_models=args.models, seed=args.seed)
    elif args.speed is not None:
        run_full_loso(epochs=args.epochs, device=args.device, selected_speeds=[args.speed], selected_models=args.models, seed=args.seed)
    else:
        run_full_loso(epochs=args.epochs, device=args.device, selected_models=args.models, seed=args.seed)
