"""Automated training and evaluation runner for new baselines and ablations:
1. Model 1.5: S4D + Physics (no FiLM)
2. BiLSTM baseline
3. Dilated TCN baseline

Runs subject split training, test evaluation, and Fukuchi zero-shot evaluation.
Optionally runs LOSO cross-validation folds.
"""

import os
import sys
import json
import time
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.train import run_training
from src.evaluate import evaluate_primary_test_split, evaluate_fukuchi_zero_shot

MODELS = [
    {
        "config": "physics_only",
        "label": "Model 1.5 (S4D + Physics)",
        "save_dir": "experiments/phase3_physics_only",
        "lambda_jerk": 1e-8,
        "lambda_energy": 0.01,
    },
    {
        "config": "lstm",
        "label": "BiLSTM Baseline",
        "save_dir": "experiments/baseline_lstm",
        "lambda_jerk": 0.0,
        "lambda_energy": 0.0,
    },
    {
        "config": "tcn",
        "label": "Dilated TCN Baseline",
        "save_dir": "experiments/baseline_tcn",
        "lambda_jerk": 0.0,
        "lambda_energy": 0.0,
    },
]


def run_subject_split_suite(models=None, epochs=50, device="cpu", seed=42):
    """Train and evaluate models on the primary subject-split."""
    if models is None:
        models = MODELS

    results_summary = {}
    print("==================================================")
    print(f"STARTING SUBJECT-SPLIT EXPERIMENTS (epochs={epochs}, seed={seed}, device={device})")
    print("==================================================")

    for m in models:
        c_name = m["config"]
        c_label = m["label"]
        save_dir = m["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        ckpt_path = os.path.join(save_dir, "model_best.pt")
        metrics_path = os.path.join(save_dir, "train_metrics.json")
        if os.path.exists(ckpt_path) and os.path.exists(metrics_path):
            print(f">>> Found existing trained checkpoint at {ckpt_path}. Loading cached training metrics...")
            with open(metrics_path, "r") as f:
                train_res = json.load(f)
            t_train = train_res.get("total_train_time_seconds", 0.0)
        else:
            print(f"\n==================================================")
            print(f">>> Training {c_label} [{c_name}] on Subject Split...")
            print(f"==================================================")
            t_start = time.time()
            train_res = run_training(
                config=c_name,
                subset_subjects=0,
                split_mode="subject",
                epochs=epochs,
                batch_size=16,
                lr=1e-3,
                lambda_jerk_final=m["lambda_jerk"],
                lambda_energy_final=m["lambda_energy"],
                warmup_epochs=10,
                seed=seed,
                device_name=device,
                save_dir=save_dir,
            )
            t_train = time.time() - t_start
            print(f">>> Training finished in {t_train:.1f}s. Best Val Loss: {train_res['best_val_loss']:.5f}")

        # Decoupled primary test split evaluation
        ckpt_path = os.path.join(save_dir, "model_best.pt")
        print(f">>> Evaluating on primary test split...")
        test_res = evaluate_primary_test_split(
            checkpoint_path=ckpt_path,
            config=c_name,
            split_mode="subject",
            device_name=device,
            save_dir=save_dir,
        )

        # Cross-dataset zero-shot evaluation on Fukuchi
        print(f">>> Evaluating zero-shot on Fukuchi dataset...")
        fukuchi_res = evaluate_fukuchi_zero_shot(
            checkpoint_path=ckpt_path,
            config=c_name,
            device_name=device,
            save_dir=save_dir,
        )

        results_summary[c_name] = {
            "label": c_label,
            "train_time_sec": t_train,
            "best_val_loss": train_res["best_val_loss"],
            "test_overall_mae": test_res["test_overall_mae"],
            "test_overall_rmse": test_res["test_overall_rmse"],
            "test_overall_pearson_r": test_res["test_overall_pearson_r"],
            "hip_mae": test_res["per_joint_metrics"]["Hip Flexion/Extension"]["mae"],
            "hip_r": test_res["per_joint_metrics"]["Hip Flexion/Extension"]["pearson_r"],
            "knee_mae": test_res["per_joint_metrics"]["Knee Flexion/Extension"]["mae"],
            "knee_r": test_res["per_joint_metrics"]["Knee Flexion/Extension"]["pearson_r"],
            "ankle_mae": test_res["per_joint_metrics"]["Ankle Plantar/Dorsiflexion"]["mae"],
            "ankle_r": test_res["per_joint_metrics"]["Ankle Plantar/Dorsiflexion"]["pearson_r"],
            "pred_jerk_rms": test_res["physical_consistency"]["predicted_jerk_rms"],
            "power_residual_rmse": test_res["physical_consistency"]["power_residual_rmse"],
            "fukuchi_mae": fukuchi_res["overall_mae"],
            "fukuchi_rmse": fukuchi_res["overall_rmse"],
            "fukuchi_r": fukuchi_res["overall_pearson_r"],
        }

    # Save summary
    out_summary_path = "experiments/results/baselines_subject_summary.json"
    with open(out_summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved subject-split baselines summary to {out_summary_path}")

    # Print nicely formatted summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<28} | {'Val MSE':<8} | {'Test MAE':<9} | {'Pearson r':<9} | {'Jerk RMS':<9} | {'Fukuchi MAE':<11} | {'Fukuchi r':<9}")
    print("-" * 90)
    for c_name, res in results_summary.items():
        print(f"{res['label']:<28} | {res['best_val_loss']:<8.4f} | {res['test_overall_mae']:<9.4f} | {res['test_overall_pearson_r']:<9.4f} | {res['pred_jerk_rms']:<9.2f} | {res['fukuchi_mae']:<11.4f} | {res['fukuchi_r']:<9.4f}")
    print("=" * 90 + "\n")

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=None, choices=["physics_only", "lstm", "tcn"])
    args = parser.parse_args()

    models_to_run = MODELS
    if args.model is not None:
        models_to_run = [m for m in MODELS if m["config"] == args.model]

    run_subject_split_suite(models=models_to_run, epochs=args.epochs, device=args.device, seed=args.seed)
