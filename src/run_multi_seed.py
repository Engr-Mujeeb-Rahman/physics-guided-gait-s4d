"""Multi-Seed Experiment Runner & Aggregator.
Trains and evaluates all 6 models across 3 random seeds (42, 123, 456)
to obtain mean +/- std estimates for Table I in the manuscript.

Models:
1. BiLSTM (lstm)
2. Dilated TCN (tcn)
3. Model 1: S4D Baseline (baseline)
4. Model 1.5: S4D + Phys (physics_only)
5. Model 2: S4D + FiLM (conditioning)
6. Model 3: S4D + FiLM + Phys (conditioning_physics)
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set optimal CPU thread count
torch.set_num_threads(6)

from src.train import run_training
from src.evaluate import evaluate_primary_test_split

MODELS = [
    {
        "key": "lstm",
        "config": "lstm",
        "label": "BiLSTM",
        "lambda_jerk": 0.0,
        "lambda_energy": 0.0,
        "seed42_dir": "experiments/baseline_lstm",
    },
    {
        "key": "tcn",
        "config": "tcn",
        "label": "Dilated TCN",
        "lambda_jerk": 0.0,
        "lambda_energy": 0.0,
        "seed42_dir": "experiments/baseline_tcn",
    },
    {
        "key": "baseline",
        "config": "baseline",
        "label": "Model 1 (Baseline)",
        "lambda_jerk": 0.0,
        "lambda_energy": 0.0,
        "seed42_dir": "experiments/phase2_baseline",
    },
    {
        "key": "physics_only",
        "config": "physics_only",
        "label": "Model 1.5 (+Phys.)",
        "lambda_jerk": 1e-8,
        "lambda_energy": 0.01,
        "seed42_dir": "experiments/phase3_physics_only",
    },
    {
        "key": "conditioning",
        "config": "conditioning",
        "label": "Model 2 (+Cond.)",
        "lambda_jerk": 0.0,
        "lambda_energy": 0.0,
        "seed42_dir": "experiments/phase3_conditioning",
    },
    {
        "key": "conditioning_physics",
        "config": "conditioning_physics",
        "label": "Model 3 (+Cond.+Phys.)",
        "lambda_jerk": 1e-8,
        "lambda_energy": 0.01,
        "seed42_dir": "experiments/blind_candidate_3",
    },
]

SEEDS = [42, 123, 456]


def run_multi_seed_suite(epochs=50, device="cpu"):
    """Execute training and evaluation across seeds 42, 123, 456."""
    base_dir = "experiments/multi_seed"
    os.makedirs(base_dir, exist_ok=True)

    all_seed_results = {m["key"]: [] for m in MODELS}

    for seed in SEEDS:
        print("\n" + "=" * 70)
        print(f"   STARTING SEED {seed}")
        print("=" * 70)

        seed_dir = os.path.join(base_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for m in MODELS:
            m_key = m["key"]
            m_cfg = m["config"]
            m_lbl = m["label"]
            save_dir = os.path.join(seed_dir, m_key)
            os.makedirs(save_dir, exist_ok=True)

            test_metrics_path = os.path.join(save_dir, "test_metrics.json")
            ckpt_path = os.path.join(save_dir, "model_best.pt")

            # Check if seed 42 can be imported from existing run
            if seed == 42 and not os.path.exists(test_metrics_path):
                src_test = os.path.join(m["seed42_dir"], "test_metrics.json")
                src_ckpt = os.path.join(m["seed42_dir"], "model_best.pt")
                if os.path.exists(src_test) and os.path.exists(src_ckpt):
                    print(f"[{m_lbl} | Seed {seed}] Copying existing checkpoint & metrics from {m['seed42_dir']}...")
                    shutil.copy2(src_test, test_metrics_path)
                    shutil.copy2(src_ckpt, ckpt_path)

            # If metrics already exist, load them
            if os.path.exists(test_metrics_path):
                print(f"[{m_lbl} | Seed {seed}] Already evaluated. Loading {test_metrics_path}...")
                with open(test_metrics_path, "r") as f:
                    test_res = json.load(f)
            else:
                # Train if checkpoint doesn't exist
                if not os.path.exists(ckpt_path):
                    print(f"\n[{m_lbl} | Seed {seed}] Training config '{m_cfg}' for {epochs} epochs...")
                    t0 = time.time()
                    run_training(
                        config=m_cfg,
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
                    t_train = time.time() - t0
                    print(f"[{m_lbl} | Seed {seed}] Training completed in {t_train:.1f}s.")

                # Decoupled primary test split evaluation
                print(f"[{m_lbl} | Seed {seed}] Evaluating on held-out test split...")
                test_res = evaluate_primary_test_split(
                    checkpoint_path=ckpt_path,
                    config=m_cfg,
                    split_mode="subject",
                    device_name=device,
                    save_dir=save_dir,
                )

            # Store metrics for aggregation
            metrics_dict = {
                "seed": seed,
                "mae": test_res["test_overall_mae"],
                "rmse": test_res["test_overall_rmse"],
                "pearson_r": test_res["test_overall_pearson_r"],
                "hip_mae": test_res["per_joint_metrics"]["Hip Flexion/Extension"]["mae"],
                "knee_mae": test_res["per_joint_metrics"]["Knee Flexion/Extension"]["mae"],
                "ankle_mae": test_res["per_joint_metrics"]["Ankle Plantar/Dorsiflexion"]["mae"],
                "jerk_rms": test_res["physical_consistency"]["predicted_jerk_rms"],
                "power_residual_rmse": test_res["physical_consistency"]["power_residual_rmse"],
            }
            all_seed_results[m_key].append(metrics_dict)
            print(f"[{m_lbl} | Seed {seed}] MAE: {metrics_dict['mae']:.4f} | Jerk: {metrics_dict['jerk_rms']:.2f}")

    # Aggregation across seeds
    print("\n" + "=" * 70)
    print("   AGGREGATING RESULTS ACROSS SEEDS")
    print("=" * 70)

    summary = {}
    baseline_jerk_mean = np.mean([r["jerk_rms"] for r in all_seed_results["baseline"]])

    for m in MODELS:
        m_key = m["key"]
        runs = all_seed_results[m_key]

        maes = [r["mae"] for r in runs]
        rmses = [r["rmse"] for r in runs]
        rs = [r["pearson_r"] for r in runs]
        hip_maes = [r["hip_mae"] for r in runs]
        knee_maes = [r["knee_mae"] for r in runs]
        ankle_maes = [r["ankle_mae"] for r in runs]
        jerks = [r["jerk_rms"] for r in runs]
        powers = [r["power_residual_rmse"] for r in runs]

        jerk_mean = float(np.mean(jerks))
        jerk_red = float((baseline_jerk_mean - jerk_mean) / baseline_jerk_mean * 100.0)

        summary[m_key] = {
            "label": m["label"],
            "seed_count": len(runs),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes, ddof=1)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses, ddof=1)),
            "r_mean": float(np.mean(rs)),
            "r_std": float(np.std(rs, ddof=1)),
            "hip_mae_mean": float(np.mean(hip_maes)),
            "hip_mae_std": float(np.std(hip_maes, ddof=1)),
            "knee_mae_mean": float(np.mean(knee_maes)),
            "knee_mae_std": float(np.std(knee_maes, ddof=1)),
            "ankle_mae_mean": float(np.mean(ankle_maes)),
            "ankle_mae_std": float(np.std(ankle_maes, ddof=1)),
            "jerk_rms_mean": jerk_mean,
            "jerk_rms_std": float(np.std(jerks, ddof=1)),
            "jerk_reduction_percent": jerk_red,
            "power_rmse_mean": float(np.mean(powers)),
            "power_rmse_std": float(np.std(powers, ddof=1)),
            "raw_runs": runs,
        }

    out_file = "experiments/results/multi_seed_summary.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMulti-seed summary saved to {out_file}")

    # Print Table
    print("\n" + "=" * 105)
    print(f"{'Model':<22} | {'MAE':<16} | {'RMSE':<16} | {'Pearson r':<16} | {'Jerk RMS':<16} | {'Jerk Red. %'}")
    print("-" * 105)
    for m in MODELS:
        s = summary[m["key"]]
        mae_str = f"{s['mae_mean']:.4f} +/- {s['mae_std']:.4f}"
        rmse_str = f"{s['rmse_mean']:.4f} +/- {s['rmse_std']:.4f}"
        r_str = f"{s['r_mean']:.4f} +/- {s['r_std']:.4f}"
        jerk_str = f"{s['jerk_rms_mean']:.2f} +/- {s['jerk_rms_std']:.2f}"
        jerk_red_str = f"{s['jerk_reduction_percent']:+.1f}%"
        print(f"{s['label']:<22} | {mae_str:<16} | {rmse_str:<16} | {r_str:<16} | {jerk_str:<16} | {jerk_red_str}")
    print("=" * 105)

    return summary


if __name__ == "__main__":
    run_multi_seed_suite()
