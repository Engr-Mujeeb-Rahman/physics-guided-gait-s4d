"""Comprehensive statistical significance and effect size analysis for gait kinetics models.

Upgraded to address low-power N=6 non-parametric floor:
1. Paired Cohen's d_z effect sizes
2. 95% Bootstrap Confidence Intervals (10,000 resamples)
3. Exact sign-permutation test (2^N = 64 exhaustive permutations for N=6)
4. Rank-biserial correlation effect size
5. Per-trial paired analysis on held-out test cohort (N=40 trials)
"""

import os
import sys
import json
import itertools
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.data.dataset import get_dataloaders
from src.evaluate import build_eval_model, safe_pearson_r


def paired_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d_z for paired samples: mean(diff) / std(diff)."""
    diff = x - y
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-12:
        return 0.0
    return float(np.mean(diff) / std_diff)


def bootstrap_ci_paired(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 10000,
    ci: float = 95.0,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for paired difference mean(x - y).
    
    Returns:
        (mean_diff, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    diff = x - y
    n = len(diff)
    mean_diff = float(np.mean(diff))
    
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(diff[indices], axis=1)
    
    alpha = (100.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha))
    ci_upper = float(np.percentile(boot_means, 100.0 - alpha))
    return mean_diff, ci_lower, ci_upper


def exact_paired_permutation_test(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided") -> float:
    """Exact paired sign permutation test for small samples (N <= 15).
    
    Exhaustively tests all 2^N sign assignments for d_i = x_i - y_i.
    """
    diff = x - y
    n = len(diff)
    observed_t = np.sum(diff)
    
    all_sums = []
    # Generate all 2^n sign vectors (+1 or -1)
    for signs in itertools.product([-1.0, 1.0], repeat=n):
        s = np.array(signs)
        all_sums.append(np.sum(s * np.abs(diff)))
        
    all_sums = np.array(all_sums)
    total_perm = len(all_sums)
    
    if alternative == "less":
        p_val = np.sum(all_sums <= observed_t) / total_perm
    elif alternative == "greater":
        p_val = np.sum(all_sums >= observed_t) / total_perm
    else:  # two-sided
        p_val = np.sum(np.abs(all_sums) >= np.abs(observed_t)) / total_perm
        
    return float(p_val)


def rank_biserial_correlation(w_stat: float, n: int) -> float:
    """Compute rank-biserial correlation from Wilcoxon signed-rank test statistic W."""
    total_ranks = n * (n + 1) / 2.0
    return float(1.0 - (2.0 * w_stat / total_ranks))


def evaluate_trial_level(checkpoint_path: str, config: str, device_name: str = "cpu"):
    """Evaluate model on primary test split and return per-trial MAE and Jerk RMS arrays."""
    device = torch.device(device_name)
    _, _, test_loader, _ = get_dataloaders(batch_size=1, split_mode="subject", include_muscles=True)
    
    model, use_film = build_eval_model(config, device=device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    trial_maes = []
    trial_jerks = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            speed = batch["speed"].to(device)
            dt = batch["dt"].to(device)
            
            if not use_film:
                pred = model(x)
            else:
                pred = model(x, speed=speed)
                
            p_np = pred.cpu().numpy()[0]  # (100, 3)
            y_np = y.cpu().numpy()[0]     # (100, 3)
            dt_val = float(dt.cpu().numpy()[0, 0])
            
            # Per-trial overall MAE
            mae = np.mean(np.abs(p_np - y_np))
            trial_maes.append(float(mae))
            
            # Per-trial jerk RMS
            d2tau = (p_np[2:, :] - 2.0 * p_np[1:-1, :] + p_np[:-2, :]) / (dt_val ** 2)
            jerk = np.sqrt(np.mean(d2tau ** 2))
            trial_jerks.append(float(jerk))
            
    return np.array(trial_maes), np.array(trial_jerks)


def run_comprehensive_analysis():
    print("==================================================")
    print("RUNNING COMPREHENSIVE STATISTICAL ANALYSIS")
    print("==================================================")
    
    # 1. Fold-level analysis from LOSO summary
    summary_path = "experiments/results/loso_summary.json"
    if not os.path.exists(summary_path):
        print(f"Warning: {summary_path} not found.")
        return
        
    with open(summary_path, "r") as f:
        loso_data = json.load(f)
        
    speeds = ["0.6", "0.8", "1.0", "1.2", "1.4", "1.6"]
    n_folds = len(speeds)
    
    mae_m1 = np.array([loso_data[s]["baseline"]["test_overall_mae"] for s in speeds])
    mae_m2 = np.array([loso_data[s]["conditioning"]["test_overall_mae"] for s in speeds])
    mae_m3 = np.array([loso_data[s]["conditioning_physics"]["test_overall_mae"] for s in speeds])
    
    jerk_m1 = np.array([loso_data[s]["baseline"]["pred_jerk_rms"] for s in speeds])
    jerk_m2 = np.array([loso_data[s]["conditioning"]["pred_jerk_rms"] for s in speeds])
    jerk_m3 = np.array([loso_data[s]["conditioning_physics"]["pred_jerk_rms"] for s in speeds])
    
    # Check if physics_only, lstm, tcn exist in loso_data
    has_m15 = "physics_only" in loso_data[speeds[0]]
    has_lstm = "lstm" in loso_data[speeds[0]]
    has_tcn = "tcn" in loso_data[speeds[0]]
    
    # Effect sizes & Bootstrap CIs on LOSO folds
    print("\n--- A. FOLD-LEVEL EFFECT SIZES & 95% BOOTSTRAP CIs (N=6 Folds) ---")
    
    # Jerk Reduction: Model 3 vs Model 1
    d_jerk_31 = paired_cohens_d(jerk_m3, jerk_m1)
    mean_diff_j31, ci_low_j31, ci_high_j31 = bootstrap_ci_paired(jerk_m3, jerk_m1)
    w_jerk_31 = stats.wilcoxon(jerk_m3, jerk_m1, alternative="less")
    p_exact_j31 = exact_paired_permutation_test(jerk_m3, jerk_m1, alternative="less")
    rb_j31 = rank_biserial_correlation(w_jerk_31.statistic, n_folds)
    
    print(f"Jerk (M3 vs M1): Mean Diff = {mean_diff_j31:.2f} [95% CI: {ci_low_j31:.2f}, {ci_high_j31:.2f}]")
    print(f"  Cohen's d_z = {d_jerk_31:.3f} | Rank-biserial r = {rb_j31:.3f}")
    print(f"  Wilcoxon p = {w_jerk_31.pvalue:.5f} | Exact Permutation p = {p_exact_j31:.5f}")
    
    # Jerk Reduction: Model 3 vs Model 2
    d_jerk_32 = paired_cohens_d(jerk_m3, jerk_m2)
    mean_diff_j32, ci_low_j32, ci_high_j32 = bootstrap_ci_paired(jerk_m3, jerk_m2)
    w_jerk_32 = stats.wilcoxon(jerk_m3, jerk_m2, alternative="less")
    p_exact_j32 = exact_paired_permutation_test(jerk_m3, jerk_m2, alternative="less")
    rb_j32 = rank_biserial_correlation(w_jerk_32.statistic, n_folds)
    
    print(f"Jerk (M3 vs M2): Mean Diff = {mean_diff_j32:.2f} [95% CI: {ci_low_j32:.2f}, {ci_high_j32:.2f}]")
    print(f"  Cohen's d_z = {d_jerk_32:.3f} | Rank-biserial r = {rb_j32:.3f}")
    print(f"  Wilcoxon p = {w_jerk_32.pvalue:.5f} | Exact Permutation p = {p_exact_j32:.5f}")
    
    # MAE Comparison: Model 2 vs Model 1
    d_mae_21 = paired_cohens_d(mae_m2, mae_m1)
    mean_diff_m21, ci_low_m21, ci_high_m21 = bootstrap_ci_paired(mae_m2, mae_m1)
    w_mae_21 = stats.wilcoxon(mae_m2, mae_m1, alternative="less")
    p_exact_m21 = exact_paired_permutation_test(mae_m2, mae_m1, alternative="less")
    rb_m21 = rank_biserial_correlation(w_mae_21.statistic, n_folds)
    
    print(f"MAE (M2 vs M1): Mean Diff = {mean_diff_m21:.4f} [95% CI: {ci_low_m21:.4f}, {ci_high_m21:.4f}]")
    print(f"  Cohen's d_z = {d_mae_21:.3f} | Rank-biserial r = {rb_m21:.3f}")
    print(f"  Wilcoxon p = {w_mae_21.pvalue:.5f} | Exact Permutation p = {p_exact_m21:.5f}")
    
    # 2. Per-Trial Analysis (N=40 primary test trials)
    print("\n--- B. TRIAL-LEVEL STATISTICAL TESTS (N=40 Primary Test Trials) ---")
    ckpts = {
        "m1": ("experiments/phase2_baseline/model_best.pt", "baseline"),
        "m2": ("experiments/phase3_conditioning/model_best.pt", "conditioning"),
        "m3": ("experiments/phase3_conditioning_physics/model_best.pt", "conditioning_physics"),
    }
    if os.path.exists("experiments/phase3_physics_only/model_best.pt"):
        ckpts["m15"] = ("experiments/phase3_physics_only/model_best.pt", "physics_only")
    if os.path.exists("experiments/baseline_lstm/model_best.pt"):
        ckpts["lstm"] = ("experiments/baseline_lstm/model_best.pt", "lstm")
    if os.path.exists("experiments/baseline_tcn/model_best.pt"):
        ckpts["tcn"] = ("experiments/baseline_tcn/model_best.pt", "tcn")
        
    trial_data = {}
    for key, (ckpt, cfg) in ckpts.items():
        if os.path.exists(ckpt):
            print(f"Evaluating trial-level predictions for {key} [{cfg}]...")
            t_maes, t_jerks = evaluate_trial_level(ckpt, cfg)
            trial_data[key] = {"mae": t_maes, "jerk": t_jerks}
            
    trial_results = {}
    if "m1" in trial_data and "m3" in trial_data:
        m1_jerk = trial_data["m1"]["jerk"]
        m3_jerk = trial_data["m3"]["jerk"]
        w_trial_j31 = stats.wilcoxon(m3_jerk, m1_jerk, alternative="less")
        tt_trial_j31 = stats.ttest_rel(m3_jerk, m1_jerk, alternative="less")
        dz_trial_j31 = paired_cohens_d(m3_jerk, m1_jerk)
        diff_j, ci_l_j, ci_u_j = bootstrap_ci_paired(m3_jerk, m1_jerk)
        
        print(f"\n[Trial-Level N=40] Jerk M3 vs M1:")
        print(f"  M1 Jerk Mean: {np.mean(m1_jerk):.2f} +- {np.std(m1_jerk, ddof=1):.2f}")
        print(f"  M3 Jerk Mean: {np.mean(m3_jerk):.2f} +- {np.std(m3_jerk, ddof=1):.2f}")
        print(f"  Mean Diff: {diff_j:.2f} [95% CI: {ci_l_j:.2f}, {ci_u_j:.2f}]")
        print(f"  Cohen's d_z = {dz_trial_j31:.3f}")
        print(f"  Wilcoxon Signed-Rank: stat={w_trial_j31.statistic}, p-value={w_trial_j31.pvalue:.4e}")
        print(f"  Paired t-test:        t={tt_trial_j31.statistic:.3f}, p-value={tt_trial_j31.pvalue:.4e}")
        
        trial_results["jerk_m3_vs_m1"] = {
            "m1_mean": float(np.mean(m1_jerk)),
            "m1_std": float(np.std(m1_jerk, ddof=1)),
            "m3_mean": float(np.mean(m3_jerk)),
            "m3_std": float(np.std(m3_jerk, ddof=1)),
            "mean_diff": diff_j,
            "ci_95": [ci_l_j, ci_u_j],
            "cohens_d": dz_trial_j31,
            "wilcoxon_stat": float(w_trial_j31.statistic),
            "wilcoxon_pvalue": float(w_trial_j31.pvalue),
            "ttest_stat": float(tt_trial_j31.statistic),
            "ttest_pvalue": float(tt_trial_j31.pvalue),
        }
        
    if "m1" in trial_data and "m2" in trial_data:
        m1_mae = trial_data["m1"]["mae"]
        m2_mae = trial_data["m2"]["mae"]
        w_trial_m21 = stats.wilcoxon(m2_mae, m1_mae, alternative="less")
        tt_trial_m21 = stats.ttest_rel(m2_mae, m1_mae, alternative="less")
        dz_trial_m21 = paired_cohens_d(m2_mae, m1_mae)
        diff_m, ci_l_m, ci_u_m = bootstrap_ci_paired(m2_mae, m1_mae)
        
        print(f"\n[Trial-Level N=40] MAE M2 vs M1:")
        print(f"  M1 MAE Mean: {np.mean(m1_mae):.4f} +- {np.std(m1_mae, ddof=1):.4f}")
        print(f"  M2 MAE Mean: {np.mean(m2_mae):.4f} +- {np.std(m2_mae, ddof=1):.4f}")
        print(f"  Mean Diff: {diff_m:.4f} [95% CI: {ci_l_m:.4f}, {ci_u_m:.4f}]")
        print(f"  Cohen's d_z = {dz_trial_m21:.3f}")
        print(f"  Wilcoxon Signed-Rank: stat={w_trial_m21.statistic}, p-value={w_trial_m21.pvalue:.4e}")
        print(f"  Paired t-test:        t={tt_trial_m21.statistic:.3f}, p-value={tt_trial_m21.pvalue:.4e}")
        
        trial_results["mae_m2_vs_m1"] = {
            "m1_mean": float(np.mean(m1_mae)),
            "m1_std": float(np.std(m1_mae, ddof=1)),
            "m2_mean": float(np.mean(m2_mae)),
            "m2_std": float(np.std(m2_mae, ddof=1)),
            "mean_diff": diff_m,
            "ci_95": [ci_l_m, ci_u_m],
            "cohens_d": dz_trial_m21,
            "wilcoxon_stat": float(w_trial_m21.statistic),
            "wilcoxon_pvalue": float(w_trial_m21.pvalue),
            "ttest_stat": float(tt_trial_m21.statistic),
            "ttest_pvalue": float(tt_trial_m21.pvalue),
        }

    # Consolidated output
    final_stats = {
        "fold_level_n6": {
            "jerk_m3_vs_m1": {
                "mean_diff": mean_diff_j31,
                "ci_95": [ci_low_j31, ci_high_j31],
                "cohens_dz": d_jerk_31,
                "rank_biserial_r": rb_j31,
                "wilcoxon_p": float(w_jerk_31.pvalue),
                "exact_permutation_p": p_exact_j31,
            },
            "jerk_m3_vs_m2": {
                "mean_diff": mean_diff_j32,
                "ci_95": [ci_low_j32, ci_high_j32],
                "cohens_dz": d_jerk_32,
                "rank_biserial_r": rb_j32,
                "wilcoxon_p": float(w_jerk_32.pvalue),
                "exact_permutation_p": p_exact_j32,
            },
            "mae_m2_vs_m1": {
                "mean_diff": mean_diff_m21,
                "ci_95": [ci_low_m21, ci_high_m21],
                "cohens_dz": d_mae_21,
                "rank_biserial_r": rb_m21,
                "wilcoxon_p": float(w_mae_21.pvalue),
                "exact_permutation_p": p_exact_m21,
            },
        },
        "trial_level_n40": trial_results,
    }
    
    out_file = "experiments/results/statistical_summary_comprehensive.json"
    with open(out_file, "w") as f:
        json.dump(final_stats, f, indent=2)
    print(f"\nSaved comprehensive statistical summary to {out_file}")
    return final_stats


if __name__ == "__main__":
    run_comprehensive_analysis()
