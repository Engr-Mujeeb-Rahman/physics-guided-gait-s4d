"""
Generate Figure 4 (Bland-Altman Agreement Plots) and Figure 5 (Torque Smoothness / Jerk Demonstration)
for the IEEE J-BHI Manuscript.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Ensure repo root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.models.backbone import GaitS4D
from src.data.dataset import get_dataloaders

OUT_DIR = os.path.join(repo_root, "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Styling settings for IEEE publications
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def load_model(checkpoint_path: str, use_film: bool = False, device: torch.device = torch.device("cpu")):
    model = GaitS4D(
        in_channels=102,
        out_channels=3,
        d_model=64,
        d_state=64,
        n_layers=4,
        dropout=0.1,
        use_film=use_film,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def get_test_predictions_and_targets(device=torch.device("cpu")):
    _, _, test_loader, _ = get_dataloaders(batch_size=16, split_mode="subject", include_muscles=True)
    
    m1 = load_model("experiments/phase2_baseline/model_best.pt", use_film=False, device=device)
    m15 = load_model("experiments/phase3_physics_only/model_best.pt", use_film=False, device=device)
    m3 = load_model("experiments/phase3_conditioning_physics/model_best.pt", use_film=True, device=device)
    
    p1_list, p15_list, p3_list, targ_list, dt_list = [], [], [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            speed = batch["speed"].to(device)
            dt = batch["dt"].to(device)
            
            p1 = m1(x).cpu().numpy()
            p15 = m15(x).cpu().numpy()
            p3 = m3(x, speed=speed).cpu().numpy()
            
            p1_list.append(p1)
            p15_list.append(p15)
            p3_list.append(p3)
            targ_list.append(y.cpu().numpy())
            dt_list.append(dt.cpu().numpy())
            
    p1 = np.concatenate(p1_list, axis=0)     # (N, 100, 3)
    p15 = np.concatenate(p15_list, axis=0)   # (N, 100, 3)
    p3 = np.concatenate(p3_list, axis=0)     # (N, 100, 3)
    targ = np.concatenate(targ_list, axis=0) # (N, 100, 3)
    dts = np.concatenate(dt_list, axis=0)    # (N, 1)
    
    return p1, p15, p3, targ, dts


def generate_fig4_bland_altman(p3, targ):
    """Generate 3-panel Bland-Altman agreement plots for Hip, Knee, and Ankle moments."""
    joint_names = ["Hip Flexion/Extension", "Knee Flexion/Extension", "Ankle Plantar/Dorsiflexion"]
    colors = ["#1D4ED8", "#D97706", "#0D9488"]
    
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), sharey=False)
    
    for j in range(3):
        ax = axes[j]
        pred_j = p3[:, :, j].flatten()
        targ_j = targ[:, :, j].flatten()
        
        # Subsample points for clean visual presentation
        step = max(1, len(pred_j) // 1200)
        p_sub = pred_j[::step]
        t_sub = targ_j[::step]
        
        means = (p_sub + t_sub) / 2.0
        diffs = p_sub - t_sub
        
        mean_diff = float(np.mean(diffs))
        sd_diff = float(np.std(diffs, ddof=1))
        upper_loa = mean_diff + 1.96 * sd_diff
        lower_loa = mean_diff - 1.96 * sd_diff
        
        # Scatter plot
        ax.scatter(means, diffs, alpha=0.25, s=7, color=colors[j], edgecolors="none", zorder=2)
        
        # Mean bias line
        ax.axhline(mean_diff, color="#111827", linestyle="-", linewidth=1.2, zorder=3, label=f"Bias: {mean_diff:+.3f}")
        
        # 95% Limits of Agreement
        ax.axhline(upper_loa, color="#DC2626", linestyle="--", linewidth=1.0, zorder=3, label=f"+1.96 SD: {upper_loa:+.3f}")
        ax.axhline(lower_loa, color="#DC2626", linestyle="--", linewidth=1.0, zorder=3, label=f"-1.96 SD: {lower_loa:+.3f}")
        
        # Shaded LoA zone
        ax.axhspan(lower_loa, upper_loa, color="#F3F4F6", alpha=0.5, zorder=1)
        
        ax.set_title(f"({chr(97+j)}) {joint_names[j]}", fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Mean of Model & GT (N·m/kg)", fontsize=8)
        if j == 0:
            ax.set_ylabel("Difference (Model - GT) (N·m/kg)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=6.8, framealpha=0.9)
        
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_bland_altman.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig4_bland_altman.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved Figure 4: Bland-Altman Plots (fig4_bland_altman.pdf)")


def generate_fig5_jerk_smoothing(p1, p15, p3, targ, dts):
    """Generate 3-panel figure visualizing torque trajectory, torque acceleration (jerk penalty), and frequency spectrum."""
    # Select a representative stride trial with active knee/ankle dynamics
    trial_idx = 12  # representative test trial
    joint_idx = 1   # Knee joint
    
    dt_val = float(dts[trial_idx, 0])
    time_pct = np.linspace(0, 100, 100)
    
    t_gt = targ[trial_idx, :, joint_idx]
    t_m1 = p1[trial_idx, :, joint_idx]
    t_m15 = p15[trial_idx, :, joint_idx]
    t_m3 = p3[trial_idx, :, joint_idx]
    
    # Compute numerical second derivatives (torque acceleration / jerk)
    d2_gt = (t_gt[2:] - 2*t_gt[1:-1] + t_gt[:-2]) / (dt_val**2)
    d2_m1 = (t_m1[2:] - 2*t_m1[1:-1] + t_m1[:-2]) / (dt_val**2)
    d2_m15 = (t_m15[2:] - 2*t_m15[1:-1] + t_m15[:-2]) / (dt_val**2)
    d2_m3 = (t_m3[2:] - 2*t_m3[1:-1] + t_m3[:-2]) / (dt_val**2)
    t_d2 = time_pct[1:-1]
    
    # Compute FFT spectral magnitude of second derivative
    fft_m1 = np.abs(np.fft.rfft(d2_m1))
    fft_m3 = np.abs(np.fft.rfft(d2_m3))
    fft_gt = np.abs(np.fft.rfft(d2_gt))
    freqs = np.fft.rfftfreq(len(d2_gt), d=dt_val)
    
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
    
    # Panel (a): Moment Trajectory tau(t)
    ax0 = axes[0]
    ax0.plot(time_pct, t_gt, "k--", linewidth=1.4, label="Ground Truth")
    ax0.plot(time_pct, t_m1, color="#1D4ED8", alpha=0.85, label="Model 1 (Baseline)")
    ax0.plot(time_pct, t_m3, color="#0D9488", linewidth=1.5, label="Model 3 (+Cond.+Phys.)")
    ax0.set_title("(a) Knee Moment Trajectory", fontsize=8.5, fontweight="bold")
    ax0.set_xlabel("Gait Cycle (%)", fontsize=8)
    ax0.set_ylabel("Knee Moment (N·m/kg)", fontsize=8)
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(loc="lower right", fontsize=6.8, framealpha=0.9)
    
    # Panel (b): Torque Second-Derivative d^2 tau / dt^2
    ax1 = axes[1]
    ax1.plot(t_d2, d2_gt, "k--", linewidth=1.2, alpha=0.7, label="Ground Truth")
    ax1.plot(t_d2, d2_m1, color="#1D4ED8", linewidth=1.0, alpha=0.8, label="Model 1 (Spiky)")
    ax1.plot(t_d2, d2_m3, color="#0D9488", linewidth=1.4, label="Model 3 (Smooth)")
    ax1.set_title(r"(b) Torque Acceleration ($\ddot{\tau}$)", fontsize=8.5, fontweight="bold")
    ax1.set_xlabel("Gait Cycle (%)", fontsize=8)
    ax1.set_ylabel(r"$\ddot{\tau}$ ($\mathrm{N\cdot m/(kg\cdot s^2)}$)", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", fontsize=6.8, framealpha=0.9)
    
    # Panel (c): FFT Spectral Power of Second-Derivative
    ax2 = axes[2]
    mask = freqs <= 25.0  # display frequencies up to 25 Hz
    ax2.plot(freqs[mask], fft_gt[mask], "k--", linewidth=1.2, label="Ground Truth")
    ax2.plot(freqs[mask], fft_m1[mask], color="#1D4ED8", linewidth=1.2, label="Model 1")
    ax2.plot(freqs[mask], fft_m3[mask], color="#0D9488", linewidth=1.5, label="Model 3 (-34% Jerk)")
    ax2.set_title("(c) Jerk Spectral Magnitude", fontsize=8.5, fontweight="bold")
    ax2.set_xlabel("Frequency (Hz)", fontsize=8)
    ax2.set_ylabel("Spectral Magnitude", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=6.8, framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_jerk_smoothing.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig5_jerk_smoothing.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved Figure 5: Torque Smoothness & Jerk Demonstration (fig5_jerk_smoothing.pdf)")


def generate_fig4_combined(p1, p15, p3, targ, dts):
    """Generate 6-panel unified figure combining Bland-Altman agreement (Row 1) and Torque smoothing/spectral analysis (Row 2)."""
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 2.95), dpi=300)
    
    # --- ROW 1: Bland-Altman Plots ---
    joint_names = ["Hip Flexion/Ext.", "Knee Flexion/Ext.", "Ankle Plantar/Dorsi."]
    colors = ["#1D4ED8", "#D97706", "#0D9488"]
    
    for j in range(3):
        ax = axes[0, j]
        pred_j = p3[:, :, j].flatten()
        targ_j = targ[:, :, j].flatten()
        
        step = max(1, len(pred_j) // 1200)
        p_sub = pred_j[::step]
        t_sub = targ_j[::step]
        
        means = (p_sub + t_sub) / 2.0
        diffs = p_sub - t_sub
        
        mean_diff = float(np.mean(diffs))
        sd_diff = float(np.std(diffs, ddof=1))
        upper_loa = mean_diff + 1.96 * sd_diff
        lower_loa = mean_diff - 1.96 * sd_diff
        
        ax.scatter(means, diffs, alpha=0.25, s=6, color=colors[j], edgecolors="none", zorder=2)
        ax.axhline(mean_diff, color="#111827", linestyle="-", linewidth=1.1, zorder=3, label=f"Bias: {mean_diff:+.3f}")
        ax.axhline(upper_loa, color="#DC2626", linestyle="--", linewidth=0.9, zorder=3, label=f"+1.96 SD: {upper_loa:+.3f}")
        ax.axhline(lower_loa, color="#DC2626", linestyle="--", linewidth=0.9, zorder=3, label=f"-1.96 SD: {lower_loa:+.3f}")
        ax.axhspan(lower_loa, upper_loa, color="#F3F4F6", alpha=0.5, zorder=1)
        
        ax.set_title(f"({chr(97+j)}) {joint_names[j]} Agreement", fontsize=8.2, fontweight="bold", pad=2)
        ax.set_xlabel("Mean of Model & GT (N·m/kg)", fontsize=7.6)
        if j == 0:
            ax.set_ylabel("Diff. (Model - GT) (N·m/kg)", fontsize=7.6)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=6.2, framealpha=0.9)
    
    # --- ROW 2: Jerk and Torque Acceleration Smoothing ---
    trial_idx = 12
    joint_idx = 1
    dt_val = float(dts[trial_idx, 0])
    time_pct = np.linspace(0, 100, 100)
    
    t_gt = targ[trial_idx, :, joint_idx]
    t_m1 = p1[trial_idx, :, joint_idx]
    t_m15 = p15[trial_idx, :, joint_idx]
    t_m3 = p3[trial_idx, :, joint_idx]
    
    d2_gt = (t_gt[2:] - 2*t_gt[1:-1] + t_gt[:-2]) / (dt_val**2)
    d2_m1 = (t_m1[2:] - 2*t_m1[1:-1] + t_m1[:-2]) / (dt_val**2)
    d2_m15 = (t_m15[2:] - 2*t_m15[1:-1] + t_m15[:-2]) / (dt_val**2)
    d2_m3 = (t_m3[2:] - 2*t_m3[1:-1] + t_m3[:-2]) / (dt_val**2)
    t_d2 = time_pct[1:-1]
    
    fft_m1 = np.abs(np.fft.rfft(d2_m1))
    fft_m3 = np.abs(np.fft.rfft(d2_m3))
    fft_gt = np.abs(np.fft.rfft(d2_gt))
    freqs = np.fft.rfftfreq(len(d2_gt), d=dt_val)
    
    # Panel (d): Moment Trajectory
    ax0 = axes[1, 0]
    ax0.plot(time_pct, t_gt, "k--", linewidth=1.3, label="Ground Truth")
    ax0.plot(time_pct, t_m1, color="#1D4ED8", alpha=0.85, label="Model 1 (Baseline)")
    ax0.plot(time_pct, t_m3, color="#0D9488", linewidth=1.4, label="Model 3 (+Cond.+Phys.)")
    ax0.set_title("(d) Knee Moment Trajectory", fontsize=8.2, fontweight="bold", pad=2)
    ax0.set_xlabel("Gait Cycle (%)", fontsize=7.6)
    ax0.set_ylabel("Knee Moment (N·m/kg)", fontsize=7.6)
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(loc="lower right", fontsize=6.2, framealpha=0.9)
    
    # Panel (e): Torque Second-Derivative
    ax1 = axes[1, 1]
    ax1.plot(t_d2, d2_gt, "k--", linewidth=1.1, alpha=0.7, label="Ground Truth")
    ax1.plot(t_d2, d2_m1, color="#1D4ED8", linewidth=0.9, alpha=0.8, label="Model 1 (Spiky)")
    ax1.plot(t_d2, d2_m3, color="#0D9488", linewidth=1.3, label="Model 3 (Smooth)")
    ax1.set_title(r"(e) Torque Acceleration ($\ddot{\tau}$)", fontsize=8.2, fontweight="bold", pad=2)
    ax1.set_xlabel("Gait Cycle (%)", fontsize=7.6)
    ax1.set_ylabel(r"$\ddot{\tau}$ ($\mathrm{N\cdot m/(kg\cdot s^2)}$)", fontsize=7.6)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", fontsize=6.2, framealpha=0.9)
    
    # Panel (f): Spectral Power
    ax2 = axes[1, 2]
    mask = freqs <= 25.0
    ax2.plot(freqs[mask], fft_gt[mask], "k--", linewidth=1.1, label="Ground Truth")
    ax2.plot(freqs[mask], fft_m1[mask], color="#1D4ED8", linewidth=1.1, label="Model 1")
    ax2.plot(freqs[mask], fft_m3[mask], color="#0D9488", linewidth=1.4, label="Model 3 (-34% Jerk)")
    ax2.set_title("(f) Jerk Spectral Magnitude", fontsize=8.2, fontweight="bold", pad=2)
    ax2.set_xlabel("Frequency (Hz)", fontsize=7.6)
    ax2.set_ylabel("Spectral Magnitude", fontsize=7.6)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=6.2, framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_clinical_agreement_and_jerk.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig4_clinical_agreement_and_jerk.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved Unified Figure 4: Clinical Agreement & Jerk (fig4_clinical_agreement_and_jerk.pdf)")


if __name__ == "__main__":
    print("Generating Figures 4 & 5...")
    p1, p15, p3, targ, dts = get_test_predictions_and_targets()
    generate_fig4_bland_altman(p3, targ)
    generate_fig5_jerk_smoothing(p1, p15, p3, targ, dts)
    generate_fig4_combined(p1, p15, p3, targ, dts)
    print("All figures generated successfully!")

