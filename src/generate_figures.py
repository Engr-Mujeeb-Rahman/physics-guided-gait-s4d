"""
Generate Publication-Quality Figures for IEEE J-BHI Manuscript (Revision 2).

Generates:
1. paper/figures/fig1_architecture.pdf & .png: Redesigned S4D Architecture diagram (zero label collisions, clean spacing, tensor badges)
2. paper/figures/fig2_waveforms.pdf & .png: Joint moment trajectories with Ground Truth cohort variability envelopes (+-1 SD)
3. paper/figures/fig3_generalization_and_jerk.pdf & .png: Unified two-panel figure combining Jerk RMS bar chart (Panel a) and MAE speed trend (Panel b)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
import torch

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.models.backbone import GaitS4D
from src.data.dataset import get_dataloaders

# Styling settings for IEEE publications
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10.5,
    "lines.linewidth": 1.4,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# Color palette (IEEE-compliant, colorblind-friendly)
C_GT = "#111827"        # Dark charcoal for Ground Truth
C_M1 = "#1D4ED8"        # Royal Blue for Baseline (Model 1)
C_M2 = "#EA580C"        # Amber/Coral for +Conditioning (Model 2)
C_M3 = "#0D9488"        # Deep Teal for +Cond+Physics (Model 3)
C_BAND = "#D1D5DB"      # Soft Gray for Cohort Variability Envelope

OUT_DIR = os.path.join(repo_root, "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================================================================
# FIGURE 1: ARCHITECTURE DIAGRAM (Publication-Grade Modern Card Architecture)
# ==============================================================================
def generate_fig1_architecture():
    fig = plt.figure(figsize=(7.16, 3.45), dpi=300)
    # Use exact coordinates mapping 0..100 directly without tight_layout distortion
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Cohesive Modern IEEE Color Tokens
    C_CONTAINER_BG = "#F8FAFC"
    C_CONTAINER_BORDER = "#475569"
    
    # Input styling (Clean Blue)
    C_INPUT_BG = "#F0F7FF"
    C_INPUT_HDR = "#DBEAFE"
    C_INPUT_BORDER = "#2563EB"
    C_INPUT_TXT = "#1E3A8A"
    
    # Speed styling (Amber/Gold)
    C_SPEED_BG = "#FFFBEB"
    C_SPEED_HDR = "#FEF3C7"
    C_SPEED_BORDER = "#D97706"
    C_SPEED_TXT = "#92400E"

    # SSM styling (Emerald Green)
    C_SSM_BG = "#F0FDF4"
    C_SSM_HDR = "#DCFCE7"
    C_SSM_BORDER = "#16A34A"
    C_SSM_TXT = "#14532D"

    # FiLM styling (Vibrant Orange)
    C_FILM_BG = "#FFF7ED"
    C_FILM_HDR = "#FFEDD5"
    C_FILM_BORDER = "#EA580C"
    C_FILM_TXT = "#9A3412"

    # Neutral/Projection styling (Slate)
    C_NEUT_BG = "#F8FAFC"
    C_NEUT_HDR = "#E2E8F0"
    C_NEUT_BORDER = "#64748B"
    C_NEUT_TXT = "#1E293B"

    # Output styling (Indigo/Purple)
    C_OUT_BG = "#F5F3FF"
    C_OUT_HDR = "#EDE9FE"
    C_OUT_BORDER = "#7C3AED"
    C_OUT_TXT = "#4C1D95"

    # Loss styling (Ruby Red)
    C_LOSS_BG = "#FEF2F2"
    C_LOSS_HDR = "#FEE2E2"
    C_LOSS_BORDER = "#DC2626"
    C_LOSS_TXT = "#991B1B"

    # Helper function: Draw a sleek card with distinct header banner
    def draw_card(x, y, w, h, title, body_lines, hdr_bg, hdr_txt, card_bg, border_col,
                  lw=1.1, r=1.4, title_fs=7.2, body_fs=5.7, fixed_spacing=None):
        # Background card with extra padding for margins
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={r}",
                                     facecolor=card_bg, edgecolor=border_col, linewidth=lw, zorder=2)
        ax.add_patch(box)

        # Header banner height
        hdr_h = 6.0
        hdr_box = patches.FancyBboxPatch((x, y + h - hdr_h), w, hdr_h,
                                         boxstyle=f"round,pad=0.0,rounding_size={r}",
                                         facecolor=hdr_bg, edgecolor=border_col, linewidth=lw, zorder=3)
        ax.add_patch(hdr_box)

        # Cover bottom rounded corners of header box for seamless look
        cover = patches.Rectangle((x, y + h - hdr_h), w, r, facecolor=hdr_bg, edgecolor="none", zorder=4)
        ax.add_patch(cover)
        # Header separator line
        ax.plot([x, x + w], [y + h - hdr_h, y + h - hdr_h], color=border_col, lw=lw, zorder=5)

        # Title text (centered with clean margin - increased left/right padding)
        ax.text(x + w / 2.0, y + h - (hdr_h / 2.0), title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=hdr_txt, zorder=6)

        # Body lines with generous left margin and line spacing
        if body_lines:
            n_lines = len(body_lines)
            if fixed_spacing is not None:
                line_step = fixed_spacing
            else:
                usable_h = h - hdr_h - 2.5  # Increased bottom margin
                line_step = usable_h / max(n_lines, 1)
            
            start_y = y + h - hdr_h - 1.5
            for i, line in enumerate(body_lines):
                cur_y = start_y - (i * line_step)
                is_bold = line.startswith("•") or line.startswith("  $\\to")
                color = "#0F172A" if is_bold else "#334155"
                ax.text(x + 1.3, cur_y, line, ha="left", va="top",  # Increased left margin from 1.1 to 1.3
                        fontsize=body_fs, fontweight="bold" if is_bold else "normal",
                        color=color, zorder=6)

        return box

    # ==========================================
    # 1. STAGE 1: MULTIMODAL & CONTEXT INPUTS
    # ==========================================
    # Multimodal Features Card
    draw_card(1.0, 39.0, 18.0, 58.0,
              "Multimodal Inputs",
              ["• Joint Kinematics (18 ch):",
               "   Hip, Knee, Ankle 3D (deg)",
               "   Angular Vel. $\\dot{\\mathbf{q}}$ (rad/s)",
               "• Kinetic Loads (6 ch):",
               "   3D Bilateral GRF (N/kg)",
               "• Musculotendon State (80 ch):",
               "   Fiber Length $l_m$ (40 ch)",
               "   Fiber Velocity $v_m$ (40 ch)",
               "• Unified Sensor Dimension:",
               "   Input Vector $\\mathbf{x}_t \\in \\mathbb{R}^{102}$"],
              C_INPUT_HDR, C_INPUT_TXT, C_INPUT_BG, C_INPUT_BORDER, body_fs=5.7)

    # Walking Speed Scalar Card
    draw_card(1.0, 5.0, 18.0, 29.0,
              "Speed Context",
              ["• Treadmill Velocity $s$:",
               "   Continuous: 0.6–1.6 m/s",
               "• Normalized Scalar:",
               "   Replicated across $T=100$",
               "• Conditioning Driver:",
               "   Gait Dynamics Modulation"],
              C_SPEED_HDR, C_SPEED_TXT, C_SPEED_BG, C_SPEED_BORDER, body_fs=5.7)

    # Connector from Input to Projection
    ax.annotate("", xy=(23.0, 67.0), xytext=(19.0, 67.0),
                arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=1.3, mutation_scale=9), zorder=8)
    # Dimension badge on arrow - increased size for better text fit
    b1 = patches.FancyBboxPatch((19.3, 68.2), 3.4, 3.6, boxstyle="round,pad=0.0,rounding_size=0.6",
                                facecolor="#DBEAFE", edgecolor="#2563EB", lw=0.6, zorder=9)
    ax.add_patch(b1)
    ax.text(21.0, 70.0, "102", ha="center", va="center", fontsize=5.8, fontweight="bold", color="#1E3A8A", zorder=10)

    # ==========================================
    # 2. STAGE 2: LINEAR PROJECTION
    # ==========================================
    draw_card(23.0, 48.0, 11.5, 38.0,
              "Projection",
              ["• Linear Dense:",
               "   $\\mathbb{R}^{102} \\to \\mathbb{R}^{64}$",
               "• Regularization:",
               "   Dropout ($p=0.1$)",
               "• Normalization:",
               "   LayerNorm $\\mathbb{R}^{64}$",
               "• Embed Token:",
               "   $\\mathbf{u}_t \\in \\mathbb{R}^{64}$"],
              C_NEUT_HDR, C_NEUT_TXT, C_NEUT_BG, C_NEUT_BORDER, body_fs=5.7)

    # Connector from Projection straight INTO Diagonal S4D
    ax.annotate("", xy=(40.5, 67.0), xytext=(34.5, 67.0),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.3, mutation_scale=9), zorder=8)
    # Dimension badge on arrow - increased size
    b2 = patches.FancyBboxPatch((35.1, 68.2), 3.3, 3.6, boxstyle="round,pad=0.0,rounding_size=0.6",
                                facecolor="#E2E8F0", edgecolor="#475569", lw=0.6, zorder=9)
    ax.add_patch(b2)
    ax.text(36.75, 70.0, "64", ha="center", va="center", fontsize=5.8, fontweight="bold", color="#1E293B", zorder=10)

    # ==========================================
    # 3. STAGE 3: S4D BACKBONE CONTAINER
    # ==========================================
    # Outer dashed container box
    s4d_container = patches.FancyBboxPatch((38.5, 5.0), 37.5, 92.5, boxstyle="round,pad=0.0,rounding_size=2.2",
                                          facecolor=C_CONTAINER_BG, edgecolor=C_CONTAINER_BORDER, linewidth=1.2,
                                          linestyle="--", zorder=1)
    ax.add_patch(s4d_container)

    # Container Header Pill
    pill_w = 36.5
    pill_h = 5.6
    pill = patches.FancyBboxPatch((57.25 - pill_w/2.0, 95.0 - pill_h/2.0), pill_w, pill_h,
                                 boxstyle="round,pad=0.0,rounding_size=1.2",
                                 facecolor="#1E293B", edgecolor="#0F172A", lw=0.8, zorder=3)
    ax.add_patch(pill)
    ax.text(57.25, 95.0, "Stacked S4D Backbone (×4 Blocks, $d_{\\mathrm{model}}=64$)", ha="center", va="center",
            fontsize=6.8, fontweight="bold", color="#FFFFFF", zorder=4)

    # Inside Container Sub-elements:
    # A) Diagonal S4D Layer Card
    draw_card(40.5, 49.0, 11.5, 35.0,
              "Diagonal S4D",
              ["• Continuous SSM:",
               "   $\\dot{\\mathbf{h}} = \\mathbf{A}\\mathbf{h} + \\mathbf{B}\\mathbf{x}$",
               "• Discretization:",
               "   Zero-Order Hold",
               "• HiPPO State:",
               "   $d_{\\mathrm{state}}=64$",
               "• Diag Kernel $\\mathbf{K}$:",
               "   $\\mathcal{O}(N)$ FFT Conv"],
              C_SSM_HDR, C_SSM_TXT, C_SSM_BG, C_SSM_BORDER, body_fs=5.6)

    # Connector from S4D to FiLM Node
    ax.annotate("", xy=(53.5, 67.0), xytext=(52.0, 67.0),
                arrowprops=dict(arrowstyle="-|>", color="#16A34A", lw=1.3, mutation_scale=9), zorder=8)
    ax.text(52.75, 69.2, "$\\mathbf{z}_l$", fontsize=6.6, fontweight="bold", color="#14532D", ha="center")

    # B) FiLM Affine Modulation Node (Circle)
    circ_r = 2.5
    cx, cy = 56.0, 67.0
    circle_bg = patches.Circle((cx, cy), circ_r, facecolor="#FFEDD5", edgecolor="#EA580C", lw=1.3, zorder=6)
    ax.add_patch(circle_bg)
    ax.text(cx, cy, "$\\odot, +$", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#9A3412", zorder=7)
    ax.text(cx, cy + 3.4, "FiLM", ha="center", va="bottom", fontsize=6.2, fontweight="bold", color="#9A3412", zorder=7)

    # Connector from FiLM Node to Feed-Forward
    ax.annotate("", xy=(60.0, 67.0), xytext=(58.5, 67.0),
                arrowprops=dict(arrowstyle="-|>", color="#EA580C", lw=1.3, mutation_scale=9), zorder=8)
    ax.text(59.25, 69.2, "$\\tilde{\\mathbf{z}}_l$", fontsize=6.6, fontweight="bold", color="#9A3412", ha="center")

    # C) Feed-Forward Card
    draw_card(60.0, 49.0, 14.5, 35.0,
              "Feed-Forward",
              ["• Non-linearity:",
               "   GELU Activation",
               "• Regularization:",
               "   Dropout ($p=0.1$)",
               "• Projection:",
               "   Dense $\\mathbb{R}^{64} \\to \\mathbb{R}^{64}$",
               "• Residual Add:",
               "   $\\mathbf{x} + \\tilde{\\mathbf{z}}_l$ & Norm"],
              C_NEUT_HDR, C_NEUT_TXT, C_NEUT_BG, C_NEUT_BORDER, body_fs=5.6)

    # D) Skip Residual Loop (Over S4D Layer)
    # Branch dot at tap point
    tap_dot = patches.Circle((39.5, 67.0), 0.45, facecolor="#475569", edgecolor="none", zorder=9)
    ax.add_patch(tap_dot)

    skip_path_data = [
        (Path.MOVETO, (39.5, 67.0)),
        (Path.LINETO, (39.5, 87.0)),
        (Path.LINETO, (67.25, 87.0)),
        (Path.LINETO, (67.25, 84.0))
    ]
    codes, verts = zip(*skip_path_data)
    skip_path = Path(verts, codes)
    skip_patch = patches.PathPatch(skip_path, facecolor="none", edgecolor="#475569", lw=1.1, linestyle=":", zorder=5)
    ax.add_patch(skip_patch)
    # Arrow tip entering Feed-Forward
    ax.annotate("", xy=(67.25, 84.0), xytext=(67.25, 85.8),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.1, mutation_scale=8), zorder=8)
    # Label badge on skip path - increased size
    s_badge = patches.FancyBboxPatch((48.0, 85.5), 10.6, 3.0, boxstyle="round,pad=0.0,rounding_size=0.6",
                                     facecolor="#E2E8F0", edgecolor="#64748B", lw=0.5, zorder=6)
    ax.add_patch(s_badge)
    ax.text(53.3, 87.0, "+ Skip Residual", ha="center", va="center", fontsize=5.5, fontweight="bold", color="#334155", zorder=7)

    # E) FiLM Generator MLP (Bottom half of S4D Container)
    draw_card(40.5, 8.5, 34.0, 35.0,
              "FiLM Conditioning Generator (Block $l$)",
              ["• 2-Layer Speed Adaptation MLP (SiLU):",
               "   $s \\in \\mathbb{R} \\to \\mathrm{Linear}(1 \\to 32) \\to \\mathrm{SiLU}$",
               "   $\\to \\mathrm{Linear}(32 \\to 128) \\to (\\boldsymbol{\\gamma}_l, \\boldsymbol{\\beta}_l) \\in \\mathbb{R}^{2d_{\\mathrm{model}}}$",
               "• Affine Transformation Applied to State $\\mathbf{z}_l$:",
               "   $\\tilde{\\mathbf{z}}_l = (1 + \\boldsymbol{\\gamma}_l) \\odot \\mathbf{z}_l + \\boldsymbol{\\beta}_l$",
               "• Zero Initialized for Identity Bias:",
               "   $\\mathbf{W}_{l,2} = \\mathbf{0}, \\mathbf{b}_{l,2} = \\mathbf{0} \\Rightarrow \\tilde{\\mathbf{z}}_l = \\mathbf{z}_l$ at init"],
              C_FILM_HDR, C_FILM_TXT, C_FILM_BG, C_FILM_BORDER, body_fs=5.6)

    # Connector from Speed Context to FiLM Generator (Horizontal arrow)
    ax.annotate("", xy=(40.5, 19.5), xytext=(19.0, 19.5),
                arrowprops=dict(arrowstyle="-|>", color="#D97706", lw=1.3, mutation_scale=9), zorder=8)
    # Badge on speed arrow - increased size and position
    sb = patches.FancyBboxPatch((23.0, 17.6), 12.5, 3.8, boxstyle="round,pad=0.0,rounding_size=0.6",
                                facecolor="#FEF3C7", edgecolor="#D97706", lw=0.6, zorder=9)
    ax.add_patch(sb)
    ax.text(29.25, 19.5, "Speed $s$ [B, 1]", ha="center", va="center", fontsize=5.8, fontweight="bold", color="#92400E", zorder=10)

    # Vertical arrow from FiLM Generator UP to FiLM Node
    ax.annotate("", xy=(56.0, 64.5), xytext=(56.0, 41.0),
                arrowprops=dict(arrowstyle="-|>", color="#EA580C", lw=1.3, mutation_scale=9), zorder=8)
    # Parameter badge on vertical arrow - increased size
    pb = patches.FancyBboxPatch((47.5, 43.5), 17.0, 3.9, boxstyle="round,pad=0.0,rounding_size=0.6",
                                facecolor="#FFEDD5", edgecolor="#EA580C", lw=0.6, zorder=9)
    ax.add_patch(pb)
    ax.text(56.0, 45.45, "$(\\boldsymbol{\\gamma}_l, \\boldsymbol{\\beta}_l) \\in \\mathbb{R}^{64 \\times 2}$",
            ha="center", va="center", fontsize=5.8, fontweight="bold", color="#9A3412", zorder=10)

    # ==========================================
    # 4. STAGE 4: PREDICTED JOINT KINETICS
    # ==========================================
    # Connector from S4D Backbone to Output
    ax.annotate("", xy=(78.5, 67.0), xytext=(76.0, 67.0),
                arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.3, mutation_scale=9), zorder=8)
    b3 = patches.FancyBboxPatch((75.5, 68.2), 3.3, 3.6, boxstyle="round,pad=0.0,rounding_size=0.6",
                                facecolor="#EDE9FE", edgecolor="#7C3AED", lw=0.6, zorder=9)
    ax.add_patch(b3)
    ax.text(77.15, 70.0, "64", ha="center", va="center", fontsize=5.8, fontweight="bold", color="#4C1D95", zorder=10)

    # Output Card with FIXED SPACING at top so waveform inset below is 100% clean
    draw_card(78.5, 48.0, 20.5, 49.0,
              "Predicted Joint Kinetics",
              ["• LayerNorm + Projection Head:",
               "   $\\mathbb{R}^{64} \\to \\mathbb{R}^3$ Sagittal Moments",
               "• Predicted Torques $\\hat{\\boldsymbol{\\tau}}(t)$:",
               "   Sagittal Kinetics (0–100% Gait):"],
              C_OUT_HDR, C_OUT_TXT, C_OUT_BG, C_OUT_BORDER, body_fs=5.7, fixed_spacing=4.5)

    # Miniature Waveform Inset placed perfectly inside Output Card below text
    wave_ax = ax.inset_axes([79.5, 49.5, 18.5, 19.5], transform=ax.transData)
    t_demo = np.linspace(0, 100, 100)
    hip_w = 0.8 * np.sin(2 * np.pi * t_demo / 100 + 0.8)
    knee_w = 0.5 * np.sin(2 * np.pi * t_demo / 100 - 0.5) - 0.2 * np.sin(4 * np.pi * t_demo / 100)
    ankle_w = 1.4 * np.exp(-((t_demo - 48)/12)**2) - 0.2
    
    wave_ax.plot(t_demo, hip_w, color="#2563EB", lw=1.2, label="Hip")
    wave_ax.plot(t_demo, knee_w, color="#16A34A", lw=1.2, label="Knee")
    wave_ax.plot(t_demo, ankle_w, color="#DC2626", lw=1.3, label="Ankle")
    wave_ax.set_xlim(0, 100)
    wave_ax.set_xticks([0, 50, 100])
    wave_ax.set_xticklabels(["0%", "50%", "100%"], fontsize=5.0)
    wave_ax.set_yticks([])
    wave_ax.grid(True, linestyle=":", alpha=0.5)
    wave_ax.legend(loc="upper right", fontsize=4.8, frameon=True, framealpha=0.88, edgecolor="#CBD5E1", ncol=3, handlelength=0.8, borderpad=0.2)
    for spine in wave_ax.spines.values():
        spine.set_color("#94A3B8")
        spine.set_linewidth(0.6)
    wave_ax.set_facecolor("#FFFFFF")

    # ==========================================
    # 5. STAGE 5: AUXILIARY PHYSICS REGULARIZATION
    # ==========================================
    draw_card(78.5, 5.0, 20.5, 36.0,
              "Physics Regularization",
              ["• Supervised Tracking Loss:",
               "   $\\mathcal{L}_{\\mathrm{task}} = \\mathrm{MSE}(\\hat{\\boldsymbol{\\tau}}, \\boldsymbol{\\tau}^*)$",
               "• Torque Jerk Penalty:",
               "   $\\mathcal{L}_{\\mathrm{jerk}} = \\frac{1}{3T}\\sum_{t,j} (\\frac{d^2 \\hat{\\tau}_j}{dt^2})^2$",
               "• Power Consistency Loss:",
               "   $\\mathcal{L}_{\\mathrm{power}} = \\frac{1}{3T}\\sum_{t,j} (\\hat{\\tau}_j \\dot{q}_j - P_j^*)^2$"],
              C_LOSS_HDR, C_LOSS_TXT, C_LOSS_BG, C_LOSS_BORDER, body_fs=5.6)

    # Connector from Output Moments straight DOWN to Physics Regularization
    ax.annotate("", xy=(88.75, 39.0), xytext=(88.75, 48.0),
                arrowprops=dict(arrowstyle="-|>", color="#DC2626", lw=1.3, linestyle="--", mutation_scale=9), zorder=8)
    ax.text(88.75, 43.5, "Multi-Task Loss $\\mathcal{L}$", ha="center", va="center",
            fontsize=5.8, fontweight="bold", color="#991B1B",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FEE2E2", ec="#DC2626", lw=0.6), zorder=9)

    fig.savefig(os.path.join(OUT_DIR, "fig1_architecture.pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(OUT_DIR, "fig1_architecture.png"), bbox_inches="tight", dpi=300, pad_inches=0.03)
    plt.close(fig)
    print("Generated Fig 1: Publication-Grade Architecture Diagram (fig1_architecture.pdf & .png)")


# ==============================================================================
# FIGURE 2: JOINT MOMENT WAVEFORMS (With Cohort Variance Envelopes)
# ==============================================================================
def generate_fig2_waveforms():
    device = torch.device("cpu")
    
    m1_path = os.path.join(repo_root, "experiments", "phase2_baseline", "model_best.pt")
    m2_path = os.path.join(repo_root, "experiments", "phase3_conditioning", "model_best.pt")
    m3_path = os.path.join(repo_root, "experiments", "phase3_conditioning_physics", "model_best.pt")

    model1 = GaitS4D(in_channels=102, out_channels=3, d_model=64, d_state=64, n_layers=4, dropout=0.1, use_film=False).to(device)
    model1.load_state_dict(torch.load(m1_path, map_location=device))
    model1.eval()

    model2 = GaitS4D(in_channels=102, out_channels=3, d_model=64, d_state=64, n_layers=4, dropout=0.1, use_film=True).to(device)
    model2.load_state_dict(torch.load(m2_path, map_location=device))
    model2.eval()

    model3 = GaitS4D(in_channels=102, out_channels=3, d_model=64, d_state=64, n_layers=4, dropout=0.1, use_film=True).to(device)
    model3.load_state_dict(torch.load(m3_path, map_location=device))
    model3.eval()

    # Collect all test trials across subjects for 0.8 and 1.2 m/s
    _, _, test_loader, _ = get_dataloaders(batch_size=1, split_mode="subject", include_muscles=True)
    
    gts_08, gts_12 = [], []
    preds1_08, preds2_08, preds3_08 = [], [], []
    preds1_12, preds2_12, preds3_12 = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            sp = float(batch["speed_raw"][0])
            gt = batch["y"].squeeze(0).numpy()
            p1 = model1(batch["x"]).squeeze(0).numpy()
            p2 = model2(batch["x"], batch["speed"]).squeeze(0).numpy()
            p3 = model3(batch["x"], batch["speed"]).squeeze(0).numpy()

            if abs(sp - 0.8) < 0.05:
                gts_08.append(gt)
                preds1_08.append(p1)
                preds2_08.append(p2)
                preds3_08.append(p3)
            elif abs(sp - 1.2) < 0.05:
                gts_12.append(gt)
                preds1_12.append(p1)
                preds2_12.append(p2)
                preds3_12.append(p3)

    gts_08 = np.array(gts_08) # (N_subj, 100, 3)
    gts_12 = np.array(gts_12) # (N_subj, 100, 3)

    time_pct = np.linspace(0, 100, 100)
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 2.95), sharex=True, dpi=300)
    joint_names = ["Hip Flexion/Extension", "Knee Flexion/Extension", "Ankle Plantar/Dorsiflexion"]
    speeds_data = [
        ("Slow Walking (0.8 m/s)", gts_08, preds1_08, preds2_08, preds3_08),
        ("Brisk Walking (1.2 m/s)", gts_12, preds1_12, preds2_12, preds3_12)
    ]

    for row_idx, (speed_label, gts, p1s, p2s, p3s) in enumerate(speeds_data):
        gt_mean = np.mean(gts, axis=0)
        gt_std = np.std(gts, axis=0)
        
        # Representative subject predictions (first subject)
        rep_p1 = p1s[0]
        rep_p2 = p2s[0]
        rep_p3 = p3s[0]
        rep_gt = gts[0]

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            
            # Shaded Cohort Standard Deviation Envelope
            ax.fill_between(time_pct, gt_mean[:, col_idx] - gt_std[:, col_idx], 
                            gt_mean[:, col_idx] + gt_std[:, col_idx], 
                            color=C_BAND, alpha=0.45, label="Cohort Ground Truth (±1 SD)")
            
            # Ground Truth line (Representative)
            ax.plot(time_pct, rep_gt[:, col_idx], color=C_GT, linestyle="--", linewidth=1.6, label="Ground Truth (Subject)", alpha=0.95)
            
            # Model Predictions
            ax.plot(time_pct, rep_p1[:, col_idx], color=C_M1, linestyle="-", linewidth=1.2, label="Model 1 (Baseline)", alpha=0.85)
            ax.plot(time_pct, rep_p2[:, col_idx], color=C_M2, linestyle="-", linewidth=1.2, label="Model 2 (+Cond.)", alpha=0.85)
            ax.plot(time_pct, rep_p3[:, col_idx], color=C_M3, linestyle="-", linewidth=1.5, label="Model 3 (+Cond.+Phys.)", alpha=0.95)
            
            ax.grid(True)
            if row_idx == 0:
                ax.set_title(joint_names[col_idx], fontsize=8.8, fontweight="bold", pad=3)
            if col_idx == 0:
                ax.set_ylabel(f"{speed_label}\nMoment (N·m/kg)", fontsize=8.0)
            if row_idx == 1:
                ax.set_xlabel("Gait Cycle (%)", fontsize=8.0)

    # Unified Legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5, frameon=True, fontsize=7.4)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig(os.path.join(OUT_DIR, "fig2_waveforms.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig2_waveforms.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Generated Fig 2: Waveforms with Cohort Variance Envelopes")


# ==============================================================================
# FIGURE 3: UNIFIED TWO-PANEL FIGURE (Jerk RMS + MAE Generalization Trend)
# ==============================================================================
def generate_fig3_generalization_and_jerk():
    with open("experiments/results/loso_summary.json", "r") as f:
        loso_data = json.load(f)

    speeds = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    speed_keys = ["0.6", "0.8", "1.0", "1.2", "1.4", "1.6"]
    x_labels = [f"{s}" for s in speeds] + ["Mean"]
    
    # Jerk data
    jerk_m1 = [loso_data[s]["baseline"]["pred_jerk_rms"] for s in speed_keys]
    jerk_m2 = [loso_data[s]["conditioning"]["pred_jerk_rms"] for s in speed_keys]
    jerk_m3 = [loso_data[s]["conditioning_physics"]["pred_jerk_rms"] for s in speed_keys]
    jerk_m1.append(float(np.mean(jerk_m1)))
    jerk_m2.append(float(np.mean(jerk_m2)))
    jerk_m3.append(float(np.mean(jerk_m3)))

    # MAE data
    mae_m1 = [loso_data[s]["baseline"]["test_overall_mae"] for s in speed_keys]
    mae_m2 = [loso_data[s]["conditioning"]["test_overall_mae"] for s in speed_keys]
    mae_m3 = [loso_data[s]["conditioning_physics"]["test_overall_mae"] for s in speed_keys]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.1, 2.7), dpi=300, gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.26})

    # --- Panel (a): Jerk RMS Bar Chart ---
    x = np.arange(len(x_labels))
    width = 0.27

    ax_a.bar(x - width, jerk_m1, width, label="Model 1 (Baseline)", color=C_M1, edgecolor="#1E3A8A", alpha=0.9, zorder=3)
    ax_a.bar(x, jerk_m2, width, label="Model 2 (+Cond.)", color=C_M2, edgecolor="#9A3412", alpha=0.9, zorder=3)
    ax_a.bar(x + width, jerk_m3, width, label="Model 3 (+Cond.+Phys.)", color=C_M3, edgecolor="#115E59", alpha=0.95, zorder=3)

    for i in range(len(x_labels)):
        red_pct = (1.0 - jerk_m3[i] / jerk_m1[i]) * 100.0
        ax_a.text(x[i] + width, jerk_m3[i] + 4.5, f"-{red_pct:.0f}%", ha="center", va="bottom",
                  fontsize=6.8, fontweight="bold", color="#0F766E")

    ax_a.set_ylabel("Torque Jerk RMS (N·m/(kg·s²))", fontsize=8.8)
    ax_a.set_xlabel("Held-Out Speed Fold (m/s)", fontsize=8.8)
    ax_a.set_title("(a) Torque Jerk RMS Across Folds", fontsize=8.6, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(x_labels)
    ax_a.set_ylim(0, 260)
    ax_a.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax_a.axvspan(len(x_labels) - 1.5, len(x_labels) - 0.5, color="#F1F5F9", alpha=0.6, zorder=1)
    ax_a.legend(loc="upper left", frameon=True, fontsize=7.4)

    # --- Panel (b): Cross-Speed MAE Generalization Trend ---
    ax_b.plot(speeds, mae_m1, marker="o", markersize=4.8, color=C_M1, linewidth=1.6, label="Model 1 (Mean: 0.0703)")
    ax_b.plot(speeds, mae_m2, marker="s", markersize=4.8, color=C_M2, linewidth=1.6, label="Model 2 (Mean: 0.0664, -5.5%)")
    ax_b.plot(speeds, mae_m3, marker="^", markersize=5.2, color=C_M3, linewidth=1.6, label="Model 3 (Mean: 0.0711)")

    # Callouts
    ax_b.annotate("0.6 m/s OOD:\nMAE -6.4%",
                  xy=(0.6, mae_m3[0]), xytext=(0.65, 0.082),
                  arrowprops=dict(arrowstyle="-|>", color="#0D9488", lw=0.9),
                  fontsize=7.2, fontweight="bold", color="#0F766E",
                  bbox=dict(boxstyle="round,pad=0.25", fc="#CCFBF1", ec="#14B8A6", alpha=0.85))

    ax_b.annotate("1.4 m/s Anomaly",
                  xy=(1.4, mae_m2[4]), xytext=(1.20, 0.085),
                  arrowprops=dict(arrowstyle="-|>", color="#EA580C", lw=0.9),
                  fontsize=7.2, fontweight="bold", color="#C2410C",
                  bbox=dict(boxstyle="round,pad=0.25", fc="#FFEDD5", ec="#F97316", alpha=0.85))

    ax_b.set_xlabel("Held-Out Walking Speed (m/s)", fontsize=8.8)
    ax_b.set_ylabel("Overall Joint Moment MAE (N·m/kg)", fontsize=8.8)
    ax_b.set_title("(b) Generalization Error vs. Speed", fontsize=8.6, fontweight="bold")
    ax_b.set_xticks(speeds)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.set_ylim(0.050, 0.100)
    ax_b.legend(loc="upper left", frameon=True, fontsize=7.4)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_generalization_and_jerk.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig3_generalization_and_jerk.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Generated Fig 3: Unified Two-Panel Generalization Figure (fig3_generalization_and_jerk.pdf)")


def generate_fig3_single_column():
    with open("experiments/results/loso_summary.json", "r") as f:
        loso_data = json.load(f)

    speeds = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    speed_keys = ["0.6", "0.8", "1.0", "1.2", "1.4", "1.6"]
    x_labels = [f"{s}" for s in speeds] + ["Mean"]
    
    jerk_m1 = [loso_data[s]["baseline"]["pred_jerk_rms"] for s in speed_keys]
    jerk_m2 = [loso_data[s]["conditioning"]["pred_jerk_rms"] for s in speed_keys]
    jerk_m3 = [loso_data[s]["conditioning_physics"]["pred_jerk_rms"] for s in speed_keys]
    jerk_m1.append(float(np.mean(jerk_m1)))
    jerk_m2.append(float(np.mean(jerk_m2)))
    jerk_m3.append(float(np.mean(jerk_m3)))

    mae_m1 = [loso_data[s]["baseline"]["test_overall_mae"] for s in speed_keys]
    mae_m2 = [loso_data[s]["conditioning"]["test_overall_mae"] for s in speed_keys]
    mae_m3 = [loso_data[s]["conditioning_physics"]["test_overall_mae"] for s in speed_keys]

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.5, 3.15), dpi=300)

    # --- Panel (a): Jerk RMS Bar Chart ---
    x = np.arange(len(x_labels))
    width = 0.27

    ax_a.bar(x - width, jerk_m1, width, label="M1 (Base)", color=C_M1, edgecolor="#1E3A8A", alpha=0.9, zorder=3)
    ax_a.bar(x, jerk_m2, width, label="M2 (+Cond)", color=C_M2, edgecolor="#9A3412", alpha=0.9, zorder=3)
    ax_a.bar(x + width, jerk_m3, width, label="M3 (+Phys)", color=C_M3, edgecolor="#115E59", alpha=0.95, zorder=3)

    for i in range(len(x_labels)):
        red_pct = (1.0 - jerk_m3[i] / jerk_m1[i]) * 100.0
        ax_a.text(x[i] + width, jerk_m3[i] + 4.0, f"-{red_pct:.0f}%", ha="center", va="bottom",
                  fontsize=5.8, fontweight="bold", color="#0F766E")

    ax_a.set_ylabel("Jerk RMS (N·m/(kg·s²))", fontsize=7.6)
    ax_a.set_title("(a) Torque Jerk Across LOSO Folds", fontsize=8.0, fontweight="bold", pad=2)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(x_labels, fontsize=7.0)
    ax_a.set_ylim(0, 265)
    ax_a.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax_a.legend(loc="upper left", frameon=True, fontsize=6.2, ncol=3)

    # --- Panel (b): Cross-Speed MAE Generalization Trend ---
    ax_b.plot(speeds, mae_m1, marker="o", markersize=4.0, color=C_M1, linewidth=1.3, label="M1 (0.0703)")
    ax_b.plot(speeds, mae_m2, marker="s", markersize=4.0, color=C_M2, linewidth=1.3, label="M2 (0.0664)")
    ax_b.plot(speeds, mae_m3, marker="^", markersize=4.5, color=C_M3, linewidth=1.3, label="M3 (0.0711)")

    ax_b.set_xlabel("Held-Out Speed (m/s)", fontsize=7.6)
    ax_b.set_ylabel("MAE (N·m/kg)", fontsize=7.6)
    ax_b.set_title("(b) Generalization Error vs. Speed", fontsize=8.0, fontweight="bold", pad=2)
    ax_b.set_xticks(speeds)
    ax_b.set_xticklabels([f"{s}" for s in speeds], fontsize=7.0)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.set_ylim(0.050, 0.100)
    ax_b.legend(loc="upper left", frameon=True, fontsize=6.2, ncol=3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_generalization_and_jerk_1col.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig3_generalization_and_jerk_1col.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Generated Fig 3 Single-Column: fig3_generalization_and_jerk_1col.pdf")


if __name__ == "__main__":
    generate_fig1_architecture()
    generate_fig2_waveforms()
    generate_fig3_generalization_and_jerk()
    generate_fig3_single_column()
    print("All Revision 2 figures generated successfully in paper/figures/")

